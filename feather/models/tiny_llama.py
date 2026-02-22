from dataclasses import dataclass
from typing import List, Dict, Tuple
import torch

from feather.packers.fp8 import pack_fp8_tensor, unpack_fp8_tensor
from feather.routines.attention import *
from feather.routines.gemv import *
from feather.routines.misc import *
from feather.routines.utils import _pack_fp32_to_e4m3, pack_tensor_gpu


@dataclass
class FeatherTinyLlamaConfig:
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    intermediate_size: int = 11008
    vocab_size: int = 32000
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    kv_block_size: int = 16
    max_num_blocks: int = 512
    num_heads_per_chunk: int = 16


class FeatherTinyLlama:
    def __init__(
        self,
        config: FeatherTinyLlamaConfig,
        packed_weights: Dict[str, torch.Tensor],
        device: torch.device = torch.device("cuda"),
    ):
        self.config = config
        self.w = self.w = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in packed_weights.items()
        }
        self.device = device

        self.head_dim = config.hidden_size // config.num_attention_heads
        self.head_dim_packed = self.head_dim // 4

        self._build_kv_cache()
        self._rope_cos, self._rope_sin = self._build_rope_tables()

        self.scale_acc = torch.zeros(size=(1,), dtype=torch.float32, device=self.device)

        hs_packed = self.config.hidden_size // 4
        int_packed = self.config.intermediate_size // 4
        kv_packed = (self.config.num_key_value_heads * self.head_dim) // 4

        self.buf_q = torch.empty(hs_packed, dtype=torch.int32, device=self.device)
        self.buf_k = torch.empty(kv_packed, dtype=torch.int32, device=self.device)
        self.buf_v = torch.empty(kv_packed, dtype=torch.int32, device=self.device)
        self.buf_o = torch.empty(hs_packed, dtype=torch.int32, device=self.device)
        self.buf_gate = torch.empty(int_packed, dtype=torch.int32, device=self.device)
        self.buf_up = torch.empty(int_packed, dtype=torch.int32, device=self.device)
        self.buf_down = torch.empty(hs_packed, dtype=torch.int32, device=self.device)
        self.buf_norm = torch.empty(hs_packed, dtype=torch.int32, device=self.device)
        self.buf_norm1 = torch.empty(size=(1,), dtype=torch.float32, device=self.device)
        self.buf_pack = torch.empty(
            self.config.hidden_size // 4, dtype=torch.int32, device=self.device
        )

    def _build_kv_cache(self):
        """
        Utility function to build kv cache structure
        """
        shape = (
            self.config.max_num_blocks,
            self.config.num_attention_heads,
            self.head_dim_packed,
            self.config.kv_block_size,
        )
        self.k_cache = [
            torch.zeros(shape, dtype=torch.int32, device=self.device)
            for _ in range(self.config.num_hidden_layers)
        ]
        self.v_cache = [
            torch.zeros(shape, dtype=torch.int32, device=self.device)
            for _ in range(self.config.num_hidden_layers)
        ]
        max_lb = self.config.max_position_embeddings // self.config.kv_block_size + 1
        self.block_table = torch.arange(
            max_lb, dtype=torch.int32, device=self.device
        ).unsqueeze(0)
        self.context_lens = torch.zeros((1,), dtype=torch.int32, device=self.device)

    def _insert_kv(
        self, layer: int, pos: int, k_packed: torch.Tensor, v_packed: torch.Tensor
    ):
        """
        Utility function to insert items into kv cache
        """
        block_idx = pos // self.config.kv_block_size
        slot = pos % self.config.kv_block_size
        phys = block_idx
        groups = self.config.num_attention_heads // self.config.num_key_value_heads

        k_expanded = (
            k_packed.unsqueeze(1)
            .expand(-1, groups, -1)
            .reshape(-1, self.head_dim_packed)
        )
        v_expanded = (
            v_packed.unsqueeze(1)
            .expand(-1, groups, -1)
            .reshape(-1, self.head_dim_packed)
        )

        self.k_cache[layer][phys, :, :, slot] = k_expanded
        self.v_cache[layer][phys, :, :, slot] = v_expanded

    def _build_rope_tables(self):
        """
        Utility function to construct rope table
        """
        inv_freq = 1.0 / (
            self.config.rope_theta
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        pos = torch.arange(self.config.max_position_embeddings, dtype=torch.float32)
        emb = torch.cat([torch.outer(pos, inv_freq)] * 2, dim=-1)
        return emb.cos().to(self.device), emb.sin().to(self.device)

    def reset_kv_cache(self):
        """
        Reset's KV cache
        """
        for i in range(self.config.num_hidden_layers):
            self.k_cache[i].zero_()
            self.v_cache[i].zero_()
        self.context_lens.zero_()

    def _norm(
        self, x: torch.Tensor, w: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        return rms_norm_fp8_e4m3_out_packed_gpu(
            x=x,
            w=w,
            n=x.numel(),
            scale=scale,
            eps=self.config.rms_norm_eps,
            out=self.buf_norm,
            norm=self.buf_norm1,
        )

    def _projection(
        self,
        m: torch.Tensor,
        v: torch.Tensor,
        original_shape: torch.Tensor,
        scale: torch.Tensor,
        out: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # accumulator scale
        self.scale_acc.zero_()
        projection = gemv_fp8_e4m3_out_packed_gpu(
            m=m,
            v=v,
            m_shape=original_shape.tolist(),
            scale_acc=self.scale_acc,
            scale_w=scale,
            out=out,
        )

        return projection, self.scale_acc

    def _attention(
        self, x: torch.Tensor, layer: int, pos: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        n = f"model.layers.{layer}.self_attn"

        w_q_projection_key = f"{n}.q_proj.weight"
        w_k_projection_key = f"{n}.k_proj.weight"
        w_v_projection_key = f"{n}.v_proj.weight"
        w_o_projection_key = f"{n}.o_proj.weight"

        w_q_projection_original_shape_key = f"{n}.q_proj.weight" + "_original_shape"
        w_k_projection_original_shape_key = f"{n}.k_proj.weight" + "_original_shape"
        w_v_projection_original_shape_key = f"{n}.v_proj.weight" + "_original_shape"
        w_o_projection_original_shape_key = f"{n}.o_proj.weight" + "_original_shape"

        w_q_projection_scale_key = f"{n}.q_proj.weight" + "_scale"
        w_k_projection_scale_key = f"{n}.k_proj.weight" + "_scale"
        w_v_projection_scale_key = f"{n}.v_proj.weight" + "_scale"
        w_o_projection_scale_key = f"{n}.o_proj.weight" + "_scale"

        # q projection
        q, _ = self._projection(
            m=self.w[w_q_projection_key],
            v=x,
            original_shape=self.w[w_q_projection_original_shape_key],
            scale=self.w[w_q_projection_scale_key],
            out=self.buf_q,
        )
        q = q.view(self.config.num_attention_heads, self.head_dim_packed)

        # k projection
        k, _ = self._projection(
            m=self.w[w_k_projection_key],
            v=x,
            original_shape=self.w[w_k_projection_original_shape_key],
            scale=self.w[w_k_projection_scale_key],
            out=self.buf_k,
        )
        k = k.view(self.config.num_key_value_heads, self.head_dim_packed)

        # v projection
        v, _ = self._projection(
            m=self.w[w_v_projection_key],
            v=x,
            original_shape=self.w[w_v_projection_original_shape_key],
            scale=self.w[w_v_projection_scale_key],
            out=self.buf_v,
        )
        v = v.view(self.config.num_key_value_heads, self.head_dim_packed)

        # insert positional embeddings
        rope_fp8_e4m3_inplace_gpu(
            q, self._rope_cos[pos], self._rope_sin[pos], self.head_dim_packed
        )
        rope_fp8_e4m3_inplace_gpu(
            k, self._rope_cos[pos], self._rope_sin[pos], self.head_dim_packed
        )

        # insert into kv cache table
        self._insert_kv(layer, pos, k, v)
        self.context_lens[0] = pos + 1

        # attend
        out = paged_attention_fp8_e4m3_acc_fp32_gpu(
            q=q.unsqueeze(0),
            k_cache=self.k_cache[layer],
            v_cache=self.v_cache[layer],
            block_table=self.block_table,
            context_lens=self.context_lens,
            h_dim=self.head_dim_packed,
            num_heads_per_chunk=self.config.num_heads_per_chunk,
        )

        out = out.view(-1)
        out_packed = pack_tensor_gpu(out, out=self.buf_pack)

        # output projection
        o, o_scale = self._projection(
            m=self.w[w_o_projection_key],
            v=out_packed,
            original_shape=self.w[w_o_projection_original_shape_key],
            scale=self.w[w_o_projection_scale_key],
            out=self.buf_o,
        )
        return o, o_scale

    def _mlp(self, x: torch.Tensor, layer: int) -> Tuple[torch.Tensor, torch.Tensor]:
        n = f"model.layers.{layer}.mlp"

        w_gate_projection_key = f"{n}.gate_proj.weight"
        w_up_projection_key = f"{n}.up_proj.weight"
        w_down_projection_key = f"{n}.down_proj.weight"

        w_gate_projection_original_shape_key = (
            f"{n}.gate_proj.weight" + "_original_shape"
        )
        w_up_projection_original_shape_key = f"{n}.up_proj.weight" + "_original_shape"
        w_down_projection_original_shape_key = (
            f"{n}.down_proj.weight" + "_original_shape"
        )

        w_gate_projection_scale_key = f"{n}.gate_proj.weight" + "_scale"
        w_up_projection_scale_key = f"{n}.up_proj.weight" + "_scale"
        w_down_projection_scale_key = f"{n}.down_proj.weight" + "_scale"

        gate, _ = self._projection(
            m=self.w[w_gate_projection_key],
            v=x,
            original_shape=self.w[w_gate_projection_original_shape_key],
            scale=self.w[w_gate_projection_scale_key],
            out=self.buf_gate,
        )
        up, _ = self._projection(
            m=self.w[w_up_projection_key],
            v=x,
            original_shape=self.w[w_up_projection_original_shape_key],
            scale=self.w[w_up_projection_scale_key],
            out=self.buf_up,
        )

        activation = swiglu_fp8_e4m3_packed_gpu(gate, up, gate.numel())

        down, down_scale = self._projection(
            m=self.w[w_down_projection_key],
            v=activation,
            original_shape=self.w[w_down_projection_original_shape_key],
            scale=self.w[w_down_projection_scale_key],
            out=self.buf_down,
        )

        return down, down_scale

    def forward(self, token_id: int, position_id: int) -> torch.Tensor:

        x_fp16 = self.w["model.embed_tokens.weight"][token_id].to(torch.float16)
        x_packed = pack_tensor_gpu(x_fp16, out=self.buf_pack)

        norm_weight = self.w["model.norm.weight"]
        norm_weight_scale = self.w["model.norm.weight" + "_scale"]
        lm_weight = self.w["lm_head.weight"].to(dtype=torch.float32, device=self.device)

        for i in range(self.config.num_hidden_layers):
            pre_attn_norm = self.w[f"model.layers.{i}.input_layernorm.weight"]
            pre_attn_norm_scale = self.w[
                f"model.layers.{i}.input_layernorm.weight" + "_scale"
            ]
            post_attn_norm = self.w[f"model.layers.{i}.post_attention_layernorm.weight"]
            post_attn_norm_scale = self.w[
                f"model.layers.{i}.post_attention_layernorm.weight" + "_scale"
            ]
            o_packed, o_scale = self._attention(
                self._norm(x_packed, pre_attn_norm, pre_attn_norm_scale),
                i,
                position_id,
            )
            x_fp16, x_packed = fused_add_e4m3_acc_fp32_dual_out_gpu(
                a_original=x_fp16, b_packed=o_packed, b_packed_scale=o_scale
            )
            down_packed, down_scale = self._mlp(
                self._norm(x_packed, post_attn_norm, post_attn_norm_scale),
                i,
            )
            x_fp16, x_packed = fused_add_e4m3_acc_fp32_dual_out_gpu(
                a_original=x_fp16, b_packed=down_packed, b_packed_scale=down_scale
            )

        final_norm = self._norm(x_packed, norm_weight, norm_weight_scale)

        # custom converstion logic unpack_fp8_tensor isn't working as expected :(

        u_8 = final_norm.view(torch.uint8).to(torch.int32)
        sign = torch.where((u_8 & 0x80) != 0, -1.0, 1.0)
        exp = (u_8 & 0x78) >> 3
        mant = (u_8 & 0x07).to(torch.float32)
        norm_val = sign * (2.0 ** (exp - 7.0)) * (1.0 + mant / 8.0)
        sub_val = sign * mant * 0.001953125

        xf = torch.where(exp > 0, norm_val, sub_val).to(torch.float32).to(self.device)

        return torch.mv(lm_weight, xf)

    def generate(
        self,
        prompt_ids: List[int],
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
        penalty_window: int = 64,
        eos_token_id: int = 2,
    ) -> List[int]:
        import torch.nn.functional as F

        self.reset_kv_cache()
        tokens = list(prompt_ids)
        logits = None
        for pos, tok in enumerate(tokens):
            logits = self.forward(tok, pos)

        for _ in range(max_new_tokens):
            if repetition_penalty != 1.0:
                recent_tokens = set(tokens[-penalty_window:])
                for tok in recent_tokens:
                    if logits[tok] > 0:
                        logits[tok] /= repetition_penalty
                    else:
                        logits[tok] *= repetition_penalty

            if temperature == 0.0:
                next_tok = int(logits.argmax())
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                if top_p < 1.0:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                    cumulative = torch.cumsum(sorted_probs, dim=0)
                    sorted_probs[cumulative - sorted_probs > top_p] = 0.0
                    probs = torch.zeros_like(probs).scatter_(
                        0, sorted_idx, sorted_probs
                    )
                    probs /= probs.sum()
                next_tok = int(torch.multinomial(probs, num_samples=1))

            tokens.append(next_tok)
            if next_tok == eos_token_id:
                break

            logits = self.forward(next_tok, len(tokens) - 1)

        return tokens
