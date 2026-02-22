import torch
from transformers import LlamaForCausalLM
import logging

from typing import Dict
from feather.packers.fp8 import pack_fp8_tensor


def pack_llama_weights(
    hf_model: str,
    savefile: str,
    dtype: torch.dtype = torch.float16,
    mode: str = "E4M3",
) -> Dict[str, torch.Tensor]:

    preserve: set[str] = {"model.embed_tokens.weight", "lm_head.weight"}
    gemv_suffixes: set[str] = {
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    }
    norm_suffixes: set[str] = {
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "norm.weight",
    }

    model = LlamaForCausalLM.from_pretrained(
        hf_model,
        dtype=dtype,
    )

    packed_weights: Dict[str, torch.Tensor] = {}

    for key, tensor in model.state_dict().items():
        tensor: torch.Tensor
        key: str

        if key in preserve:
            packed_weights[key] = tensor.to(dtype=dtype)
            logging.info(f"preserved {key} in [{dtype}]")

        # projection weights
        elif any(key.endswith(s) for s in gemv_suffixes):
            # per tensor scaling
            w_original = tensor.to(dtype=dtype)

            # still i am unclear why this works only when scaled for E5M2
            # and works with scale = 1.0 for E4M3
            if "mlp" in key:
                w_normed = w_original
                w_scale = torch.tensor(1.0, dtype=torch.float32)
            else:
                w_max = w_original.abs().max()
                w_scale = w_max if w_max > 0 else torch.tensor(1.0, dtype=dtype)
                w_normed = w_original / w_scale

            original_shape = tuple(w_normed.shape)

            # pack the weights
            w_packed = pack_fp8_tensor(w_normed, mode=mode).to("cuda")

            # reshape flat vector
            w_packed = w_packed.view(
                original_shape[0], original_shape[1] // 4
            ).contiguous()

            # write to the dictionary
            packed_weights[key] = w_packed
            packed_weights[key + "_original_shape"] = torch.tensor(
                list(original_shape), dtype=torch.int64
            )
            packed_weights[key + "_scale"] = w_scale.to(torch.float32).to("cuda")

            logging.info(
                f"packed {key} in [FP8{mode}] | original_shape : {original_shape} | scale : {w_scale}"
            )

        # norm weights
        elif any(key.endswith(s) for s in norm_suffixes):
            # per tensor scaling
            w_original = tensor.to(dtype=dtype)
            w_max = w_original.abs().max()
            w_scale = w_max if w_max > 0 else torch.tensor(1.0, dtype=dtype)

            # normalise
            w_norm = (w_original / w_scale).contiguous()
            w_packed = pack_fp8_tensor(w_norm, mode=mode).to("cuda")

            # write to the dictionary
            packed_weights[key] = w_packed.contiguous()
            packed_weights[key + "_scale"] = w_scale.to(torch.float32).to("cuda")

            logging.info(f"packed {key} in [FP8{mode}] | scale : {w_scale}")

        # scalars
        elif tensor.ndim == 1:
            packed_weights[key] = tensor.to(dtype=dtype).to("cuda")
            logging.info(f"packed {key} in [{dtype}]")

        else:
            packed_weights[key] = tensor.to(dtype=dtype).to("cuda")
            logging.info(f"packed {key} in [{dtype}]")

    torch.save(packed_weights, savefile)
    logging.info(f"saved packed tensor to {savefile}")

    return packed_weights
