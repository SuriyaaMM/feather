import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, LlamaForCausalLM
from datasets import load_dataset
import math
from tqdm import tqdm

from feather.models.tiny_llama import FeatherTinyLlama, FeatherTinyLlamaConfig


def get_wikitext2_tokens(tokenizer, max_tokens=4096):
    test_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(test_data["text"])
    encodings = tokenizer(
        text, return_tensors="pt", max_length=max_tokens, truncation=True
    )
    input_ids = encodings.input_ids[0]

    return input_ids.to("cuda")


def evaluate_hf_ppl(model, input_ids):

    model.eval()
    nlls = []
    seq_len = 2048
    with torch.no_grad():
        for i in tqdm(range(0, len(input_ids) - 1, seq_len)):
            chunk = input_ids[i : i + seq_len].unsqueeze(0)
            target = chunk.clone()

            outputs = model(chunk, labels=target)
            neg_log_likelihood = outputs.loss
            nlls.append(neg_log_likelihood * chunk.size(1))

    avg_nll = torch.stack(nlls).sum() / len(input_ids)
    ppl = torch.exp(avg_nll).item()
    return ppl


def evaluate_feather_ppl(model, input_ids):
    nlls = []

    seq_len = 2048
    with torch.inference_mode():
        for i in tqdm(range(0, len(input_ids) - 1, seq_len)):
            chunk = input_ids[i : i + seq_len]
            model.reset_kv_cache()
            chunk_nlls = []
            for pos in range(len(chunk) - 1):
                token = chunk[pos].item()
                target = chunk[pos + 1].clone().detach()
                logits = model.forward(token, pos)
                loss = F.cross_entropy(logits, target)
                chunk_nlls.append(loss)
            nlls.extend(chunk_nlls)
    avg_nll = torch.stack(nlls).mean()
    ppl = torch.exp(avg_nll).item()
    return ppl


def main():
    device = "cuda"

    # perplexity for huggingface
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    input_ids = get_wikitext2_tokens(tokenizer, max_tokens=4096)

    hf_model = LlamaForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.float16,
        device_map=device,
    )
    hf_ppl = evaluate_hf_ppl(hf_model, input_ids)

    del hf_model
    torch.cuda.empty_cache()

    # perpexity for feather
    packed_weights = torch.load("tinyllama_fp8.pt", weights_only=False)
    cfg = FeatherTinyLlamaConfig(
        hidden_size=2048,
        num_hidden_layers=22,
        num_attention_heads=32,
        num_key_value_heads=4,
        intermediate_size=5632,
        vocab_size=32768,
        max_position_embeddings=2048,
        kv_block_size=16,
        max_num_blocks=256,
        num_heads_per_chunk=1,
    )
    feather_model = FeatherTinyLlama(cfg, packed_weights, device=device)

    feather_ppl = evaluate_feather_ppl(feather_model, input_ids)

    print(f"feather perplexity = {feather_ppl}")
    print(f"hf perplexity = {hf_ppl}")


if __name__ == "__main__":
    main()
