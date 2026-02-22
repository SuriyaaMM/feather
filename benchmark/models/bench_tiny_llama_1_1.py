from feather.models.tiny_llama import *
from feather.models.pack_weights import *
from transformers import AutoTokenizer
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

packed_weights = torch.load("tinyllama_fp8.pt")

cfg = FeatherTinyLlamaConfig(
    hidden_size=2048,
    num_hidden_layers=22,
    num_attention_heads=32,
    num_key_value_heads=4,
    intermediate_size=5632,
    vocab_size=32000,
    max_position_embeddings=2048,
    kv_block_size=16,
    max_num_blocks=256,
    num_heads_per_chunk=1,
)

model = FeatherTinyLlama(cfg, packed_weights, device="cuda")
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "tell me about unuited states of america"},
]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
ids = tokenizer.encode(prompt)

total_time = 0.0

for i in range(25):
    model.reset_kv_cache()
    begin = time.perf_counter()
    with torch.inference_mode():
        out_ids = model.generate(
            ids, max_new_tokens=256, temperature=0.0, top_p=0.9, repetition_penalty=1.15
        )
    end = time.perf_counter()
    total_time += end - begin

print(f"feather elapsed time = {total_time / 25}")

del model
torch.cuda.empty_cache()

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float32,
    device_map="cuda",
    attn_implementation="sdpa",
    use_cache=True,
).eval()

input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
gen_kwargs = {
    "max_new_tokens": 256,
    "temperature": 0.0,
    "top_p": 0.9,
    "repetition_penalty": 1.15,
    "do_sample": False,
    "use_cache": True,
    "pad_token_id": tokenizer.eos_token_id,
}

total_time = 0.0
for i in range(25):
    start = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(input_ids, **gen_kwargs)
    end = time.perf_counter()
    total_time += end - start

print(f"hf fp32 elapsed time = {total_time / 25} seconds")
