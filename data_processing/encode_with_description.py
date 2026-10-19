

import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from tqdm import tqdm
import argparse


parse = argparse.ArgumentParser()
parse.add_argument("--dataset_name", type=str, default="Movie_noise_user0.1")
parse.add_argument("--batch_size", type=int, default=16)
parse.add_argument("--base_model", type=str, default="models/llama3_8B_Instruct_sft_denoised_merged")
args = parse.parse_args()

base_model = args.base_model


print(f"📦 Loading model from {base_model}...")
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16
).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(base_model)


if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id


tokenizer.padding_side = "left"
model.eval()


hidden_size = model.config.hidden_size
print(f"✅ Model loaded, hidden size: {hidden_size}")


save_item_embedding_path = os.path.join("./data/", args.dataset_name, "item_embedding_with_desc.pt")
id2desc_path = os.path.join("./data/", args.dataset_name, "id2title_description.json")


print(f"📖 Loading item descriptions from {id2desc_path}...")
with open(id2desc_path, "r", encoding="utf-8") as f:
    id2desc = json.load(f)


item_cids = [int(k) for k in id2desc.keys()]
item_descriptions = [v.strip() for v in id2desc.values()]  
item_max_id = max(item_cids)

print(f"📊 Total items: {len(item_cids)}, max ID: {item_max_id}")


def batch(list, batch_size):
    chunk_size = (len(list) - 1) // batch_size + 1
    for i in range(chunk_size):
        yield list[batch_size * i : batch_size * (i + 1)]


print(f"🚀 Generating embeddings with batch_size={args.batch_size}...")
with torch.no_grad():
    all_embeddings = torch.zeros((item_max_id + 1, hidden_size), device="cuda", dtype=torch.bfloat16)
    
    pbar = tqdm(
        zip(batch(item_cids, args.batch_size), batch(item_descriptions, args.batch_size)),
        total=len(item_descriptions) // args.batch_size + 1,
        desc="Encoding items"
    )
    
    for batch_cid, batch_text in pbar:
        batch_cid = torch.tensor(batch_cid, device="cuda")
        
        
        inputs = tokenizer(
            batch_text, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=256  
        ).to("cuda")
        
        
        outputs = model(
            inputs.input_ids, 
            attention_mask=inputs.attention_mask, 
            output_hidden_states=True
        )
        hidden_states = outputs.hidden_states
        
        
        all_embeddings[batch_cid] = hidden_states[-1][:, -1, :].detach()


print("🔍 Checking for NaN values...")
has_nan = torch.isnan(all_embeddings).any().item()
if has_nan:
    print("❌ Error: embeddings contain NaN values.")
    exit(-1)


print(f"💾 Saving embeddings to {save_item_embedding_path}...")
print(f"   Shape: {all_embeddings.shape}")
torch.save(all_embeddings, save_item_embedding_path)

print("✅ Done! Item embeddings with descriptions saved successfully.")
print(f"\n📝 Summary:")
print(f"   - Model: {base_model}")
print(f"   - Items: {len(item_cids)}")
print(f"   - Embedding dimension: {hidden_size}")
print(f"   - Format: Genre - Description")
print(f"   - Output: {save_item_embedding_path}")
