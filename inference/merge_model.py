from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import fire

def merge_lora_to_base(
    base_model_path: str,
    lora_model_path: str,
    output_model_path: str,
    cuda_device: str = "6"
):
    
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    print(f">>> Using GPU device: {cuda_device}")

    
    print(f">>> Loading tokenizer from {base_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    
    print(f">>> Loading base model from {base_model_path}...")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True
    )

    
    print(f">>> Loading LoRA weights from {lora_model_path}...")
    model = PeftModel.from_pretrained(base, lora_model_path)
    
    print(">>> Merging weights... this might take a moment.")
    merged_model = model.merge_and_unload()

    
    print(f">>> Saving merged model to {output_model_path}...")
    merged_model.save_pretrained(output_model_path, safe_serialization=True)
    tokenizer.save_pretrained(output_model_path)
    
    print(">>> [Success] Merge completed and model saved!")

if __name__ == "__main__":
    fire.Fire(merge_lora_to_base)
