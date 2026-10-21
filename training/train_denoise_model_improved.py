import re
import os
import random
from typing import List, Optional

import fire
import numpy as np
import torch
import transformers
from datasets import load_dataset
from accelerate import Accelerator

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, EarlyStoppingCallback


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


class CustomTrainer(transformers.Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.model.save_pretrained(output_dir, save_embedding_layers=False)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))


def train(
    dataset_name: str,
    manual_noise: float = -1,
    base_model: str = "models/Qwen2.5-3B-Instruct",
    sample: int = -1,
    seed: int = 42,
    
    batch_size: int = 64,
    micro_batch_size: int = 8,
    num_epochs: int = 3,
    learning_rate: float = 5e-4,
    cutoff_len: int = 256,
    
    lora_r: int = 32,  
    lora_alpha: int = 64,  
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [  
        "q_proj",
        "k_proj",  
        "v_proj",
        "o_proj",  
        "gate_proj",  
        "up_proj",  
        "down_proj", 
    ],
    
    warmup_ratio: float = 0.03,  
    weight_decay: float = 0.01,  
    val_ratio: float = 0.05,  
    early_stopping_patience: int = 10,  
):
    
    seed = int(seed)
    sample = int(sample)
    batch_size = int(batch_size)
    micro_batch_size = int(micro_batch_size)
    num_epochs = int(num_epochs)
    learning_rate = float(learning_rate)
    cutoff_len = int(cutoff_len)
    lora_r = int(lora_r)
    lora_alpha = int(lora_alpha)
    lora_dropout = float(lora_dropout)
    warmup_ratio = float(warmup_ratio)
    weight_decay = float(weight_decay)
    val_ratio = float(val_ratio)
    early_stopping_patience = int(early_stopping_patience)
    manual_noise = float(manual_noise)
    
    set_seed(seed)
    
    
    if manual_noise != -1:
        train_data_path = os.path.join(
            "./data", dataset_name, f"TDMD_noise_{manual_noise}", "train.json"
        )
        output_dir = os.path.join(
            "./save_denoise_model",
            dataset_name,
            f"improved_r{lora_r}_batch{batch_size}_sample{sample}_epoch{num_epochs}_manualnoise{manual_noise}",
        )
    else:
        
        if "TDMD" in dataset_name:
            
            train_data_path = os.path.join("./data", dataset_name, "train_with_description.json")
        else:
            
            train_data_path = os.path.join("./data", dataset_name, f"TDMD", "train_with_description.json")
        output_dir = os.path.join(
            "./save_denoise_model",
            dataset_name,
            f"improved_r{lora_r}_batch{batch_size}_sample{sample}_epoch{num_epochs}",
        )

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    micro_batch_size = batch_size // world_size
    gradient_accumulation_steps = batch_size // micro_batch_size

    if world_size != 1:
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    print("="*70)
    print("Training Configuration:")
    print("="*70)
    print(f"Base Model: {base_model}")
    print(f"Dataset: {dataset_name}")
    print(f"Sample Count: {sample if sample > 0 else 'All'}")
    print(f"LoRA Rank: {lora_r} (Alpha: {lora_alpha})")
    print(f"Target Modules: {', '.join(lora_target_modules)}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Batch Size: {batch_size} (Micro: {micro_batch_size}, Accum: {gradient_accumulation_steps})")
    print(f"Epochs: {num_epochs}")
    print(f"Warmup Ratio: {warmup_ratio}")
    print(f"Weight Decay: {weight_decay}")
    print(f"Validation Set Ratio: {val_ratio}")
    print(f"Early Stopping Patience: {early_stopping_patience}")
    print(f"Output Directory: {output_dir}")
    print("="*70)
    
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = "left"

    
    model.gradient_checkpointing_enable()
    
    
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable Parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)\n")

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(data_point):
        
        full_prompt = generate_prompt(data_point)
        user_prompt = generate_user_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt, add_eos_token=True)
        
        tokenized_user_prompt = tokenize(user_prompt, add_eos_token=True)
        user_prompt_len = len(tokenized_user_prompt["input_ids"])
        if user_prompt_len > len(tokenized_full_prompt["labels"]):
            user_prompt_len = len(tokenized_full_prompt["labels"])

        for i in range(user_prompt_len):
            tokenized_full_prompt["labels"][i] = -100

        return tokenized_full_prompt

    
    full_data = load_dataset("json", data_files=train_data_path, split="train")
    full_data = full_data.shuffle(seed=seed)
    
    if sample > -1:
        full_data = full_data.select(range(min(sample, len(full_data))))
    
    
    if val_ratio > 0:
        split_data = full_data.train_test_split(test_size=val_ratio, seed=seed)
        train_data = split_data["train"]
        val_data = split_data["test"]
        print(f"Training Set Size: {len(train_data)}")
        print(f"Validation Set Size: {len(val_data)}")
    else:
        train_data = full_data
        val_data = None
        print(f"Training Set Size: {len(train_data)}")
    
    train_data = train_data.map(lambda x: generate_and_tokenize_prompt(x))
    if val_data is not None:
        val_data = val_data.map(lambda x: generate_and_tokenize_prompt(x))
    
    print("\n=== Checking Masking Logic ===")
    sample_item = train_data[0]
    input_ids = sample_item["input_ids"]
    labels = sample_item["labels"]
    
    
    print("Input Text (First 50 tokens):", tokenizer.decode(input_ids[:50]))
    
    
    first_label_idx = 0
    for idx, lbl in enumerate(labels):
        if lbl != -100:
            first_label_idx = idx
            break
            
    print(f"Mask Length: {first_label_idx}")
    print("Text Before Mask Boundary:", tokenizer.decode(input_ids[max(0, first_label_idx-5):first_label_idx]))
    print("Text Where Loss Calculation Starts:", tokenizer.decode(input_ids[first_label_idx:first_label_idx+5]))
    print("=========================\n")
    
    
    callbacks = []
    if val_data and early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)
        )
        print(f"\nEarly stopping enabled, patience={early_stopping_patience}")
    
    
    trainer = CustomTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,  
        callbacks=callbacks if callbacks else None,
        args=transformers.TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=warmup_ratio,  
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            bf16=True,
            tf32=True,
            optim="adamw_torch",
            weight_decay=weight_decay,  
            logging_strategy="steps",
            logging_steps=50,  
            evaluation_strategy="steps" if val_data else "no",  
            eval_steps=100 if val_data else None,  
            save_strategy="steps",
            save_steps=100 if val_data else (1 / num_epochs),  
            save_total_limit=5,  
            save_on_each_node=False,
            log_on_each_node=False,
            load_best_model_at_end=True if val_data else False,  
            metric_for_best_model="eval_loss" if val_data else None,  
            greater_is_better=False,  
            ddp_find_unused_parameters=False if (world_size != 1) else None,
            report_to="tensorboard",
            ddp_backend="nccl",
            local_rank=int(os.environ.get("LOCAL_RANK", -1)),
            seed=seed,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    
    model.config.use_cache = False
    
    print("\nStarting training...")
    trainer.train()
    
    print(f"\nTraining completed! Model saved to: {output_dir}")
    model.save_pretrained(output_dir, save_embedding_layers=False)
    
    
    if trainer.state.log_history:
        import json
        summary_path = os.path.join(output_dir, "training_summary.json")
        with open(summary_path, 'w') as f:
            json.dump({
                "config": {
                    "lora_r": lora_r,
                    "lora_alpha": lora_alpha,
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "num_epochs": num_epochs,
                    "target_modules": lora_target_modules,
                },
                "final_train_loss": trainer.state.log_history[-1].get("loss", None),
                "final_eval_loss": trainer.state.log_history[-1].get("eval_loss", None) if val_data else None,
                "total_steps": trainer.state.global_step,
            }, f, indent=4)
        print(f"Training summary saved to: {summary_path}")


def generate_prompt(data_point):
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
{data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

### Instruction:
{data_point["instruction"]}

### Response:
{data_point["output"]}"""


def generate_user_prompt(data_point):
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

### Instruction:
{data_point["instruction"]}

### Response:
"""


if __name__ == "__main__":
    fire.Fire(train)
