import re
import json
import torch
import fire
from typing import List
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer


from sasrec_reward import (
    SASRecRewardCalculator, 
    sasrec_improvement_reward_func,
    sasrec_ranking_improvement_reward_func
)


from improved_reward_funcs import (
    shaped_f1_reward_func,
    denoising_action_reward_func,
    multi_component_f1_reward
)


from grpo_monitoring_callback import create_monitoring_callbacks
from rollout_logging_callback import create_rollout_callback
from functools import partial
from transformers import TrainerCallback


class RewardWeightScheduler:
    
    def __init__(
        self,
        max_steps: int,
        format_start_weight: float = 1.0,
        format_end_weight: float = 0.1,
        format_decay_start: float = 0.2,  
        format_decay_end: float = 0.5,    
        quality_start_weight: float = None,  
        quality_end_weight: float = None,    
        quality_warmup_start: float = 0.1,   
        quality_warmup_end: float = 0.4,     
        schedule_type: str = "linear",  
    ):
        self.max_steps = max_steps
        self.current_step = 0
        
        
        self.format_start_weight = format_start_weight
        self.format_end_weight = format_end_weight
        self.format_decay_start_step = int(max_steps * format_decay_start)
        self.format_decay_end_step = int(max_steps * format_decay_end)
        
        
        self.quality_start_weight = quality_start_weight
        self.quality_end_weight = quality_end_weight
        self.quality_warmup_start_step = int(max_steps * quality_warmup_start)
        self.quality_warmup_end_step = int(max_steps * quality_warmup_end)
        self.schedule_type = schedule_type
    
    def update_step(self, step: int):
        
        self.current_step = step
    
    def get_format_weight(self) -> float:
        
        if self.current_step < self.format_decay_start_step:
            return self.format_start_weight
        elif self.current_step >= self.format_decay_end_step:
            return self.format_end_weight
        else:
            
            progress = (self.current_step - self.format_decay_start_step) /                      (self.format_decay_end_step - self.format_decay_start_step + 1e-8)
            
            if self.schedule_type == "cosine":
                import math
                progress = (1 - math.cos(progress * math.pi)) / 2
            
            weight = self.format_start_weight + (self.format_end_weight - self.format_start_weight) * progress
            return weight
    
    def get_quality_weight(self, base_weight: float, max_weight: float = None, max_multiplier: float = 2.0) -> float:
        
        if max_weight is None:
            max_weight = base_weight * max_multiplier
        
        
        start_weight = self.quality_start_weight if self.quality_start_weight is not None else base_weight
        end_weight = self.quality_end_weight if self.quality_end_weight is not None else max_weight
        
        if self.current_step < self.quality_warmup_start_step:
            return start_weight
        elif self.current_step >= self.quality_warmup_end_step:
            return end_weight
        else:
            
            progress = (self.current_step - self.quality_warmup_start_step) /                      (self.quality_warmup_end_step - self.quality_warmup_start_step + 1e-8)
            
            if self.schedule_type == "cosine":
                import math
                progress = (1 - math.cos(progress * math.pi)) / 2
            
            weight = start_weight + (end_weight - start_weight) * progress
            return weight
    
    def get_weights_summary(self) -> dict:
        
        return {
            "format_weight": self.get_format_weight(),
            "step": self.current_step,
            "progress": self.current_step / self.max_steps if self.max_steps > 0 else 0.0,
        }


_global_weight_scheduler = None


class DynamicWeightUpdateCallback(TrainerCallback):
    
    def __init__(self, scheduler: RewardWeightScheduler):
        super().__init__()
        self.scheduler = scheduler
    
    def on_step_end(self, args, state, control, **kwargs):
        
        if state.global_step is not None:
            self.scheduler.update_step(state.global_step)
        
        
        if state.global_step is not None and state.global_step % 10 == 0:
            weights_summary = self.scheduler.get_weights_summary()
            
            
def create_weighted_reward_func(reward_func, weight: float, use_dynamic_scheduler: bool = False, scheduler_type: str = "format", max_multiplier: float = 2.0):
    
    def weighted_wrapper(*args, **kwargs):
        rewards = reward_func(*args, **kwargs)
        
        
        if use_dynamic_scheduler and _global_weight_scheduler is not None:
            if scheduler_type == "format":
                current_weight = _global_weight_scheduler.get_format_weight()
            else:  
                current_weight = _global_weight_scheduler.get_quality_weight(weight, max_multiplier=max_multiplier)
        else:
            current_weight = weight
        
        
        if isinstance(rewards, list):
            return [r * current_weight for r in rewards]
        else:
            return rewards * current_weight
    
    
    if use_dynamic_scheduler:
        weighted_wrapper.__name__ = f"{reward_func.__name__}_dynamic"
    else:
        weighted_wrapper.__name__ = f"{reward_func.__name__}_x{weight}"
    
    return weighted_wrapper


def extract_titles(text: str, section_name: str) -> List[str]:
    
    if section_name == "Noise Items":
        pattern = r"Noise Items:\s*(.*?)(?=\n?(?:Suggested|Alternative|Recommended|Alternate|Noise|###)|$)"
    else:
        pattern = r"(?:Suggested|Alternative|Recommended|Alternate|Recommended Replacing)\s*(?:Alternatives|Replacements|Replacement|Items|Item|Replacing Item)?:\s*(.*)"
        
    match = re.search(pattern, text, re.S | re.I)
    if not match:
        return []
    
    content = match.group(1)
    
    titles = re.findall(r'"(.*?)"', content)
    
    return [t.strip().lower() for t in titles if t.strip()]

def preprocess_dataset(example):
    
    
    enhanced_instruction = (
        f"{example['instruction']}\n\n"
        "Please follow the output format strictly and do NOT provide any explanation or extra text.\n"
        "Example Output Format:\n"
        "Noise Items: \"Item A\", \"Item B\"\n"
        "Suggested Alternatives: \"Item C\", \"Item D\""
    )

    
    prompt = (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{enhanced_instruction}\n\n"
        f"### Input:\n{example['input']}\n\n"
        "### Response:\n"
    )
    
    
    noise_match = re.search(r"Noise Items:\s*(.*?)(?=\n|Suggested|$)", example["output"], re.S)
    if noise_match:
        titles = re.findall(r'"(.*?)"', noise_match.group(1))
        gt_noises = [t.strip().lower() for t in titles]
    else:
        gt_noises = []
    
    
    return {
        "prompt": prompt,
        "gt_noises": gt_noises,
        "original_input": example['input']  
    }


def format_reward_func(completions, **kwargs) -> List[float]:
    
    rewards = []
    pattern = r"Noise Items:.*Suggested.*"
    for content in completions:
        res_text = content[0]["content"] if isinstance(content, list) else content
        if re.search(pattern, res_text, re.DOTALL | re.I):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

def noise_f1_reward_func(completions, gt_noises, **kwargs) -> List[float]:
    
    rewards = []
    
    for content, gt in zip(completions, gt_noises):
        res_text = content[0]["content"] if isinstance(content, list) else content
        pred_noises = extract_titles(res_text, "Noise Items")
        
        
        pred_set = set([p.lower().strip() for p in pred_noises])
        gt_set = set([g.lower().strip() for g in gt])
        
        if not gt_set:  
            rewards.append(1.0 if not pred_set else 0.0)
            continue
        if not pred_set:
            rewards.append(0.0)
            continue
            
        tp = len(pred_set & gt_set)
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)
        
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
        rewards.append(f1)
    return rewards

def validity_reward_func(completions, item_pool, **kwargs) -> List[float]:
    
    rewards = []
    for content in completions:
        res_text = content[0]["content"] if isinstance(content, list) else content
        suggests = extract_titles(res_text, "Suggested Alternatives")
        if not suggests:
            rewards.append(0.0)
            continue
            
        valid_count = sum(1 for t in suggests if t in item_pool)
        rewards.append(valid_count / len(suggests))
    return rewards


def train_grpo(
    dataset_type: str = "movie",  
    model_path: str = None,  
    train_data_path: str = None,  
    id2name_path: str = None,  
    sasrec_model_path: str = None,  
    output_dir: str = None,  
    lr: float = 1e-5,
    batch_size: int = 1,
    acc_steps: int = 4,
    num_generations: int = 4,
    max_steps: int = -1,  
    num_train_epochs: int = 1,  
    lora_r: int = 32,
    load_in_4bit: bool = True,  
    max_seq_length: int = 1024,
    item_num: int = None,  
    use_format_reward: bool = True,  
    f1_reward_type: str = "shaped",  
    f1_reward_weight: float = 1.0,  
    use_sasrec_reward: bool = True,
    sasrec_reward_weight: float = 1.0,  
    sasrec_reward_mode: str = "ranking",  
    sasrec_reward_mapping: str = "stratified",  
    resume_from_checkpoint: str = None,  
    use_dynamic_reward_weights: bool = False,
    format_decay_start: float = 0.2,  
    format_decay_end: float = 0.5,    
    quality_warmup_start: float = 0.1,  
    quality_warmup_end: float = 0.4,    
    quality_max_multiplier: float = 2.0,  
):
    
    
    dataset_type = dataset_type.lower()
    if dataset_type not in ["movie", "toy"]:
        raise ValueError(f"dataset_type must be 'movie' or 'toy', got '{dataset_type}'")
    
    if dataset_type == "movie":
        default_model_path = "models/llama3_8B_Instruct_sft_denoised_merged"
        default_train_data_path = "data/Movie_noise_user0.1/TDMD/train_with_description.json"
        default_id2name_path = "data/Movie_denoise_user0.1_10000_1400/id2name4Rec.json"
        default_sasrec_model_path = "models/sasrec/Movie_denoise_user0.1_10000_1400_old/best_model.pth"
        default_output_dir = "output/grpo_unsloth_sasrec_movie_v1"
        default_item_num = 5101
    else:  
        default_model_path = "models/denoise/Toy_noise_user0.1/TDMD/final_model"
        default_train_data_path = "data/Toy_noise_user0.1/TDMD/train_with_description.json"
        default_id2name_path = "data/Toy_denoise_user0.1_10000_1400/id2name4Rec.json"
        default_sasrec_model_path = "models/sasrec/Toy_denoise_user0.1_10000_1400/best_model.pth"
        default_output_dir = "output/grpo_unsloth_sasrec_toy_v1"
        default_item_num = 11758  
    
    
    model_path = model_path if model_path is not None else default_model_path
    train_data_path = train_data_path if train_data_path is not None else default_train_data_path
    id2name_path = id2name_path if id2name_path is not None else default_id2name_path
    sasrec_model_path = sasrec_model_path if sasrec_model_path is not None else default_sasrec_model_path
    output_dir = output_dir if output_dir is not None else default_output_dir
    item_num = item_num if item_num is not None else default_item_num
    
    print(f"🚀 Starting GRPO training with Unsloth + SASRec framework...")
    print(f"📋 Dataset Type: {dataset_type.upper()}")
    print(f"📦 Model: {model_path}")
    print(f"📊 Dataset: {train_data_path}")
    print(f"🗺️  ID2Name: {id2name_path}")
    print(f"🎯 SASRec Model: {sasrec_model_path}")
    print(f"📊 Item Num: {item_num}")
    print(f"💾 Output Dir: {output_dir}")
    print(f"💾 4-bit quantization: {load_in_4bit}")
    print(f"🎯 Format reward: {'ENABLED' if use_format_reward else 'DISABLED'}")
    print(f"🎯 SASRec reward: {'ENABLED' if use_sasrec_reward else 'DISABLED'}")
    if use_sasrec_reward:
        print(f"   Mode: {sasrec_reward_mode.upper()}")
        print(f"   Weight: {sasrec_reward_weight}")
    
    
    print("\n[1/6] Loading model with Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        max_lora_rank=lora_r,
    )
    
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    
    print("\n[2/6] Applying LoRA with Unsloth optimizations...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_alpha=lora_r * 2,
        lora_dropout=0.05,
        use_gradient_checkpointing="unsloth",  
        random_state=42,
    )
    
    
    print("\n[3/6] Loading and preprocessing dataset...")
    raw_dataset = load_dataset("json", data_files=train_data_path, split="train")
    raw_dataset = raw_dataset.shuffle(seed=42)
    dataset = raw_dataset.map(preprocess_dataset, remove_columns=raw_dataset.column_names)
    print(f"✅ Dataset prepared: {len(dataset)} samples")
    
    
    print("\n[4/6] Preparing reward functions...")
    with open(id2name_path, "r", encoding="utf-8") as f:
        id2name = json.load(f)
    item_pool = set([v.strip().lower() for v in id2name.values()])
    
    from functools import partial
    bound_validity_reward_func = partial(validity_reward_func, item_pool=item_pool)
    bound_validity_reward_func.__name__ = "validity_reward_func"
    
    
    sasrec_calculator = None
    bound_sasrec_reward_func = None
    
    if use_sasrec_reward:
        print("\n[5/6] Initializing SASRec Reward Calculator...")
        print("=" * 80)
        
        try:
            sasrec_calculator = SASRecRewardCalculator(
                sasrec_model_path=sasrec_model_path,
                id2name_path=id2name_path,
                item_num=item_num,
                maxlen=10,
                hidden_units=64,
                num_blocks=2,
                num_heads=1,
                dropout_rate=0.5,
                device='cuda',
                topk=50,
                reward_mapping=sasrec_reward_mapping  
            )
            
            
            if sasrec_reward_mode == "ranking":
                print("📊 Using RANKING-based reward (dense signal, ~95% coverage)")
                
                
                bound_sasrec_reward_func = partial(
                    sasrec_ranking_improvement_reward_func,
                    calculator=sasrec_calculator
                )
                bound_sasrec_reward_func.__name__ = "sasrec_ranking_reward_func"
                
            else:  
                print("📊 Using NDCG-based reward (sparse signal, ~11% coverage)")
                
                
                bound_sasrec_reward_func = partial(
                    sasrec_improvement_reward_func,
                    calculator=sasrec_calculator
                )
                bound_sasrec_reward_func.__name__ = "sasrec_improvement_reward_func"
            
            print("✅ SASRec Reward Calculator initialized successfully!")
            print(f"   Mode: {sasrec_reward_mode.upper()}")
            print(f"   Weight: {sasrec_reward_weight}")
            print("=" * 80)
        except Exception as e:
            print(f"❌ Failed to initialize SASRec Reward Calculator: {e}")
            print("   Continuing without SASRec reward...")
            use_sasrec_reward = False
    else:
        print("\n[5/6] Skipping SASRec initialization (disabled)")
    
    
    print("\n[6/6] Initializing GRPO Trainer...")
    
    max_prompt_length = 512
    max_completion_length = max_seq_length - max_prompt_length
    
    
    import math
    num_gpus = torch.cuda.device_count()
    global_batch_size = batch_size * num_gpus * acc_steps
    generation_batch_size = math.lcm(num_generations, global_batch_size)
    
    print(f"\n📊 Training Configuration:")
    print(f"  - GPUs: {num_gpus}")
    print(f"  - Per-device batch size: {batch_size}")
    print(f"  - Gradient accumulation: {acc_steps}")
    print(f"  - Global batch size: {global_batch_size}")
    print(f"  - Num generations: {num_generations}")
    print(f"  - Generation batch size: {generation_batch_size} (auto-computed)")
    
    
    if f1_reward_type == "shaped":
        f1_func = shaped_f1_reward_func
        print(f"📊 Using Shaped F1 reward function (weight={f1_reward_weight})")
    elif f1_reward_type == "action":
        f1_func = denoising_action_reward_func
        print(f"📊 Using Action reward function (weight={f1_reward_weight})")
    elif f1_reward_type == "multi":
        f1_func = multi_component_f1_reward
        print(f"📊 Using Multi-component combined reward function (weight={f1_reward_weight})")
    elif f1_reward_type == "none":
        f1_func = None
        print(f"📊 Not using F1 reward function (weight={f1_reward_weight})")
    else:  
        f1_func = noise_f1_reward_func
        print(f"📊 Using Standard F1 reward function (weight={f1_reward_weight})")
    
    
    reward_funcs = []
    
    
    if use_format_reward:
        if use_dynamic_reward_weights:
            weighted_format_func = create_weighted_reward_func(
                format_reward_func, 
                weight=1.0, 
                use_dynamic_scheduler=True, 
                scheduler_type="format"
            )
            reward_funcs.append(weighted_format_func)
        else:
            reward_funcs.append(format_reward_func)
    
    
    if f1_func is not None:
        if use_dynamic_reward_weights:
            weighted_f1_func = create_weighted_reward_func(
                f1_func, 
                weight=f1_reward_weight, 
                use_dynamic_scheduler=True, 
                scheduler_type="quality",
                max_multiplier=quality_max_multiplier
            )
            reward_funcs.append(weighted_f1_func)
        else:
            reward_funcs.append(create_weighted_reward_func(f1_func, f1_reward_weight))
    
    
    reward_funcs.append(bound_validity_reward_func)
    
    
    f1_func_name = {
        "shaped": "shaped_f1_reward_func",
        "action": "denoising_action_reward_func", 
        "multi": "multi_component_f1_reward",
        "standard": "noise_f1_reward_func"
    }.get(f1_reward_type, "noise_f1_reward_func")
    
    
    if use_sasrec_reward and bound_sasrec_reward_func is not None:
        if use_dynamic_reward_weights:
            weighted_sasrec_func = create_weighted_reward_func(
                bound_sasrec_reward_func, 
                sasrec_reward_weight, 
                use_dynamic_scheduler=True, 
                scheduler_type="quality",
                max_multiplier=quality_max_multiplier
            )
        else:
            weighted_sasrec_func = create_weighted_reward_func(bound_sasrec_reward_func, sasrec_reward_weight)
        reward_funcs.append(weighted_sasrec_func)
    
    
    reward_list = []
    idx = 1
    if use_format_reward:
        if use_dynamic_reward_weights:
            reward_list.append(f"  {idx}. format_reward_func (weight: 1.0 → 0.1, dynamic decay)")
        else:
            reward_list.append(f"  {idx}. format_reward_func (weight: 1.0)")
        idx += 1
    if f1_func is not None:
        if use_dynamic_reward_weights:
            reward_list.append(f"  {idx}. {f1_func_name} (weight: {f1_reward_weight} → {f1_reward_weight * quality_max_multiplier:.2f}, dynamic increase)")
        else:
            reward_list.append(f"  {idx}. {f1_func_name} (weight: {f1_reward_weight})")
        idx += 1
    reward_list.append(f"  {idx}. validity_reward_func (weight: 1.0, static)")
    idx += 1
    if use_sasrec_reward and bound_sasrec_reward_func is not None:
        if use_dynamic_reward_weights:
            reward_list.append(f"  {idx}. sasrec_{sasrec_reward_mode}_reward_func (weight: {sasrec_reward_weight} → {sasrec_reward_weight * quality_max_multiplier:.2f}, dynamic increase)")
        else:
            reward_list.append(f"  {idx}. sasrec_{sasrec_reward_mode}_reward_func (weight: {sasrec_reward_weight})")
    
    
    dynamic_suffix = ", with dynamic weights" if use_dynamic_reward_weights else ""
    sasrec_suffix = "" if (use_sasrec_reward and bound_sasrec_reward_func is not None) else ", without SASRec"
    print(f"\n🎯 Reward Functions ({len(reward_funcs)} total{sasrec_suffix}{dynamic_suffix}):")
    for line in reward_list:
        print(line)
    
    
    config_kwargs = {
        "output_dir": output_dir,
        "learning_rate": lr,
        "beta": 0.01,
        "adam_beta1": 0.9,
        "adam_beta2": 0.99,
        "weight_decay": 0.1,
        "warmup_ratio": 0.1,
        "lr_scheduler_type": "cosine",
        "optim": "paged_adamw_8bit" if load_in_4bit else "adamw_torch",
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": acc_steps,
        "num_generations": num_generations,
        "generation_batch_size": generation_batch_size,
        "max_prompt_length": max_prompt_length,
        "max_completion_length": max_completion_length,
        "max_grad_norm": 0.1,
    }
    
    
    if max_steps > 0:
        config_kwargs["max_steps"] = max_steps
        total_steps = max_steps
        
    else:
        config_kwargs["max_steps"] = -1
        config_kwargs["num_train_epochs"] = num_train_epochs
        
        total_steps = int(len(dataset) * num_train_epochs / (batch_size * acc_steps))
    
    
    global _global_weight_scheduler
    dynamic_weight_callback = None
    
    if use_dynamic_reward_weights:
        print("\n🎯 Enabling dynamic reward weight adjustment:")
        if use_format_reward:
            print(f"   - Format reward: 1.0 → 0.1 (decay interval: {format_decay_start*100:.0f}% - {format_decay_end*100:.0f}% of steps)")
        if f1_func is not None:
            print(f"   - Quality reward: {f1_reward_weight} → {f1_reward_weight * quality_max_multiplier:.2f} (increase interval: {quality_warmup_start*100:.0f}% - {quality_warmup_end*100:.0f}% of steps)")
        if use_sasrec_reward:
            print(f"   - SASRec reward: {sasrec_reward_weight} → {sasrec_reward_weight * quality_max_multiplier:.2f}")
        
        _global_weight_scheduler = RewardWeightScheduler(
            max_steps=total_steps,
            format_decay_start=format_decay_start,
            format_decay_end=format_decay_end,
            quality_warmup_start=quality_warmup_start,
            quality_warmup_end=quality_warmup_end,
        )
        
        dynamic_weight_callback = DynamicWeightUpdateCallback(_global_weight_scheduler)
        print("✅ Dynamic weight scheduler initialized\n")
    else:
        print("\n📊 Using static reward weights (dynamic adjustment disabled)\n")
        _global_weight_scheduler = None
    
    config = GRPOConfig(
        **config_kwargs,
        
        
        logging_steps=1,
        save_steps=100,
        report_to="tensorboard",
        bf16=not load_in_4bit,
        fp16=False,
    )
    
    
    monitoring_callbacks = create_monitoring_callbacks()
    
    
    rollout_callback = create_rollout_callback(
        print_every_steps=100,  
        num_samples=2,  
        save_to_file=True,  
        output_dir=output_dir,  
    )
    
    
    all_callbacks = monitoring_callbacks + [rollout_callback]
    if dynamic_weight_callback is not None:
        all_callbacks.append(dynamic_weight_callback)
    
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=config,
        train_dataset=dataset,
        callbacks=all_callbacks,
    )
    
    print("\n" + "=" * 70)
    print("🎯 Training started! Monitor with: tensorboard --logdir", output_dir)
    if resume_from_checkpoint:
        print(f"📂 Resuming from checkpoint: {resume_from_checkpoint}")
    print("=" * 70 + "\n")
    
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    
    print("\n✅ Training completed! Saving model...")
    model.save_pretrained(f"{output_dir}/final_lora")
    tokenizer.save_pretrained(f"{output_dir}/final_lora")
    print(f"💾 Model saved to {output_dir}/final_lora")
    
    
    print("\n📦 Saving merged 16-bit model (optional)...")
    try:
        model.save_pretrained_merged(
            f"{output_dir}/final_merged_16bit",
            tokenizer,
            save_method="merged_16bit",
        )
        print(f"💾 Merged model saved to {output_dir}/final_merged_16bit")
    except Exception as e:
        print(f"⚠️ Failed to save merged model: {e}")
        print("   (LoRA weights already saved)")

if __name__ == "__main__":
    fire.Fire(train_grpo)
