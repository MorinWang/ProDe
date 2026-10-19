import os
import sys
import json
import math
from functools import partial

import fire
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer

from configurable_srm_reward import ConfigurableSRMRewardCalculator, srm_reward_func

PRODE_ROOT = os.environ.get(
    "PRODE_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
for _p in (os.path.join(PRODE_ROOT, "training"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from train_grpo_unsloth_with_sasrec import (
    preprocess_dataset,
    format_reward_func,
    validity_reward_func,
)
from improved_reward_funcs import (
    shaped_f1_reward_func,
    denoising_action_reward_func,
    multi_component_f1_reward,
)
from grpo_monitoring_callback import create_monitoring_callbacks
from rollout_logging_callback import create_rollout_callback


def _weighted(reward_func, weight):
    def wrapper(*args, **kwargs):
        rewards = reward_func(*args, **kwargs)
        return [r * weight for r in rewards] if isinstance(rewards, list) else rewards * weight
    wrapper.__name__ = f"{reward_func.__name__}_x{weight}"
    return wrapper


def _rescale(w1, w2, w3, total=3.0):
    s = w1 + w2 + w3
    if s <= 0:
        return 1.0, 1.0, 1.0
    k = total / s
    return w1 * k, w2 * k, w3 * k


def train_grpo_sensitivity(
    reward_model_path,
    dataset_type="movie",
    model_path=None,
    train_data_path=None,
    id2name_path=None,
    output_dir=None,
    item_num=None,
    reward_backbone="sasrec",
    w1=1.0,
    w2=1.0,
    w3=1.0,
    rescale_weights=True,
    tau1=50,
    tau2=200,
    constant_mode="default",
    f1_reward_type="shaped",
    format_weight=1.0,
    lr=5e-6,
    batch_size=1,
    acc_steps=4,
    num_generations=8,
    max_steps=500,
    num_train_epochs=1,
    lora_r=32,
    load_in_4bit=True,
    max_seq_length=1024,
    reward_hidden_units=64,
    reward_num_blocks=2,
    reward_num_heads=1,
    reward_gru_layers=1,
    resume_from_checkpoint=None,
):
    dataset_type = dataset_type.lower()
    if dataset_type == "movie":
        d = dict(
            model_path="models/llama3_8B_Instruct_sft_denoised_merged",
            train_data_path="data/Movie_noise_user0.1/TDMD/train_with_description.json",
            id2name_path="data/Movie_denoise_user0.1_10000_1400/id2name4Rec.json",
            item_num=5101,
        )
    elif dataset_type == "toy":
        d = dict(
            model_path="models/denoise/Toy_noise_user0.1/TDMD/final_model",
            train_data_path="data/Toy_noise_user0.1/TDMD/train_with_description.json",
            id2name_path="data/Toy_denoise_user0.1_10000_1400/id2name4Rec.json",
            item_num=11758,
        )
    else:
        raise ValueError("dataset_type must be 'movie' or 'toy'")

    model_path = model_path or d["model_path"]
    train_data_path = train_data_path or d["train_data_path"]
    id2name_path = id2name_path or d["id2name_path"]
    item_num = item_num or d["item_num"]

    if rescale_weights:
        w1, w2, w3 = _rescale(w1, w2, w3)

    tag = f"w{w1:.2f}-{w2:.2f}-{w3:.2f}_tau{tau1}-{tau2}_{constant_mode}"
    output_dir = output_dir or f"output/grpo_sensitivity_{dataset_type}_{tag}"

    print("=" * 80)
    print(f"[sensitivity] weights(noise,valid,rank)=({w1:.3f},{w2:.3f},{w3:.3f})")
    print(f"[sensitivity] tau1={tau1} tau2={tau2} constant_mode={constant_mode}")
    print(f"[sensitivity] reward_backbone={reward_backbone} reward_model={reward_model_path}")
    print(f"[sensitivity] output_dir={output_dir}")
    print("=" * 80)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path, max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit, max_lora_rank=lora_r,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = FastLanguageModel.get_peft_model(
        model, r=lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_r * 2, lora_dropout=0.05,
        use_gradient_checkpointing="unsloth", random_state=42,
    )

    raw = load_dataset("json", data_files=train_data_path, split="train")
    raw = raw.shuffle(seed=42).select(range(min(1000, len(raw))))
    dataset = raw.map(preprocess_dataset, remove_columns=raw.column_names)

    with open(id2name_path, "r", encoding="utf-8") as f:
        id2name = json.load(f)
    item_pool = set(v.strip().lower() for v in id2name.values())
    bound_validity = partial(validity_reward_func, item_pool=item_pool)
    bound_validity.__name__ = "validity_reward_func"

    calc = ConfigurableSRMRewardCalculator(
        model_path=reward_model_path, id2name_path=id2name_path, item_num=item_num,
        backbone=reward_backbone, tau1=tau1, tau2=tau2, constant_mode=constant_mode,
        maxlen=10, hidden_units=reward_hidden_units, num_blocks=reward_num_blocks,
        num_heads=reward_num_heads, gru_layers=reward_gru_layers, device="cuda", topk=50,
    )
    bound_srm = partial(srm_reward_func, calculator=calc)
    bound_srm.__name__ = "srm_ranking_reward"

    f1_func = {
        "shaped": shaped_f1_reward_func,
        "action": denoising_action_reward_func,
        "multi": multi_component_f1_reward,
        "none": None,
    }.get(f1_reward_type, None)

    reward_funcs = [_weighted(format_reward_func, format_weight)]
    if f1_func is not None:
        reward_funcs.append(_weighted(f1_func, w1))
    reward_funcs.append(_weighted(bound_validity, w2))
    reward_funcs.append(_weighted(bound_srm, w3))

    num_gpus = max(1, torch.cuda.device_count())
    global_bs = batch_size * num_gpus * acc_steps
    gen_bs = math.lcm(num_generations, global_bs)
    max_prompt_length = 512

    cfg = dict(
        output_dir=output_dir, learning_rate=lr,
        adam_beta1=0.9, adam_beta2=0.99, weight_decay=0.1,
        warmup_ratio=0.1, lr_scheduler_type="cosine",
        optim="paged_adamw_8bit" if load_in_4bit else "adamw_torch",
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=acc_steps,
        num_generations=num_generations, generation_batch_size=gen_bs,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_seq_length - max_prompt_length,
        max_grad_norm=0.1, logging_steps=1, save_steps=100,
        report_to="tensorboard", bf16=not load_in_4bit, fp16=False,
    )
    if max_steps > 0:
        cfg["max_steps"] = max_steps
    else:
        cfg["max_steps"] = -1
        cfg["num_train_epochs"] = num_train_epochs

    trainer = GRPOTrainer(
        model=model, processing_class=tokenizer, reward_funcs=reward_funcs,
        args=GRPOConfig(**cfg), train_dataset=dataset,
        callbacks=create_monitoring_callbacks() + [
            create_rollout_callback(print_every_steps=100, num_samples=2,
                                    save_to_file=True, output_dir=output_dir)],
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(f"{output_dir}/final_lora")
    tokenizer.save_pretrained(f"{output_dir}/final_lora")
    try:
        model.save_pretrained_merged(f"{output_dir}/final_merged_16bit", tokenizer,
                                     save_method="merged_16bit")
    except Exception as e:
        print(f"[sensitivity] merged save skipped: {e}")
    print(f"[sensitivity] done -> {output_dir}/final_lora")


if __name__ == "__main__":
    fire.Fire(train_grpo_sensitivity)
