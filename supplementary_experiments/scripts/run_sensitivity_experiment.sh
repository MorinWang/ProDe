#!/usr/bin/env bash
set -euo pipefail

export PRODE_ROOT="/Users/zhuangliu/Desktop/essay_writing/ProDE"
DECOUPLE_DIR="$PRODE_ROOT/supplementary_experiments"
DATASET="movie"
ITEM_NUM=5101
MAXLEN=10
GRPO_STEPS=500
DL_EPOCHS=201
CUDA="0"

SFT_MODEL="$PRODE_ROOT/models/llama3_8B_Instruct_sft_denoised_merged"
ID2NAME="$PRODE_ROOT/data/Movie_denoise_user0.1_10000_1400/id2name4Rec.json"
ID2DESC="$PRODE_ROOT/data/Movie_noise_user0.0/id2title_description.json"
TRAIN_JSON="$PRODE_ROOT/data/Movie_noise_user0.1/TDMD/train_with_description.json"
NOISY_DATA_DIR="$PRODE_ROOT/data/Movie_noisy"
REWARD_CKPT="$PRODE_ROOT/models/sasrec/Movie_denoise_user0.1_10000_1400_old/best_model.pth"

OUT="$PRODE_ROOT/output/sensitivity_${DATASET}"
RESULTS="$OUT/results"
mkdir -p "$OUT" "$RESULTS"
cd "$DECOUPLE_DIR"

run_variant () {
  local tag="$1" w1="$2" w2="$3" w3="$4" tau1="$5" tau2="$6" mode="$7"
  local policy="$OUT/policy_$tag"
  local data="$OUT/data_$tag"

  python train_grpo_sensitivity.py \
    --reward_model_path "$REWARD_CKPT" \
    --dataset_type "$DATASET" --model_path "$SFT_MODEL" \
    --train_data_path "$TRAIN_JSON" --id2name_path "$ID2NAME" --item_num "$ITEM_NUM" \
    --w1 "$w1" --w2 "$w2" --w3 "$w3" --tau1 "$tau1" --tau2 "$tau2" --constant_mode "$mode" \
    --output_dir "$policy" --num_generations 8 --max_steps "$GRPO_STEPS"

  python denoise_trainset.py \
    --policy_model "$policy/final_merged_16bit" --source_dir "$NOISY_DATA_DIR" \
    --id2name "$ID2NAME" --id2desc "$ID2DESC" --out_dir "$data" --tp 4 --cuda "0,1,2,3"

  python downstream_eval.py \
    --dataset_dir "$data" --model sasrec --out_json "$RESULTS/$tag.json" \
    --num_epochs "$DL_EPOCHS" --maxlen "$MAXLEN" --cuda "$CUDA"
}

run_variant "default"     1.0 1.0 1.0 50 200 default
run_variant "w1_0.5"      0.5 1.0 1.0 50 200 default
run_variant "w1_2.0"      2.0 1.0 1.0 50 200 default
run_variant "w2_0.5"      1.0 0.5 1.0 50 200 default
run_variant "w2_2.0"      1.0 2.0 1.0 50 200 default
run_variant "w3_0.5"      1.0 1.0 0.5 50 200 default
run_variant "w3_2.0"      1.0 1.0 2.0 50 200 default
run_variant "tau_30_150"  1.0 1.0 1.0 30 150 default
run_variant "tau_80_250"  1.0 1.0 1.0 80 250 default
run_variant "flattened"   1.0 1.0 1.0 50 200 flattened
run_variant "shifted"     1.0 1.0 1.0 50 200 shifted

python aggregate_generic.py --results_dir "$RESULTS"
