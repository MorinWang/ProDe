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
REWARD_CKPT="$PRODE_ROOT/models/sasrec/Movie_denoise_user0.1_10000_1400_old/best_model.pth"

POLICY_10="$PRODE_ROOT/output/decouple_movie/policy_reward-sasrec/final_merged_16bit"

declare -A TRAIN_JSON
TRAIN_JSON[0.2]="$PRODE_ROOT/data/Movie_noise_user0.2/TDMD/train_with_description.json"
TRAIN_JSON[0.3]="$PRODE_ROOT/data/Movie_noise_user0.3/TDMD/train_with_description.json"

declare -A NOISY_DIR
NOISY_DIR[0.2]="$PRODE_ROOT/data/Movie_noisy_0.2"
NOISY_DIR[0.3]="$PRODE_ROOT/data/Movie_noisy_0.3"

OUT="$PRODE_ROOT/output/noise_transfer_${DATASET}"
RESULTS="$OUT/results"
mkdir -p "$OUT" "$RESULTS"
cd "$DECOUPLE_DIR"

for a in 0.2 0.3; do
  python downstream_eval.py \
    --dataset_dir "${NOISY_DIR[$a]}" --model sasrec \
    --out_json "$RESULTS/none_alpha${a}.json" \
    --num_epochs "$DL_EPOCHS" --maxlen "$MAXLEN" --cuda "$CUDA"

  python denoise_trainset.py \
    --policy_model "$POLICY_10" --source_dir "${NOISY_DIR[$a]}" \
    --id2name "$ID2NAME" --id2desc "$ID2DESC" \
    --out_dir "$OUT/data_transfer_alpha${a}" --tp 4 --cuda "0,1,2,3"
  python downstream_eval.py \
    --dataset_dir "$OUT/data_transfer_alpha${a}" --model sasrec \
    --out_json "$RESULTS/transfer_alpha${a}.json" \
    --num_epochs "$DL_EPOCHS" --maxlen "$MAXLEN" --cuda "$CUDA"

  python train_grpo_decouple.py \
    --reward_backbone sasrec --reward_model_path "$REWARD_CKPT" \
    --dataset_type "$DATASET" --model_path "$SFT_MODEL" \
    --train_data_path "${TRAIN_JSON[$a]}" --id2name_path "$ID2NAME" --item_num "$ITEM_NUM" \
    --output_dir "$OUT/policy_matched_alpha${a}" --num_generations 4 --max_steps "$GRPO_STEPS"
  python denoise_trainset.py \
    --policy_model "$OUT/policy_matched_alpha${a}/final_merged_16bit" --source_dir "${NOISY_DIR[$a]}" \
    --id2name "$ID2NAME" --id2desc "$ID2DESC" \
    --out_dir "$OUT/data_matched_alpha${a}" --tp 4 --cuda "0,1,2,3"
  python downstream_eval.py \
    --dataset_dir "$OUT/data_matched_alpha${a}" --model sasrec \
    --out_json "$RESULTS/matched_alpha${a}.json" \
    --num_epochs "$DL_EPOCHS" --maxlen "$MAXLEN" --cuda "$CUDA"
done

python aggregate_generic.py --results_dir "$RESULTS"
