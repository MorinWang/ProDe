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
CLEAN_DATA_DIR="$PRODE_ROOT/data/Movie_clean"
NOISY_DATA_DIR="$PRODE_ROOT/data/Movie_noisy"

OUT="$PRODE_ROOT/output/decouple_${DATASET}"
RESULTS="$OUT/results"
REWARD_DIR="$OUT/reward_models"
mkdir -p "$OUT" "$RESULTS" "$REWARD_DIR"

BACKBONES=(gru4rec sasrec bert4rec)

declare -A REWARD_CKPT
REWARD_CKPT[gru4rec]="$REWARD_DIR/gru4rec/best_model.pth"
REWARD_CKPT[sasrec]="$REWARD_DIR/sasrec/best_model.pth"
REWARD_CKPT[bert4rec]="$REWARD_DIR/bert4rec/best_model.pth"

cd "$DECOUPLE_DIR"

for rb in "${BACKBONES[@]}"; do
  python downstream_eval.py \
    --dataset_dir "$CLEAN_DATA_DIR" --model "$rb" \
    --out_json "$REWARD_DIR/$rb/clean_metrics.json" \
    --save_ckpt "${REWARD_CKPT[$rb]}" \
    --num_epochs "$DL_EPOCHS" --maxlen "$MAXLEN" --cuda "$CUDA"
done

for rb in "${BACKBONES[@]}"; do
  python train_grpo_decouple.py \
    --reward_backbone "$rb" \
    --reward_model_path "${REWARD_CKPT[$rb]}" \
    --dataset_type "$DATASET" \
    --model_path "$SFT_MODEL" \
    --train_data_path "$TRAIN_JSON" \
    --id2name_path "$ID2NAME" \
    --item_num "$ITEM_NUM" \
    --output_dir "$OUT/policy_reward-$rb" \
    --num_generations 8 --max_steps "$GRPO_STEPS"
done

for rb in "${BACKBONES[@]}"; do
  python denoise_trainset.py \
    --policy_model "$OUT/policy_reward-$rb/final_merged_16bit" \
    --source_dir "$NOISY_DATA_DIR" \
    --id2name "$ID2NAME" --id2desc "$ID2DESC" \
    --out_dir "$OUT/data_reward-$rb" \
    --tp 4 --cuda "0,1,2,3"
done

for rb in "${BACKBONES[@]}"; do
  for eb in "${BACKBONES[@]}"; do
    python downstream_eval.py \
      --dataset_dir "$OUT/data_reward-$rb" --model "$eb" \
      --out_json "$RESULTS/reward-${rb}_eval-${eb}.json" \
      --num_epochs "$DL_EPOCHS" --maxlen "$MAXLEN" --cuda "$CUDA"
  done
done

for eb in "${BACKBONES[@]}"; do
  python downstream_eval.py \
    --dataset_dir "$NOISY_DATA_DIR" --model "$eb" \
    --out_json "$RESULTS/none_eval-${eb}.json" \
    --num_epochs "$DL_EPOCHS" --maxlen "$MAXLEN" --cuda "$CUDA"
done

python aggregate_matrix.py --results_dir "$RESULTS" --metric "NDCG@20"
