#!/bin/bash

set -e

echo "=============================================="
echo "  Resume Training from Checkpoint - Action Reward"
echo "=============================================="
echo ""

MODEL_PATH="models/llama3_8B_Instruct_sft_denoised_merged"
OUTPUT_DIR="./outputs/grpo_action_reward_v1"
TRAIN_FILE="data/Movie_noise_user0.1/TDMD/train_with_description_1k.json"
CHECKPOINT="./outputs/grpo_action_reward_v1/checkpoint-100"
GPU_ID=6

echo "📋 Resume Training Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Output: $OUTPUT_DIR"
echo "  Data: $TRAIN_FILE"
echo "  Checkpoint: $CHECKPOINT"
echo "  GPU: $GPU_ID"
echo ""

if [ ! -d "$CHECKPOINT" ]; then
    echo "❌ Error: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

echo "✅ Checkpoint check passed"
echo ""

echo "🚀 Resuming training..."
echo "  (Will continue from step 100 to step 500)"
echo ""

CUDA_VISIBLE_DEVICES=$GPU_ID python train_grpo_unsloth_with_sasrec.py \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --train_file "$TRAIN_FILE" \
    --f1_reward_type "action" \
    --f1_reward_weight 2.0 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-6 \
    --max_steps 500 \
    --use_sasrec_reward False \
    --save_steps 100 \
    --logging_steps 10 \
    --resume_from_checkpoint "$CHECKPOINT"

EXIT_CODE=$?

echo ""
echo "=============================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "  ✅ Training Completed!"
else
    echo "  ❌ Training Failed (Exit Code: $EXIT_CODE)"
fi
echo "=============================================="
echo ""

exit $EXIT_CODE
