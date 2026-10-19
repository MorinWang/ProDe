#!/bin/bash

set -e

echo "=============================================="
echo "  GRPO Training - Multi-Component Strategy"
echo "=============================================="
echo ""

MODEL_PATH="models/llama3_8B_Instruct_sft_denoised_merged"
OUTPUT_DIR="./outputs/grpo_multi_component_v1"
TRAIN_FILE="data/Movie_noise_user0.1/TDMD/train_with_description_1k.json"
GPU_ID=5

echo "📋 Training Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Output: $OUTPUT_DIR"
echo "  Data: $TRAIN_FILE"
echo "  GPU: $GPU_ID"
echo "  Strategy: Multi-Component (Balanced & Stable)"
echo ""

if [ ! -f "$TRAIN_FILE" ]; then
    echo "❌ Error: Training data not found: $TRAIN_FILE"
    exit 1
fi

echo "✅ Training data check passed"
echo ""

mkdir -p "$OUTPUT_DIR"

echo "🚀 Starting training..."
echo "  (Estimated time: 3-4 hours)"
echo ""

CUDA_VISIBLE_DEVICES=$GPU_ID python train_grpo_unsloth_with_sasrec.py \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --train_file "$TRAIN_FILE" \
    --f1_reward_type "multi" \
    --f1_reward_weight 1.5 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-6 \
    --num_train_epochs 1 \
    --max_steps -1 \
    --use_sasrec_reward False \
    --save_steps 100 \
    --logging_steps 10

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

if [ $EXIT_CODE -eq 0 ]; then
    echo "📊 Next Steps:"
    echo "  1. View TensorBoard:"
    echo "     tensorboard --logdir $OUTPUT_DIR --port 6008"
    echo ""
    echo "  2. Compare effects of different strategies"
    echo ""
fi

exit $EXIT_CODE
