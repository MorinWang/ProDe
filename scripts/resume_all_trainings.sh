#!/bin/bash

set -e

echo "=============================================="
echo "  🔄 Batch Resume All Trainings"
echo "=============================================="
echo ""

echo "📊 Checking checkpoint status..."
echo ""

strategies=("shaped_f1" "action_reward" "multi_component")
valid_checkpoints=()

for strategy in "${strategies[@]}"; do
    checkpoint="outputs/grpo_${strategy}_v1/checkpoint-100"
    if [ -d "$checkpoint" ]; then
        echo "✅ $strategy: checkpoint-100 exists"
        valid_checkpoints+=("$strategy")
    else
        echo "❌ $strategy: checkpoint-100 not found, skipping"
    fi
done

echo ""

if [ ${#valid_checkpoints[@]} -eq 0 ]; then
    echo "❌ No checkpoints available to resume"
    exit 1
fi

echo "Will resume ${#valid_checkpoints[@]} trainings"
echo ""

read -p "Confirm to continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Cancelled"
    exit 1
fi

echo ""
echo "=============================================="
echo "  Starting Resume Training..."
echo "=============================================="
echo ""

mkdir -p logs

pids=()

for strategy in "${valid_checkpoints[@]}"; do
    echo "🚀 Starting $strategy..."
    
    nohup bash resume_grpo_${strategy}.sh \
        > logs/resume_${strategy}_$(date +%Y%m%d_%H%M%S).log 2>&1 &
    
    pid=$!
    pids+=($pid)
    echo "  ✓ PID: $pid"
    sleep 2
done

echo ""
echo "=============================================="
echo "  ✅ All Trainings Started!"
echo "=============================================="
echo ""

echo "📊 Process Information:"
for i in "${!valid_checkpoints[@]}"; do
    echo "  ${valid_checkpoints[$i]}: PID ${pids[$i]}"
done
echo ""

echo "💡 Monitoring Methods:"
echo "  1. Check processes: ps aux | grep train_grpo"
echo "  2. Check GPU: nvidia-smi"
echo "  3. Check logs: tail -f logs/resume_*.log"
echo "  4. TensorBoard: tensorboard --logdir outputs/ --port 6006"
echo ""

echo "Estimated completion time: approximately 2-3 hours (from step 100 to step 500)"
echo ""
