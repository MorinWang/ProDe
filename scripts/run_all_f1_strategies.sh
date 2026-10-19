#!/bin/bash

set -e

echo "=============================================="
echo "  🚀 Launch All F1 Strategy Comparison Experiments"
echo "=============================================="
echo ""

echo "📊 Checking GPU status..."
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader,nounits

echo ""
echo "Will launch training on the following GPUs:"
echo "  GPU 4: Standard F1 (Baseline)"
echo "  GPU 5: Multi-Component"
echo "  GPU 6: Action Reward"
echo "  GPU 7: Shaped F1 (already running)"
echo ""

read -p "Confirm to launch? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Cancelled"
    exit 1
fi

echo ""
echo "=============================================="
echo "  Starting Training..."
echo "=============================================="
echo ""

echo "🚀 [1/3] Starting Action Reward Strategy (GPU 6)..."
nohup bash run_grpo_action_reward.sh > logs/action_reward_$(date +%Y%m%d_%H%M%S).log 2>&1 &
ACTION_PID=$!
echo "  ✓ PID: $ACTION_PID"
sleep 5

echo "🚀 [2/3] Starting Multi-Component Strategy (GPU 5)..."
nohup bash run_grpo_multi_component.sh > logs/multi_component_$(date +%Y%m%d_%H%M%S).log 2>&1 &
MULTI_PID=$!
echo "  ✓ PID: $MULTI_PID"
sleep 5

echo "🚀 [3/3] Starting Standard F1 Baseline (GPU 4)..."
nohup bash run_grpo_baseline.sh > logs/baseline_$(date +%Y%m%d_%H%M%S).log 2>&1 &
BASELINE_PID=$!
echo "  ✓ PID: $BASELINE_PID"

echo ""
echo "=============================================="
echo "  ✅ All Trainings Started!"
echo "=============================================="
echo ""

echo "📊 Process Information:"
echo "  Shaped F1 (GPU 7):     Already Running"
echo "  Action Reward (GPU 6): PID $ACTION_PID"
echo "  Multi-Component (GPU 5): PID $MULTI_PID"
echo "  Standard F1 (GPU 4):   PID $BASELINE_PID"
echo ""

echo "💡 Monitoring Methods:"
echo ""
echo "1. Check all training status:"
echo "   bash monitor_all_trainings.sh"
echo ""
echo "2. View TensorBoard (all experiments):"
echo "   tensorboard --logdir_spec=shaped:outputs/grpo_shaped_f1_v1,action:outputs/grpo_action_reward_v1,multi:outputs/grpo_multi_component_v1,baseline:outputs/grpo_baseline_standard_f1_v1 --port 6006"
echo ""
echo "3. View individual logs:"
echo "   tail -f logs/action_reward_*.log"
echo "   tail -f logs/multi_component_*.log"
echo "   tail -f logs/baseline_*.log"
echo ""
echo "4. Check process status:"
echo "   ps aux | grep train_grpo"
echo ""

mkdir -p pids
echo $ACTION_PID > pids/action_reward.pid
echo $MULTI_PID > pids/multi_component.pid
echo $BASELINE_PID > pids/baseline.pid

echo "PIDs saved to pids/ directory"
echo ""
echo "Estimated completion time: 3-4 hours for all trainings"
echo ""
