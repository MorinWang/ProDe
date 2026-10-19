Large Language Model for Denoising Sequential Recommendation

---

## 📖 Project Introduction

This project uses GRPO (Generative Reinforcement Learning with Policy Optimization) to train large language models for noise identification and removal in sequential recommendation.

**Core Features**:
- ✅ Identify noise items in user behavior sequences
- ✅ Provide denoising explanations
- ✅ Support multiple F1 reward strategies
- ✅ Efficient training based on Unsloth

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Main dependencies:
- Python >= 3.8
- PyTorch >= 2.0
- transformers
- unsloth
- trl

### 2. Prepare Data

#### Method 1: Construct Data from CSV (Recommended) ⭐

If you have raw data in CSV format, use the provided tools to automatically construct training data:

```bash
cd data_processing/process/

# Construct training data
python create_train_with_description_from_csv.py \
    --input your_train.csv \
    --output train_with_description.json \
    --id2name id2name.json \
    --id2description id2description.json

# Construct test data
python create_test_with_description_from_csv.py \
    --input your_test.csv \
    --output test_with_description.json \
    --id2name id2name.json \
    --id2description id2description.json
```

**Detailed Instructions**: See `data_processing/README_DATA.md`

---

#### Method 2: Manually Prepare JSON Data

Data format: JSON format, one sample per line

```json
{
  "user_id": "user_123",
  "history": ["Item_A", "Item_B", "Item_C"],
  "noise_items": ["Item_B"],
  "description": "..."
}
```

### 3. Train Model

#### Single Strategy Training
```bash
bash scripts/run_grpo_action_reward.sh
```

#### Parallel Training of Multiple Strategies
```bash
bash scripts/run_all_f1_strategies.sh
```

### 4. Inference

```bash
python inference/infer_grpo.py \
    --model_path outputs/grpo_action_reward_v1/final_lora \
    --input_file test_data.json \
    --output_file results.json
```

---

## 📁 Directory Structure

```
LLM4DSR_Package/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
│
├── training/                  # Training code
│   ├── train_grpo_unsloth_with_sasrec.py  # Main training script
│   ├── improved_reward_funcs.py           # Reward functions
│   ├── grpo_monitoring_callback.py        # Monitoring
│   └── rollout_logging_callback.py        # Rollout logging
│
├── scripts/                   # Training scripts
│   ├── run_grpo_action_reward.sh          # Action reward strategy
│   ├── run_grpo_shaped_f1.sh              # Shaped F1 strategy
│   ├── run_grpo_multi_component.sh        # Multi-component strategy
│   ├── run_all_f1_strategies.sh           # Parallel training
│   └── resume_*.sh                        # Resume training scripts
│
├── inference/                 # Inference code
│   ├── infer_grpo.py                      # Model inference
│   ├── evaluate_denoise_model.py          # Model evaluation
│   └── merge_model.py                     # Merge LoRA
│
├── data_processing/           # Data processing ⭐
│   ├── README_DATA.md                      # Complete data processing guide
│   ├── generate_noise_data.py
│   ├── encode_with_description.py
│   ├── check_data.py
│   └── process/                           # Core data construction tools
│       ├── create_train_with_description_from_csv.py  # Training data construction ⭐
│       ├── create_test_with_description_from_csv.py   # Test data construction ⭐
│       └── prompts.py                     # Prompt templates
│
├── sasrec/                    # SASRec related
│   ├── sasrec_reward.py
│   └── SASRec_code/
│
└── docs/                      # Documentation
    ├── CURRENT_TRAINING_SUMMARY.md
    ├── F1_STRATEGIES_COMPARISON.md
    └── ...
```

---

## 🎯 Reward Strategy System

### Basic Stage: Three F1 Reward Strategies

#### 1. Action Reward ⭐ Recommended
- **Features**: Encourages the model to attempt denoising
- **Weight**: 2.0
- **Applicable**: Pursuing high coverage
- **Target**: F1 reaches 0.7+

#### 2. Shaped F1
- **Features**: Smooth reward curve
- **Weight**: 1.5
- **Applicable**: Pursuing stable training

#### 3. Multi-Component
- **Features**: Comprehensive evaluation of P+R+F1
- **Weight**: 1.5
- **Applicable**: Need fine-grained control

For details, see: `docs/F1_STRATEGIES_COMPARISON.md`

---

### Advanced Stage: SASRec Reward (Enable after F1>0.7)

**⚠️ Important**: Only enable after F1 reaches 0.7+!

#### Strategy 1: NDCG Mode
- **Coverage**: ~11% (sparse)
- **Focus**: Top-50 accuracy
- **Applicable**: Simple datasets

#### Strategy 2: Ranking Mode ⭐⭐⭐ Highly Recommended
- **Coverage**: ~95% (dense)
- **Focus**: Overall ranking improvement
- **Features**: Gradient clipping, more stable
- **Applicable**: Most scenarios

For details, see: `sasrec/README_SASREC.md`

---

### Recommended Training Workflow

```bash
# Stage 1: Basic Training (F1: 0 → 0.7)
bash scripts/run_grpo_action_reward.sh  # Only F1 reward

# Stage 2: Advanced Optimization (F1: 0.7 → 0.8+)
# Enable SASRec (ranking mode)
```

---

## 📊 Training Configuration

### Default Hyperparameters

```python
batch_size = 4
gradient_accumulation_steps = 4
learning_rate = 5e-6
max_steps = 500
lora_r = 32
load_in_4bit = True
```

### Modify Configuration

Edit the corresponding shell script, for example `scripts/run_grpo_action_reward.sh`

---

## 💡 Usage Tips

### 1. Monitor Training

**TensorBoard**:
```bash
tensorboard --logdir outputs/ --port 6006
```

**View Logs**:
```bash
tail -f logs/resume_*.log
```

### 2. Resume from Checkpoint

```bash
bash scripts/resume_grpo_action_reward.sh
```

### 3. View Rollout Samples

Training automatically prints samples every 100 steps to:
```
outputs/*/rollout_samples/
```

---

## 🔧 FAQ

### Q1: How to resume after training interruption?
A: Use `resume_*.sh` scripts, which will automatically resume from the latest checkpoint

### Q2: How to choose F1 strategy?
A: 
- Need high recall → Action reward
- Need stable training → Shaped F1
- Need balanced P/R → Multi-component

### Q3: What to do if running out of memory?
A: 
- Reduce `batch_size`
- Increase `gradient_accumulation_steps`
- Use `load_in_4bit=True`

### Q4: How to evaluate model performance?
A: 
```bash
python inference/evaluate_denoise_model.py \
    --model_path outputs/grpo_*/checkpoint-500 \
    --test_file test_data.json
```

---

## 📚 Detailed Documentation

- `docs/CURRENT_TRAINING_SUMMARY.md` - Current training summary
- `docs/F1_STRATEGIES_COMPARISON.md` - Strategy comparison
- `docs/ROLLOUT_LOGGING_GUIDE.md` - Rollout feature guide
- `docs/DATA_PREPARATION_GUIDE.md` - Data preparation
- `docs/F1_IMPROVEMENT_GUIDE.md` - F1 improvement guide

