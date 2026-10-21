# ProDe — Generative Sequence Denoising with Reinforcement Learning for Robust Sequential Recommendation

ProDe is a two-stage framework that edits noisy user behavior sequences with a large
language model policy. Stage 1 adapts a pretrained LLM by supervised fine-tuning on
synthetically corrupted windows; Stage 2 optimizes the editing policy with Group
Relative Policy Optimization (GRPO) against a fused reward. At deployment, only the
trained policy is used: it edits each downstream training window once, offline, and
the downstream recommender is then trained from scratch on the edited data — no
reward, label, or reference recommender is involved at this stage.

> The training scripts now default to the paper's reported settings (§5.3); you only
> need to supply your own SFT checkpoint, data, catalog mappings, and frozen
> reference-recommender checkpoint. Machine-local paths in the shell scripts must be
> edited before use.

---

## Method in brief

A window consists of a prefix `h` and the next interaction `y`. The policy outputs,
in a fixed two-line format, the items it judges noisy and an in-catalog replacement
for each. Let `M0` be a frozen reference recommender trained on the training split of
the original, uninjected data; it is used only to score edits and is never updated.

The Stage-2 reward `R(w, a)` combines three terms (equal weight `(1,1,1)` in the
paper):

| Paper term | Role | Implementation in this repo |
|---|---|---|
| **Noise-identification reward** `R_noise` | Penalizes missing or false noise detection using the synthetic-corruption labels; active in training only | `f1_reward_type` family in `training/improved_reward_funcs.py` (`shaped` / `action` / `multi`) |
| **Validity reward** `R_valid` | Penalizes off-catalog (hallucinated) replacement strings; requires no labels | `validity_reward_func` in `training/train_grpo_unsloth_with_sasrec.py` |
| **Ranking reward `R_rec` (SRM)** | Graded credit from the rank of `y` under `M0` before/after the edit, via the Stratified Reward Mapping with stratum boundaries `tau1/tau2 = 50/200` | `use_sasrec_reward=True`, `sasrec_reward_mapping="stratified"` (`training/sasrec_reward.py`); the decoupling package implements SRM for all backbones (`supplementary_experiments/recmodel_reward.py`) |

A separate `format_reward_func` encourages the fixed two-line output format during
training and is an auxiliary engineering term, not one of the three paper rewards.

---

## Repository layout

```
ProDE/
├── README.md
├── requirements.txt
│
├── training/
│   ├── train_denoise_model_improved.py   # Stage 1: SFT on synthetically corrupted windows (LoRA)
│   ├── train_grpo_unsloth_with_sasrec.py # Stage 2: GRPO policy optimization (Unsloth + TRL)
│   ├── improved_reward_funcs.py           # R_noise F1 variants
│   ├── sasrec_reward.py                   # SRM ranking reward under the frozen SASRec M0
│   ├── grpo_monitoring_callback.py        # Training monitoring
│   └── rollout_logging_callback.py        # Rollout sample logging
│
├── supplementary_experiments/
│   ├── README.md                          # Appendix experiment guide (start here for these)
│   ├── recmodel_reward.py                 # SRM for GRU4Rec / SASRec / BERT4Rec reference recommenders
│   ├── configurable_srm_reward.py         # Configurable fusion weights, tau1/tau2, constant mode
│   ├── train_grpo_decouple.py             # Reward-backbone decoupling policy training
│   ├── train_grpo_sensitivity.py         # Reward hyperparameter sensitivity policy training
│   ├── denoise_trainset.py                # Offline editing of a downstream train.csv (vLLM)
│   ├── downstream_eval.py                  # Train + evaluate a downstream recommender, dump JSON
│   ├── benchmark_cost.py                  # Denoising latency / throughput measurement
│   ├── aggregate_matrix.py                # 3x3 reward-vs-evaluator matrix
│   ├── aggregate_generic.py               # Generic NDCG/HR aggregation
│   └── scripts/                           # Orchestration shell scripts
│
├── sasrec/SASRec_code/                    # Reference-recommender / downstream backbone code
├── inference/                             # Standalone policy inference and checkpoint tools
├── data_processing/                      # Noise injection and training-set construction helpers
└── scripts/                               # Legacy F1-strategy experiment drivers (not the paper entry)
```

---

## Setup

```bash
pip install -r requirements.txt
```

The package expects: the SFT checkpoint, the training windows, the catalog
mapping (`id2name`, optionally `id2desc`), and a frozen reference-recommender
checkpoint for `R_rec`. Data and checkpoint paths are passed per run; several
shell scripts carry placeholder machine-local paths and must be edited before use.

### Stage 1 — Supervised fine-tuning

`training/train_denoise_model_improved.py` adapts the pretrained LLM on
synthetically corrupted windows with LoRA. It exposes `lora_r`, `lora_alpha`,
`lora_dropout`, `num_epochs`, `learning_rate`, and `batch_size`.

### Stage 2 — GRPO policy optimization

`training/train_grpo_unsloth_with_sasrec.py` (Fire CLI) starts from the SFT
checkpoint. Key switches:

- `--f1_reward_type {shaped,action,multi,none}` selects the `R_noise` variant;
- `R_rec` (SRM ranking reward) is enabled by default (`use_sasrec_reward=True`,
  `sasrec_reward_mapping="stratified"`);
- `validity_reward_func` (`R_valid`) is bound to the catalog and always included;
- `--num_generations`, `--learning_rate`, `--batch_size`,
  `--gradient_accumulation_steps`, `--train_file`, `--model_path`,
  `--output_dir` control the run.

### Offline editing and downstream evaluation

`supplementary_experiments/denoise_trainset.py` runs the trained policy over a
`train.csv`, grounds predicted noise/replacement titles to item ids by exact
catalog match, and writes the edited dataset. `supplementary_experiments/
downstream_eval.py` trains a downstream recommender (`sasrec` / `bert4rec` /
`gru4rec`) on the edited set and writes `{"NDCG@20": ...}` to JSON; the
unedited noisy-training set serves as the `None` baseline.

---

## Reported configuration (paper §5.3)

These values are now the script defaults; the table records them for reference.

| Component | Paper setting |
|---|---|
| Policy | Llama-3.1-8B-Instruct (Llama-3.2-3B for the smaller variant) |
| SFT LoRA | rank 32, scaling 64, dropout 0.05, 3 epochs, lr `5e-4`, batch 64 |
| GRPO group size `G` | 4 |
| KL coefficient `beta` | 0.01 toward the reference policy |
| GRPO lr / schedule | `1e-5`, cosine, warmup ratio 0.1 |
| Optimizer | AdamW `(beta1, beta2)=(0.9, 0.99)`, weight decay 0.1 |
| Prompt length | at most 512 tokens |
| Reward weights | `(1,1,1)`; SRM boundaries `tau1/tau2 = 50/200` |
| Reference recommender `M0` | frozen SASRec trained on the training split |
| Inference | vLLM, temperature 0.7, top-p 0.9, at most 128 new tokens |
| Edit budget | 10–30% of the positions of a window; titles matched exactly |
| Downstream SASRec | emb 64, input length 10 |
| Downstream BERT4Rec | emb 256, 4 blocks, 4 heads |
| Downstream training | Adam, lr `1e-3`, batch 256 |
| Hardware | four 40 GB A100 GPUs |

---

## Appendix experiments

See `supplementary_experiments/README.md` for the four appendix studies, which
share the same infrastructure:

- **Reward-backbone decoupling** — GRU4Rec / SASRec / BERT4Rec as `M0`, 3x3
  reward-vs-evaluator matrix.
- **Reward hyperparameter sensitivity** — fusion weights, `tau1/tau2`, and
  constant-deformation modes (`default` / `flattened` / `shifted`).
- **Computational cost** — batched vLLM denoising latency and throughput.
- **Cross-intensity noise transfer** — a policy trained on 10% noise applied to
  20% / 30% noise sets (inference only).

---

## Citation

```bibtex
@article{prode2026,
  title={Generative Sequence Denoising with Reinforcement Learning for Robust Sequential Recommendation},
  journal={ACM Transactions on Information Systems (TOIS)},
  year={2026}
}
```

## License

MIT License.
