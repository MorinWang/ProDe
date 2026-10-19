# Supplementary Experiments (Appendix)

Code for the paper's appendix / rebuttal experiments: **reward-model decoupling**
(`sec:reward_decouple`), **reward hyperparameter sensitivity** (`sec:reward_sensitivity`),
**computational cost** (`sec:cost_analysis`), and **cross-intensity noise transfer**
(`sec:noise_generalization`).

This package lives inside the ProDE repo (`ProDE/supplementary_experiments/`). It reuses
the model definitions and training utilities from the parent repo via `PRODE_ROOT`, which
auto-resolves to the parent `ProDE/` directory (override with the `PRODE_ROOT` env var if
needed). The ranking reward here shares the same unified Stratified Reward Mapping (SRM)
used by the main training code (`ProDE/training/sasrec_reward.py`).

## What it does

Train three GRPO denoising policies, each using a different frozen recommender as the
ranking-reward model (GRU4Rec / SASRec / BERT4Rec). Denoise the training set with each
policy, then train **all three** downstream recommenders **from scratch** on each
denoised set — yielding the 3×3 reward–evaluator matrix (NDCG@20 + relative improvement
over each evaluator's noisy-training "None" baseline).

"All other factors are held fixed": same SFT checkpoint, same injected noise, same GRPO
hyper-parameters, and equal reward weights across the three runs — only the reward
backbone changes.

## Files

- `recmodel_reward.py` — **the core new piece.** A generalized `RecModelRewardCalculator`
  that computes the ranking-improvement reward with `backbone ∈ {sasrec, gru4rec, bert4rec}`.
  Reuses the model classes from `ProDE/sasrec/SASRec_code/model.py` unmodified. The
  `backbone='sasrec'` path reproduces the original SASRec reward exactly (same rank bins,
  same `tanh` mapping), so the SASRec-reward row matches the main-paper ProDe results.
- `train_grpo_decouple.py` — GRPO training entry with `--reward_backbone`; reuses the
  original repo's format/F1/validity reward funcs and callbacks (imported, not copied).
- `denoise_trainset.py` — runs a trained policy (vLLM) over a downstream `train.csv`,
  grounds the predicted noise/replacement titles back to item ids, and writes a denoised
  dataset dir (`train.csv` + copied `test_5000.csv`/`id2name.json`). Keyed by each
  train row so predictions are always aligned (no random sampling).
- `downstream_eval.py` — trains + evaluates one downstream recommender on a dataset dir
  and dumps `{"NDCG@20": ...}` to JSON. **Fixes the repo's broken GRU path**: correct
  `GRU(hidden_units, item_num, gru_layers)` constructor, right-padding for
  `pack_padded_sequence`, and a GRU-appropriate next-item CE trainer + evaluator.
  Optionally saves the best checkpoint (`--save_ckpt`) so it can also produce the frozen
  reward models. SASRec/BERT4Rec reuse the repo's (correct) training loops and
  `utils.evaluate`.
- `aggregate_matrix.py` — reads the 9 downstream results + 3 baselines and prints the
  3×3 matrix (marks the diagonal, same-architecture cells with †).
- `scripts/run_decouple_experiment.sh` — full end-to-end orchestration (Steps 0–5).

## Other supplementary experiments

The same infrastructure drives three more appendix experiments (code added without
comments per request):

- **Reward hyperparameter sensitivity** (`sec:reward_sensitivity`):
  `configurable_srm_reward.py` implements the paper's Stratified Reward Mapping with
  configurable fusion weights, interval thresholds `tau1/tau2`, and constant-deformation
  modes (`default` / `flattened` / `shifted`). `train_grpo_sensitivity.py` trains a policy
  for a given `--w1/--w2/--w3 --tau1 --tau2 --constant_mode`, and
  `scripts/run_sensitivity_experiment.sh` sweeps the 11 variants (weights, thresholds,
  constants) → denoise → downstream eval → `aggregate_generic.py`.
- **Computational cost** (`sec:cost_analysis`): `benchmark_cost.py` measures the
  denoising latency (ms/seq) and throughput (seq/h) of a policy under batched vLLM, and
  writes JSON for the cost table.
- **Cross-intensity noise transfer** (`sec:noise_generalization`):
  `scripts/run_noise_transfer.sh` applies the 10%-noise policy to the 20%/30% noise sets
  (transfer, inference only), trains noise-matched policies, and the "None" baselines,
  all via the existing `denoise_trainset.py` + `downstream_eval.py`, then aggregates.

`aggregate_generic.py` prints any directory of downstream-result JSONs as a labeled
NDCG/HR table.

## Prerequisites

1. The parent `ProDE/` repo, its data, and the SFT checkpoint. `PRODE_ROOT` auto-resolves
   to the parent directory; override the env var only if you relocate this folder.
2. **Frozen reward-model checkpoints for all three backbones**, each trained on the
   *clean* dataset (the same way the existing SASRec reward model was trained). The repo
   ships a SASRec reward checkpoint; GRU4Rec/BERT4Rec ones must be trained (Step 0 of the
   script) using `ProDE/sasrec/SASRec_code/main.py` (`--model SASRec|Bert4Rec`) and
   `main_causal_version.py` (`--model GRU`).

## Quick smoke test (no GPU / checkpoints needed)

Verifies the reward calculator wires up for every backbone with random weights:

```bash
cd ProDE/supplementary_experiments
python recmodel_reward.py \
    --id2name_path ../data/Movie_denoise_user0.1_10000_1400/id2name4Rec.json \
    --item_num 5101 --device cpu
```

## Run the full experiment

Edit the CONFIG block at the top of `scripts/run_decouple_experiment.sh` (dataset,
item_num, paths to the SFT model, id2name/id2desc, and the clean/noisy dataset dirs),
then:

```bash
bash scripts/run_decouple_experiment.sh
```

Steps: (0) train frozen reward models on clean data → (1) train a GRPO policy per reward
backbone → (2) denoise the training set with each policy → (3) train+eval every
downstream backbone on every denoised set → (4) None baselines → (5) aggregate.

`aggregate_matrix.py` reads, from `--results_dir`:

```
reward-<rb>_eval-<eb>.json   # {"NDCG@20": <float>} for rb,eb in {gru4rec,sasrec,bert4rec}
none_eval-<eb>.json          # noisy-training baseline per evaluator
```

Prerequisites you must provide: the clean/noisy dataset dirs in SASRec_code format
(`train.csv` with `user_id,item_ids`; `test_5000.csv`), plus `id2name` / `id2desc`
JSONs and the SFT policy checkpoint. Item ids in the sequences must match `item_num`.

## Notes / correctness

- **Padding differs by backbone.** SASRec/BERT4Rec use left-padding (leading zeros, real
  item at the last position); GRU4Rec uses right-padding because it consumes
  `pack_padded_sequence` (padding at the end). Both `RecModelRewardCalculator` (reward)
  and `downstream_eval.py` (evaluation) handle this automatically.
- **The repo's GRU path was broken** for both reward and downstream use (wrong
  constructor, missing argparse choice, left-padding). This package fixes it in its own
  files without editing the original repo, so GRU can serve as both a reward model and a
  downstream evaluator.
- The reward values in the paper's table are real measurements; this code reproduces the
  pipeline that generates them. Re-run with your trained checkpoints to refresh numbers.
