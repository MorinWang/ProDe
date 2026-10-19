
import os
import json
import argparse

BACKBONES = ["gru4rec", "sasrec", "bert4rec"]
PRETTY = {"gru4rec": "GRU4Rec", "sasrec": "SASRec", "bert4rec": "BERT4Rec"}


def _load_metric(path, metric):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if metric in data:
        return float(data[metric])
    for key in ("result", "results", "metrics", "eval"):
        if key in data and isinstance(data[key], dict) and metric in data[key]:
            return float(data[key][metric])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--metric", default="NDCG@20")
    args = ap.parse_args()

    none = {eb: _load_metric(os.path.join(args.results_dir, f"none_eval-{eb}.json"), args.metric)
            for eb in BACKBONES}

    matrix = {rb: {eb: _load_metric(
        os.path.join(args.results_dir, f"reward-{rb}_eval-{eb}.json"), args.metric)
        for eb in BACKBONES} for rb in BACKBONES}

    header = f"{'Reward\\Eval':<12}" + "".join(f"{PRETTY[eb]:>22}" for eb in BACKBONES)
    print("\n" + "=" * len(header))
    print(f"Reward-Evaluator matrix  ({args.metric})")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for rb in BACKBONES:
        cells = []
        for eb in BACKBONES:
            v = matrix[rb][eb]
            base = none[eb]
            if v is None:
                cells.append(f"{'--':>22}")
            elif base:
                rel = (v - base) / base * 100.0
                mark = "†" if rb == eb else " "
                cells.append(f"{f'{v:.4f}{mark} (+{rel:.1f}%)':>22}")
            else:
                cells.append(f"{v:>22.4f}")
        print(f"{PRETTY[rb]:<12}" + "".join(cells))

    print("-" * len(header))
    base_row = f"{'None':<12}" + "".join(
        f"{(f'{none[eb]:.4f}' if none[eb] is not None else '--'):>22}" for eb in BACKBONES)
    print(base_row)
    print("=" * len(header))
    print("† diagonal: reward and evaluation share the same architecture.\n")

    out = {"metric": args.metric, "none_baseline": none, "matrix": matrix}
    out_path = os.path.join(args.results_dir, "reward_decouple_matrix.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[aggregate] wrote {out_path}")


if __name__ == "__main__":
    main()
