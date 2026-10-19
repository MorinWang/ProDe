import os
import json
import glob
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--metrics", default="NDCG@20,HR@20,NDCG@50,HR@50")
    args = ap.parse_args()

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    files = sorted(glob.glob(os.path.join(args.results_dir, "*.json")))
    rows = []
    for path in files:
        name = os.path.splitext(os.path.basename(path))[0]
        if name.endswith("_matrix") or name == "summary":
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        rows.append((name, {m: data.get(m) for m in metrics}))

    width = max([len(r[0]) for r in rows] + [8])
    header = f"{'variant':<{width}}" + "".join(f"{m:>12}" for m in metrics)
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for name, vals in rows:
        cells = "".join(
            (f"{vals[m]:>12.4f}" if isinstance(vals[m], (int, float)) else f"{'--':>12}")
            for m in metrics
        )
        print(f"{name:<{width}}" + cells)
    print("=" * len(header))

    out = {name: vals for name, vals in rows}
    out_path = os.path.join(args.results_dir, "summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[aggregate] wrote {out_path}")


if __name__ == "__main__":
    main()
