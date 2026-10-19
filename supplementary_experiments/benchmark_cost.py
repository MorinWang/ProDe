import os
import ast
import json
import time
import argparse

import pandas as pd

from denoise_trainset import build_prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy_model", required=True)
    ap.add_argument("--source_dir", required=True)
    ap.add_argument("--id2name", required=True)
    ap.add_argument("--id2desc", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--num_seqs", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=16)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu_mem", type=float, default=0.8)
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--cuda", type=str, default="0")
    args = ap.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.cuda)
    os.environ.setdefault("VLLM_USE_V1", "0")

    with open(args.id2name, "r", encoding="utf-8") as f:
        id2name = {int(k): v for k, v in json.load(f).items()}
    with open(args.id2desc, "r", encoding="utf-8") as f:
        id2desc = {int(k): v for k, v in json.load(f).items()}

    df = pd.read_csv(os.path.join(args.source_dir, "train.csv"))
    df["item_ids"] = df["item_ids"].apply(ast.literal_eval)
    rows = df["item_ids"].tolist()[: args.num_seqs]
    prompts = [build_prompt(seq, id2name, id2desc) for seq in rows]

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.policy_model, tensor_parallel_size=args.tp,
              gpu_memory_utilization=args.gpu_mem, trust_remote_code=True,
              dtype="bfloat16", max_model_len=2048, enforce_eager=True)
    sp = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=args.max_tokens,
                        repetition_penalty=1.1, stop=["<|endoftext|>", "###"])

    if args.warmup > 0:
        llm.generate(prompts[: args.warmup], sp)

    start = time.time()
    outputs = llm.generate(prompts, sp)
    wall = time.time() - start

    n = len(outputs)
    ms_per_seq = wall / n * 1000.0 if n else 0.0
    seq_per_hour = n / wall * 3600.0 if wall > 0 else 0.0

    result = {
        "policy_model": args.policy_model,
        "num_sequences": n,
        "wall_seconds": round(wall, 3),
        "latency_ms_per_seq": round(ms_per_seq, 2),
        "throughput_seq_per_hour": int(seq_per_hour),
        "max_tokens": args.max_tokens,
        "tensor_parallel_size": args.tp,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
