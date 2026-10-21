
import os
import re
import ast
import json
import shutil
import argparse

import pandas as pd


def build_prompt(item_ids, id2name, id2desc):
    instruction = (
        "You are to analyze a list of item titles and descriptions provided by a user. "
        "Your task is to identify any item(s) that do not align with the main interests "
        "reflected by the majority of the items. After identifying these noise items, "
        "suggest replaced items that better match the user's interests."
    )
    formatted = []
    for iid in item_ids:
        title = id2name.get(iid, "Unknown Title")
        desc = id2desc.get(iid, "")
        formatted.append(f'"{title}: {desc}"' if desc else f'"{title}"')
    history = "User Interaction Sequence: " + ", ".join(formatted)
    return (
        "Below is an instruction that describes a task, paired with an input that "
        "provides further context. Write a response that appropriately completes the "
        f"request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{history}\n\n"
        "### Response:\nNoise Items:"
    )


def parse_titles(resp):
    noise_m = re.search(
        r"Noise Items:\s*(.*?)(?=\n?(?:Suggested|Alternative|Recommended|Alternate|Replacement|Replacing)|$)",
        resp, re.S | re.I)
    sug_m = re.search(
        r"(?:Suggested|Alternative|Recommended|Alternate|Recommended Replacing)\s*"
        r"(?:Alternatives|Replacements|Replacement|Items|Item|Replacing Item)?:\s*(.*)",
        resp, re.S | re.I)
    noise = re.findall(r'"(.*?)"', noise_m.group(1)) if noise_m else []
    sug = re.findall(r'"(.*?)"', sug_m.group(1)) if sug_m else []
    norm = lambda t: t.split(":")[0].strip().lower()
    return [norm(t) for t in noise if t.strip()], [norm(t) for t in sug if t.strip()]


def apply_denoise_ids(item_ids, noise_titles, sug_titles, name2id, max_denoise_num):
    if not noise_titles or not sug_titles:
        return list(item_ids)
    out = list(item_ids)
    n = min(len(noise_titles), len(sug_titles), max_denoise_num)
    for i in range(n):
        nid = name2id.get(noise_titles[i])
        sid = name2id.get(sug_titles[i])
        if nid is None or sid is None:
            continue
        try:
            out[out.index(nid)] = sid
        except ValueError:
            continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy_model", required=True, help="merged GRPO policy (vLLM-loadable)")
    ap.add_argument("--source_dir", required=True, help="dir with train.csv & test_5000.csv")
    ap.add_argument("--id2name", required=True)
    ap.add_argument("--id2desc", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_denoise_num", type=int, default=3)
    ap.add_argument("--tp", type=int, default=1, help="tensor_parallel_size")
    ap.add_argument("--gpu_mem", type=float, default=0.8)
    ap.add_argument("--cuda", type=str, default="0")
    args = ap.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.cuda)
    os.environ.setdefault("VLLM_USE_V1", "0")

    with open(args.id2name, "r", encoding="utf-8") as f:
        id2name = {int(k): v for k, v in json.load(f).items()}
    with open(args.id2desc, "r", encoding="utf-8") as f:
        id2desc = {int(k): v for k, v in json.load(f).items()}
    name2id = {v.strip().lower(): k for k, v in id2name.items()}

    train_csv = os.path.join(args.source_dir, "train.csv")
    df = pd.read_csv(train_csv)
    df["item_ids"] = df["item_ids"].apply(ast.literal_eval)

    prompts = [build_prompt(seq, id2name, id2desc) for seq in df["item_ids"].tolist()]

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.policy_model, tensor_parallel_size=args.tp,
              gpu_memory_utilization=args.gpu_mem, trust_remote_code=True,
              dtype="bfloat16", max_model_len=2048, enforce_eager=True)
    sp = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=128,
                        stop=["<|endoftext|>", "###"])

    print(f"[denoise] generating for {len(prompts)} training sequences ...")
    outputs = llm.generate(prompts, sp)

    denoised = []
    for row_items, out in zip(df["item_ids"].tolist(), outputs):
        resp = "Noise Items:" + out.outputs[0].text
        noise_t, sug_t = parse_titles(resp)
        denoised.append(apply_denoise_ids(row_items, noise_t, sug_t, name2id, args.max_denoise_num))

    os.makedirs(args.out_dir, exist_ok=True)
    out_df = pd.DataFrame({
        "user_id": df["user_id"].astype(int).tolist(),
        "item_ids": [str(x) for x in denoised],
        "orig_item_ids": [str(x) for x in df["item_ids"].tolist()],
    })
    out_df.to_csv(os.path.join(args.out_dir, "train.csv"), index=False)

    shutil.copy(os.path.join(args.source_dir, "test_5000.csv"),
                os.path.join(args.out_dir, "test_5000.csv"))
    try:
        shutil.copy(args.id2name, os.path.join(args.out_dir, "id2name.json"))
    except Exception:
        pass

    changed = sum(1 for a, b in zip(denoised, df["item_ids"].tolist()) if a != b)
    print(f"[denoise] wrote {args.out_dir}/train.csv "
          f"({changed}/{len(denoised)} sequences modified)")


if __name__ == "__main__":
    main()
