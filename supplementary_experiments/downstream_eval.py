
import os
import sys
import copy
import json
import math
import argparse

import numpy as np
import torch

PRODE_ROOT = os.environ.get(
    "PRODE_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
_SASREC_CODE_DIR = os.path.join(PRODE_ROOT, "sasrec", "SASRec_code")
if _SASREC_CODE_DIR not in sys.path:
    sys.path.insert(0, _SASREC_CODE_DIR)

MODEL_ALIASES = {"sasrec": "SASRec", "bert4rec": "Bert4Rec", "gru4rec": "GRU"}


def _set_world_config(args, item_num=None):
    _orig_argv = sys.argv.copy()
    sys.argv = [sys.argv[0]]
    try:
        import world
    finally:
        sys.argv = _orig_argv
    world.config = {
        "model": MODEL_ALIASES[args.model],
        "dataset": os.path.basename(args.dataset_dir.rstrip("/")),
        "batchsize": args.batchsize,
        "lr": args.lr,
        "hidden_units": args.hidden_units,
        "num_blocks": args.num_blocks,
        "num_heads": args.num_heads,
        "num_epochs": args.num_epochs,
        "dropout_rate": args.dropout_rate,
        "weight_decay": 0.0,
        "maxlen": args.maxlen,
        "state_dict_path": None,
        "cuda": args.cuda,
        "train_noise_filter_threshold": 0,
        "test_noise_filter_threshold": 0,
        "train_max_denoise_num": 3,
        "test_max_denoise_num": 3,
        "denoise_category": "none",
    }
    return world


def _gru_batch(item_id_lists, maxlen, device):
    inputs, targets = [], []
    for items in item_id_lists:
        items = [int(x) for x in items if int(x) != 0]
        if len(items) < 2:
            continue
        target = items[-1]
        hist = items[:-1][-maxlen:]
        hist = hist + [0] * (maxlen - len(hist))
        inputs.append(hist)
        targets.append(target)
    if not inputs:
        return None, None
    return (torch.tensor(inputs, dtype=torch.long, device=device),
            torch.tensor(targets, dtype=torch.long, device=device))


def _metrics_from_ranks(ranks, sample_total, topk_list=(1, 5, 10, 20, 50)):
    out = {}
    ranks = ranks.float()
    for k in topk_list:
        hit = (ranks < k)
        out[f"HR@{k}"] = round((hit.sum().item()) / sample_total, 4)
        ndcg = (1.0 / torch.log(ranks[hit] + 2)).sum().item()
        out[f"NDCG@{k}"] = round(ndcg / (sample_total * (1.0 / math.log(2))), 4)
    ordered = {}
    for k in topk_list:
        ordered[f"NDCG@{k}"] = out[f"NDCG@{k}"]
    for k in topk_list:
        ordered[f"HR@{k}"] = out[f"HR@{k}"]
    return ordered


@torch.no_grad()
def _gru_evaluate(model, test_df, maxlen, device, batch=256):
    model.eval()
    all_ranks = []
    rows = test_df["item_ids"].tolist()
    for s in range(0, len(rows), batch):
        inp, tgt = _gru_batch(rows[s:s + batch], maxlen, device)
        if inp is None:
            continue
        logits = -model.predict(inp)
        rank = logits.argsort(-1).argsort(-1)
        all_ranks.append(rank[torch.arange(rank.size(0)), tgt].cpu())
    ranks = torch.cat(all_ranks) if all_ranks else torch.tensor([])
    return _metrics_from_ranks(ranks, len(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True, help="dir with train.csv & test_5000.csv")
    ap.add_argument("--model", required=True, choices=list(MODEL_ALIASES.keys()))
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--save_ckpt", default=None,
                    help="optional path to save the best state_dict (e.g. to reuse as a "
                         "frozen reward-model checkpoint)")
    ap.add_argument("--num_epochs", type=int, default=201)
    ap.add_argument("--batchsize", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--maxlen", type=int, default=10)
    ap.add_argument("--hidden_units", type=int, default=64)
    ap.add_argument("--num_blocks", type=int, default=2)
    ap.add_argument("--num_heads", type=int, default=1)
    ap.add_argument("--dropout_rate", type=float, default=0.5)
    ap.add_argument("--gru_layers", type=int, default=1)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--cuda", type=str, default="0")
    ap.add_argument("--seed", type=int, default=2024)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _set_world_config(args)
    import utils
    from dataloader import Data_Pro, Seq_dataset, bert_masking
    from model import SASRec, Bert4Rec, GRU
    from torch.utils.data import DataLoader

    utils.set_seed(args.seed)

    data_pro = Data_Pro(args.dataset_dir, train_denoise_flag=False, test_denoise_flag=False)
    train_df, test_df = data_pro.get_data_df()
    item_num = data_pro.item_num
    print(f"[downstream] {args.model} on {args.dataset_dir}  item_num={item_num}")

    if args.model == "sasrec":
        model = SASRec(item_num).to(device)
    elif args.model == "bert4rec":
        model = Bert4Rec(item_num).to(device)
    else:
        model = GRU(args.hidden_units, item_num, gru_layers=args.gru_layers).to(device)

    bce = torch.nn.BCEWithLogitsLoss()
    ce = torch.nn.CrossEntropyLoss(ignore_index=0)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)

    train_loader = DataLoader(Seq_dataset(train_df), batch_size=args.batchsize,
                              shuffle=True, num_workers=4)
    test_loader = DataLoader(Seq_dataset(test_df), batch_size=args.batchsize,
                             shuffle=False, num_workers=4)
    train_rows = train_df["item_ids"].tolist()

    best_ndcg, patience, best_state, best_metrics = 0.0, 0, None, None
    for epoch in range(args.num_epochs):
        model.train()
        if args.model == "gru4rec":
            import random
            random.shuffle(train_rows)
            for s in range(0, len(train_rows), args.batchsize):
                inp, tgt = _gru_batch(train_rows[s:s + args.batchsize], args.maxlen, device)
                if inp is None:
                    continue
                logits = model.predict(inp)
                loss = ce(logits, tgt)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        else:
            for seq in train_loader:
                if args.model == "bert4rec":
                    masked, labels = bert_masking(seq, item_num, mask_prob=0.15)
                    masked, labels = masked.to(device), labels.to(device)
                    logits = model(masked)
                    loss = ce(logits.view(-1, logits.size(-1)), labels.view(-1))
                else:
                    s2, pos, neg = utils.get_negative_items(seq, item_num)
                    pos_logits, neg_logits, _ = model(s2, pos, neg)
                    idx = pos != 0
                    loss = bce(pos_logits[idx], torch.ones_like(pos_logits[idx]))
                    loss += bce(neg_logits[idx], torch.zeros_like(neg_logits[idx]))
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()

        model.eval()
        if args.model == "gru4rec":
            t = _gru_evaluate(model, test_df, args.maxlen, device, args.batchsize)
        else:
            t = utils.evaluate(model, test_loader, len(test_df))
        print(f"[downstream] epoch {epoch}: NDCG@20={t['NDCG@20']} HR@20={t['HR@20']}")

        if t["NDCG@20"] > best_ndcg:
            best_ndcg, patience = t["NDCG@20"], 0
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = t
        else:
            patience += 1
            if patience >= args.patience:
                print(f"[downstream] early stop at epoch {epoch}")
                break

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    payload = dict(best_metrics or {})
    payload.update({"model": args.model, "dataset_dir": args.dataset_dir, "item_num": int(item_num)})
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[downstream] best NDCG@20={best_ndcg} -> {args.out_json}")

    if args.save_ckpt and best_state is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_ckpt)), exist_ok=True)
        torch.save(best_state, args.save_ckpt)
        print(f"[downstream] saved best checkpoint -> {args.save_ckpt}")


if __name__ == "__main__":
    main()
