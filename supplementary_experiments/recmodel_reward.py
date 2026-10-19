
import os
import re
import sys
import json
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch


PRODE_ROOT = os.environ.get(
    "PRODE_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
_SASREC_CODE_DIR = os.path.join(PRODE_ROOT, "sasrec", "SASRec_code")

if _SASREC_CODE_DIR not in sys.path:
    sys.path.insert(0, _SASREC_CODE_DIR)

_original_argv = sys.argv.copy()
sys.argv = [sys.argv[0]]
try:
    import world
    from model import SASRec, GRU, Bert4Rec
finally:
    sys.argv = _original_argv


SUPPORTED_BACKBONES = ("sasrec", "gru4rec", "bert4rec")


def _default_srm_constants():
    return {
        "high": {"base": 0.85, "slope": 0.15, "degrade": 0.7, "b2": 0.6, "b3": 0.2, "b4": 0.0},
        "mid": {"c1": 0.9, "c2": 0.75, "c3": 0.5, "c4": 0.35, "c5": 0.0},
        "low": {"c1": 0.95, "c2": 0.8, "c3": 0.4, "c4": 0.1, "c5": 0.0},
    }


def _apply_srm_mode(consts, mode):
    out = {k: dict(v) for k, v in consts.items()}
    if mode == "shifted":
        for zone in out.values():
            for key, val in zone.items():
                if key == "slope":
                    continue
                if val > 0:
                    zone[key] = max(0.0, round(val - 0.1, 6))
    elif mode == "flattened":
        for zone in out.values():
            level_keys = [k for k in zone if k != "slope"]
            nonzero = [zone[k] for k in level_keys if zone[k] > 0]
            if not nonzero:
                continue
            mean = sum(nonzero) / len(nonzero)
            for k in level_keys:
                if zone[k] > 0:
                    zone[k] = round(mean + 0.5 * (zone[k] - mean), 6)
    return out


class SRMConfig:
    def __init__(self, tau1: int = 50, tau2: int = 200, constant_mode: str = "default"):
        self.tau1 = tau1
        self.tau2 = tau2
        self.constant_mode = constant_mode
        self.c = _apply_srm_mode(_default_srm_constants(), constant_mode)

    def reward(self, rank_before: int, rank_after: int) -> float:
        if rank_before <= 0 or rank_after <= 0:
            return 0.0
        delta = rank_before - rank_after
        if rank_before <= self.tau1:
            h = self.c["high"]
            if delta >= 0 and rank_after <= self.tau1:
                return min(1.0, h["base"] + h["slope"] * min(1.0, delta / 20.0))
            if delta < 0 and rank_after <= self.tau1:
                return h["degrade"]
            if rank_after <= 2 * self.tau1:
                return h["b2"]
            if rank_after <= self.tau2:
                return h["b3"]
            return h["b4"]
        if rank_before <= self.tau2:
            m = self.c["mid"]
            if delta > self.tau1:
                return m["c1"]
            if delta > 0.4 * self.tau1:
                return m["c2"]
            if delta > 0:
                return m["c3"]
            if delta == 0:
                return m["c4"]
            return m["c5"]
        l = self.c["low"]
        if delta > self.tau2:
            return l["c1"]
        if delta > 0.5 * self.tau2:
            return l["c2"]
        if delta > 0:
            return l["c3"]
        if delta == 0:
            return l["c4"]
        return l["c5"]


class RecModelRewardCalculator:

    def __init__(
        self,
        backbone: str,
        model_path: Optional[str],
        id2name_path: str,
        item_num: int,
        maxlen: int = 10,
        hidden_units: int = 64,
        num_blocks: int = 2,
        num_heads: int = 1,
        dropout_rate: float = 0.5,
        gru_layers: int = 1,
        device: str = "cuda",
        topk: int = 50,
        reward_mapping: str = "tanh",
    ):
        backbone = backbone.lower()
        if backbone not in SUPPORTED_BACKBONES:
            raise ValueError(
                f"backbone must be one of {SUPPORTED_BACKBONES}, got '{backbone}'"
            )

        self.backbone = backbone
        self.device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self.maxlen = maxlen
        self.topk = topk
        self.item_num = item_num
        self.reward_mapping = reward_mapping.lower()
        self.srm = SRMConfig()

        self.pad_side = "right" if backbone == "gru4rec" else "left"

        print(f"[RecReward:{backbone}] Loading id2name from {id2name_path} ...")
        with open(id2name_path, "r", encoding="utf-8") as f:
            self.id2name = json.load(f)
        self.title2id = {}
        for item_id, title in self.id2name.items():
            self.title2id[title.strip().lower()] = int(item_id)
        print(
            f"[RecReward:{backbone}] {len(self.id2name)} items, "
            f"{len(self.title2id)} title keys."
        )

        self.model = self._build_model(
            hidden_units=hidden_units,
            num_blocks=num_blocks,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            gru_layers=gru_layers,
        ).to(self.device)

        if model_path is not None:
            print(f"[RecReward:{backbone}] Loading checkpoint {model_path} ...")
            state_dict = torch.load(model_path, map_location=self.device)
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            self.model.load_state_dict(state_dict)
        else:
            print(
                f"[RecReward:{backbone}] WARNING: no checkpoint provided -- using random "
                f"weights (smoke-test only, results are meaningless)."
            )
        self.model.eval()
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"[RecReward:{backbone}] model ready ({n_params:,} params, pad={self.pad_side}).")

    def _build_model(self, hidden_units, num_blocks, num_heads, dropout_rate, gru_layers):
        if self.backbone in ("sasrec", "bert4rec"):
            world.config = {
                "hidden_units": hidden_units,
                "maxlen": self.maxlen,
                "dropout_rate": dropout_rate,
                "num_blocks": num_blocks,
                "num_heads": num_heads,
            }
            if self.backbone == "sasrec":
                return SASRec(self.item_num)
            return Bert4Rec(self.item_num)
        return GRU(hidden_units, self.item_num, gru_layers=gru_layers)

    def ground_title_to_id(self, title: str) -> int:
        return self.title2id.get(title.strip().lower(), 0)

    def titles_to_ids(self, titles: List[str]) -> List[int]:
        return [self.ground_title_to_id(t) for t in titles]

    def parse_item_sequence(self, input_text: str) -> List[str]:
        matches = re.findall(r'"([^"]+)"', input_text)
        cleaned = []
        for match in matches:
            genre_match = re.search(r":\s+([^:]+?)\s+-\s+", match)
            if genre_match:
                title = match[: genre_match.start()].strip()
            else:
                last_colon = match.rfind(": ")
                title = match[:last_colon].strip() if last_colon > 0 else match.strip()
            cleaned.append(title.strip().lower())
        return cleaned

    def extract_titles_from_response(self, text: str) -> Tuple[List[str], List[str]]:
        noise_pat = r"Noise Items:\s*(.*?)(?=\n?(?:Suggested|Alternative|Recommended|###)|$)"
        m = re.search(noise_pat, text, re.S | re.I)
        noise = [t.strip().lower() for t in re.findall(r'"(.*?)"', m.group(1))] if m else []

        sug_pat = r"(?:Suggested|Alternative|Recommended)\s*(?:Alternatives|Replacements|Items)?:\s*(.*)"
        m = re.search(sug_pat, text, re.S | re.I)
        suggested = [t.strip().lower() for t in re.findall(r'"(.*?)"', m.group(1))] if m else []
        return noise, suggested

    def apply_denoising(self, original_titles, noise_titles, suggested_titles) -> List[str]:
        if not noise_titles or not suggested_titles:
            return original_titles.copy()
        denoised = original_titles.copy()
        for i in range(min(len(noise_titles), len(suggested_titles))):
            try:
                idx = denoised.index(noise_titles[i])
                denoised[idx] = suggested_titles[i]
            except ValueError:
                continue
        return denoised

    def _pad(self, item_ids: List[int]) -> List[int]:
        if len(item_ids) > self.maxlen:
            item_ids = item_ids[-self.maxlen:]
        pad = [0] * (self.maxlen - len(item_ids))
        return (pad + item_ids) if self.pad_side == "left" else (item_ids + pad)

    def compute_ranking_and_reward(self, item_ids: List[int], target_id: int) -> Tuple[float, int, float]:
        if target_id == 0 or len([x for x in item_ids if x != 0]) == 0:
            return 0.0, -1, 0.0
        seq = torch.tensor([self._pad(item_ids)], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = -self.model.predict(seq)
        target_score = (-logits[0, target_id]).item()
        rank_tensor = logits.argsort(dim=-1).argsort(dim=-1)
        target_rank = rank_tensor[0, target_id].item() + 1

        if target_rank <= 50:
            reward = 1.0 / np.log2(target_rank + 1)
        elif target_rank <= 200:
            reward = 0.5 / np.log2(target_rank + 1)
        elif target_rank <= 500:
            reward = 0.2 / np.log2(target_rank + 1)
        else:
            reward = 0.1 / np.log2(target_rank + 1)
        return min(reward, 1.0), target_rank, target_score

    def _map_improvement_to_reward(self, improvement: float) -> float:
        imp = float(np.clip(improvement, -0.5, 0.5))
        if self.reward_mapping == "sigmoid":
            return 1.0 / (1.0 + np.exp(-5.0 * imp))
        return 0.5 * (1.0 + np.tanh(10.0 * (imp - 0.1)))

    def compute_ranking_improvement_reward(
        self, original_input: str, generated_response: str
    ) -> Tuple[float, Dict]:
        original_titles = self.parse_item_sequence(original_input)
        if len(original_titles) < 2:
            return 0.0, {"error": "sequence_too_short"}

        input_titles, target_title = original_titles[:-1], original_titles[-1]
        input_ids = self.titles_to_ids(input_titles)
        target_id = self.ground_title_to_id(target_title)
        if target_id == 0:
            return 0.0, {"error": "target_is_ood"}

        reward_before, rank_before, _ = self.compute_ranking_and_reward(input_ids, target_id)

        noise_titles, suggested_titles = self.extract_titles_from_response(generated_response)
        if not noise_titles or not suggested_titles:
            return 0.0, {"error": "no_denoising", "rank_before": rank_before}

        denoised_titles = self.apply_denoising(input_titles, noise_titles, suggested_titles)
        denoised_ids = self.titles_to_ids(denoised_titles)
        reward_after, rank_after, _ = self.compute_ranking_and_reward(denoised_ids, target_id)

        reward = self.srm.reward(rank_before, rank_after)
        debug = {
            "backbone": self.backbone,
            "rank_before": rank_before,
            "rank_after": rank_after,
            "reward": round(float(reward), 4),
        }
        return float(reward), debug


def recmodel_ranking_improvement_reward_func(
    completions: List,
    prompts: Optional[List[str]] = None,
    calculator: Optional[RecModelRewardCalculator] = None,
    **kwargs,
) -> List[float]:
    if calculator is None:
        raise ValueError("RecModelRewardCalculator must be provided!")

    rewards: List[float] = []
    original_inputs = kwargs.get("original_input", None)

    for i, content in enumerate(completions):
        if isinstance(content, list) and content:
            generated_text = content[0].get("content", "")
        elif isinstance(content, str):
            generated_text = content
        else:
            rewards.append(0.0)
            continue

        if original_inputs is not None and i < len(original_inputs):
            original_input = original_inputs[i]
        elif prompts is not None and i < len(prompts):
            m = re.search(r"### Input:\n(.*?)\n\n### Response:", prompts[i], re.DOTALL)
            if not m:
                rewards.append(0.0)
                continue
            original_input = m.group(1)
        else:
            rewards.append(0.0)
            continue

        try:
            reward, _ = calculator.compute_ranking_improvement_reward(
                original_input=original_input, generated_response=generated_text
            )
            rewards.append(reward)
        except Exception as e:
            print(f"[RecReward:{calculator.backbone}] error on sample {i}: {e}")
            rewards.append(0.0)
    return rewards


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--id2name_path", required=True)
    p.add_argument("--item_num", type=int, required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    test_input = '"Movie A: Action", "Movie B: Drama", "Movie C: Comedy"'
    test_resp = 'Noise Items: "Movie B"\nSuggested Alternatives: "Movie D"'
    for bk in SUPPORTED_BACKBONES:
        calc = RecModelRewardCalculator(
            backbone=bk, model_path=None, id2name_path=args.id2name_path,
            item_num=args.item_num, device=args.device,
        )
        r, dbg = calc.compute_ranking_improvement_reward(test_input, test_resp)
        print(f"[{bk}] reward={r:.4f} debug={dbg}")
