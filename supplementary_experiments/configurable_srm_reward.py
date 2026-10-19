import re
from typing import List, Optional

from recmodel_reward import RecModelRewardCalculator, SRMConfig


class ConfigurableSRMRewardCalculator:
    def __init__(
        self,
        model_path: Optional[str],
        id2name_path: str,
        item_num: int,
        backbone: str = "sasrec",
        tau1: int = 50,
        tau2: int = 200,
        constant_mode: str = "default",
        maxlen: int = 10,
        hidden_units: int = 64,
        num_blocks: int = 2,
        num_heads: int = 1,
        dropout_rate: float = 0.5,
        gru_layers: int = 1,
        device: str = "cuda",
        topk: int = 50,
    ):
        self.calc = RecModelRewardCalculator(
            backbone=backbone, model_path=model_path, id2name_path=id2name_path,
            item_num=item_num, maxlen=maxlen, hidden_units=hidden_units,
            num_blocks=num_blocks, num_heads=num_heads, dropout_rate=dropout_rate,
            gru_layers=gru_layers, device=device, topk=topk,
        )
        self.srm = SRMConfig(tau1=tau1, tau2=tau2, constant_mode=constant_mode)
        self.backbone = backbone

    def compute(self, original_input: str, generated_response: str) -> float:
        titles = self.calc.parse_item_sequence(original_input)
        if len(titles) < 2:
            return 0.0
        input_titles, target_title = titles[:-1], titles[-1]
        input_ids = self.calc.titles_to_ids(input_titles)
        target_id = self.calc.ground_title_to_id(target_title)
        if target_id == 0:
            return 0.0
        _, rank_before, _ = self.calc.compute_ranking_and_reward(input_ids, target_id)
        noise_titles, suggested_titles = self.calc.extract_titles_from_response(generated_response)
        if not noise_titles or not suggested_titles:
            return 0.0
        denoised = self.calc.apply_denoising(input_titles, noise_titles, suggested_titles)
        denoised_ids = self.calc.titles_to_ids(denoised)
        _, rank_after, _ = self.calc.compute_ranking_and_reward(denoised_ids, target_id)
        return float(self.srm.reward(rank_before, rank_after))


def srm_reward_func(
    completions: List,
    prompts: Optional[List[str]] = None,
    calculator: Optional[ConfigurableSRMRewardCalculator] = None,
    **kwargs,
) -> List[float]:
    if calculator is None:
        raise ValueError("ConfigurableSRMRewardCalculator must be provided!")
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
            rewards.append(calculator.compute(original_input, generated_text))
        except Exception:
            rewards.append(0.0)
    return rewards
