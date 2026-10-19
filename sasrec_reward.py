

import os
import sys
import re
import json
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional


original_argv = sys.argv.copy()
sys.argv = [sys.argv[0]]


script_dir = os.path.dirname(os.path.abspath(__file__))
sasrec_code_dir = os.path.join(script_dir, "SASRec_code")
sys.path.insert(0, sasrec_code_dir)

from model import SASRec


sys.argv = original_argv


class SASRecRewardCalculator:
    
    
    def __init__(
        self,
        sasrec_model_path: str,
        id2name_path: str,
        item_num: int,
        maxlen: int = 10,
        hidden_units: int = 64,
        num_blocks: int = 2,
        num_heads: int = 1,
        dropout_rate: float = 0.5,
        device: str = 'cuda',
        topk: int = 50
    ):
        
        self.device = device
        self.maxlen = maxlen
        self.topk = topk
        
        
        print(f"[SASRecReward] Loading id2name mapping from {id2name_path}...")
        with open(id2name_path, 'r', encoding='utf-8') as f:
            self.id2name = json.load(f)
        
        
        self.title2id = {}
        for item_id, title in self.id2name.items():
            
            normalized_title = title.strip().lower()
            self.title2id[normalized_title] = int(item_id)
        
        print(f"[SASRecReward] Loaded {len(self.id2name)} items")
        print(f"[SASRecReward] Created reverse mapping with {len(self.title2id)} titles")
        
        
        print(f"[SASRecReward] Loading SASRec model from {sasrec_model_path}...")
        
        
        import world
        world.config = {
            "hidden_units": hidden_units,
            "maxlen": maxlen,
            "dropout_rate": dropout_rate,
            "num_blocks": num_blocks,
            "num_heads": num_heads
        }
        
        self.model = SASRec(item_num).to(device)
        state_dict = torch.load(sasrec_model_path, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        print(f"[SASRecReward] SASRec model loaded successfully")
        print(f"[SASRecReward] Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def ground_title_to_id(self, title: str) -> int:
        
        normalized = title.strip().lower()
        return self.title2id.get(normalized, 0)
    
    def extract_titles_from_response(self, text: str) -> Tuple[List[str], List[str]]:
        
        
        noise_pattern = r"Noise Items:\s*(.*?)(?=\n?(?:Suggested|Alternative|Recommended|###)|$)"
        noise_match = re.search(noise_pattern, text, re.S | re.I)
        if noise_match:
            noise_content = noise_match.group(1)
            noise_titles = re.findall(r'"(.*?)"', noise_content)
            noise_titles = [t.strip().lower() for t in noise_titles if t.strip()]
        else:
            noise_titles = []
        
        
        suggest_pattern = r"(?:Suggested|Alternative|Recommended)\s*(?:Alternatives|Replacements|Items)?:\s*(.*)"
        suggest_match = re.search(suggest_pattern, text, re.S | re.I)
        if suggest_match:
            suggest_content = suggest_match.group(1)
            suggested_titles = re.findall(r'"(.*?)"', suggest_content)
            suggested_titles = [t.strip().lower() for t in suggested_titles if t.strip()]
        else:
            suggested_titles = []
        
        return noise_titles, suggested_titles
    
    def parse_item_sequence(self, input_text: str) -> List[str]:
        
        
        matches = re.findall(r'"([^"]+)"', input_text)
        
        cleaned_titles = []
        for match in matches:
            
            
            genre_pattern = r':\s+([^:]+?)\s+-\s+'
            genre_match = re.search(genre_pattern, match)
            
            if genre_match:
                
                genre_start = genre_match.start()
                title = match[:genre_start].strip()
            else:
                
                last_colon_idx = match.rfind(': ')
                if last_colon_idx > 0:
                    title = match[:last_colon_idx].strip()
                else:
                    title = match.strip()
            
            cleaned_titles.append(title.strip().lower())
        
        return cleaned_titles
    
    def apply_denoising(
        self, 
        original_titles: List[str], 
        noise_titles: List[str], 
        suggested_titles: List[str]
    ) -> List[str]:
        
        if not noise_titles or not suggested_titles:
            return original_titles.copy()
        
        denoised = original_titles.copy()
        
        
        num_replacements = min(len(noise_titles), len(suggested_titles))
        
        
        for i in range(num_replacements):
            noise_title = noise_titles[i]
            suggest_title = suggested_titles[i]
            
            
            try:
                noise_idx = denoised.index(noise_title)
                denoised[noise_idx] = suggest_title
            except ValueError:
                
                continue
        
        return denoised
    
    def titles_to_ids(self, titles: List[str]) -> List[int]:
        
        ids = [self.ground_title_to_id(title) for title in titles]
        return ids
    
    def compute_ndcg_for_sequence(self, item_ids: List[int], target_id: int) -> float:
        
        if target_id == 0 or len([x for x in item_ids if x != 0]) == 0:
            return 0.0
        
        
        if len(item_ids) > self.maxlen:
            item_ids = item_ids[-self.maxlen:]
        else:
            item_ids = [0] * (self.maxlen - len(item_ids)) + item_ids
        
        
        seq_tensor = torch.tensor([item_ids], dtype=torch.long, device=self.device)
        
        
        with torch.no_grad():
            logits = -self.model.predict(seq_tensor)  
        
        
        rank_tensor = logits.argsort(dim=-1).argsort(dim=-1)  
        target_rank = rank_tensor[0, target_id].item()
        
        
        if target_rank < self.topk:
            ndcg = 1.0 / np.log2(target_rank + 2)
        else:
            ndcg = 0.0
        
        return ndcg
    
    def compute_ranking_and_reward(self, item_ids: List[int], target_id: int) -> Tuple[float, int, float]:
        
        if target_id == 0 or len([x for x in item_ids if x != 0]) == 0:
            return 0.0, -1, 0.0
        
        
        if len(item_ids) > self.maxlen:
            item_ids = item_ids[-self.maxlen:]
        else:
            item_ids = [0] * (self.maxlen - len(item_ids)) + item_ids
        
        
        seq_tensor = torch.tensor([item_ids], dtype=torch.long, device=self.device)
        
        
        with torch.no_grad():
            logits = -self.model.predict(seq_tensor)  
        
        
        target_score = logits[0, target_id].item()
        
        
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
        
        
        reward = min(reward, 1.0)
        
        return reward, target_rank, target_score
    
    def compute_reward_for_sample(
        self,
        original_input: str,
        generated_response: str
    ) -> Tuple[float, Dict]:
        
        
        original_titles = self.parse_item_sequence(original_input)
        
        if len(original_titles) < 2:
            
            return 0.0, {"error": "sequence_too_short"}
        
        
        input_titles = original_titles[:-1]
        target_title = original_titles[-1]
        
        
        input_ids = self.titles_to_ids(input_titles)
        target_id = self.ground_title_to_id(target_title)
        
        if target_id == 0:
            
            return 0.0, {"error": "target_is_ood"}
        
        
        ndcg_before = self.compute_ndcg_for_sequence(input_ids, target_id)
        
        
        noise_titles, suggested_titles = self.extract_titles_from_response(generated_response)
        
        if not noise_titles or not suggested_titles:
            
            return 0.0, {
                "error": "no_denoising",
                "ndcg_before": ndcg_before
            }
        
        
        denoised_titles = self.apply_denoising(input_titles, noise_titles, suggested_titles)
        denoised_ids = self.titles_to_ids(denoised_titles)
        
        
        ndcg_after = self.compute_ndcg_for_sequence(denoised_ids, target_id)
        
        
        if ndcg_before < 1e-6 and ndcg_after < 1e-6:
            
            return 0.0, {
                "error": "both_ndcg_zero",
                "ndcg_before": ndcg_before,
                "ndcg_after": ndcg_after
            }
        elif ndcg_before < 1e-6:
            
            improvement = ndcg_after
        else:
            
            improvement = (ndcg_after - ndcg_before) / ndcg_before
        
        
        reward = 1.0 / (1.0 + np.exp(-improvement))  
        
        debug_info = {
            "ndcg_before": round(ndcg_before, 4),
            "ndcg_after": round(ndcg_after, 4),
            "improvement": round(improvement, 4),
            "reward": round(reward, 4),
            "num_noises": len(noise_titles),
            "num_suggests": len(suggested_titles),
            "ood_in_input": sum(1 for x in input_ids if x == 0),
            "ood_in_denoised": sum(1 for x in denoised_ids if x == 0)
        }
        
        return reward, debug_info
    
    def compute_ranking_improvement_reward(
        self,
        original_input: str,
        generated_response: str
    ) -> Tuple[float, Dict]:
        
        
        original_titles = self.parse_item_sequence(original_input)
        
        if len(original_titles) < 2:
            return 0.0, {"error": "sequence_too_short"}
        
        
        input_titles = original_titles[:-1]
        target_title = original_titles[-1]
        
        
        input_ids = self.titles_to_ids(input_titles)
        target_id = self.ground_title_to_id(target_title)
        
        if target_id == 0:
            return 0.0, {"error": "target_is_ood"}
        
        
        reward_before, rank_before, score_before = self.compute_ranking_and_reward(input_ids, target_id)
        
        
        noise_titles, suggested_titles = self.extract_titles_from_response(generated_response)
        
        if not noise_titles or not suggested_titles:
            return 0.0, {
                "error": "no_denoising",
                "rank_before": rank_before,
                "reward_before": reward_before
            }
        
        
        denoised_titles = self.apply_denoising(input_titles, noise_titles, suggested_titles)
        denoised_ids = self.titles_to_ids(denoised_titles)
        
        
        reward_after, rank_after, score_after = self.compute_ranking_and_reward(denoised_ids, target_id)
        
        
        improvement = reward_after - reward_before
        
        
        improvement_clipped = np.clip(improvement, -0.5, 0.5)
        
        
        reward = 1.0 / (1.0 + np.exp(-5 * improvement_clipped))
        
        debug_info = {
            "rank_before": rank_before,
            "rank_after": rank_after,
            "reward_before": round(reward_before, 4),
            "reward_after": round(reward_after, 4),
            "improvement": round(improvement, 4),
            "improvement_clipped": round(improvement_clipped, 4),
            "was_clipped": abs(improvement) > 0.5,  
            "reward": round(reward, 4),
            "num_noises": len(noise_titles),
            "num_suggestions": len(suggested_titles),
            "score_before": round(score_before, 4),
            "score_after": round(score_after, 4),
        }
        
        return reward, debug_info


def sasrec_improvement_reward_func(
    completions,
    prompts=None,
    calculator: Optional[SASRecRewardCalculator] = None,
    **kwargs
) -> List[float]:
    
    if calculator is None:
        raise ValueError("SASRecRewardCalculator must be provided!")
    
    rewards = []
    
    for i, content in enumerate(completions):
        
        
        if isinstance(content, list) and len(content) > 0:
            generated_text = content[0].get("content", "")
        elif isinstance(content, str):
            generated_text = content
        else:
            rewards.append(0.0)
            continue
        
        
        if prompts is None or i >= len(prompts):
            rewards.append(0.0)
            continue
        
        original_input = prompts[i]
        
        
        try:
            reward, debug_info = calculator.compute_reward_for_sample(
                original_input=original_input,
                generated_response=generated_text
            )
            rewards.append(reward)
            
            
            if i % 50 == 0:
                print(f"\n[SASRecReward Sample {i}]")
                print(f"  NDCG before: {debug_info.get('ndcg_before', 0):.4f}")
                print(f"  NDCG after:  {debug_info.get('ndcg_after', 0):.4f}")
                print(f"  Improvement: {debug_info.get('improvement', 0):.4f}")
                print(f"  Reward:      {debug_info.get('reward', 0):.4f}")
        except Exception as e:
            print(f"[SASRecReward] Error processing sample {i}: {e}")
            rewards.append(0.0)
    
    return rewards


def sasrec_ranking_improvement_reward_func(
    completions: List,
    prompts: Optional[List[str]] = None,
    calculator: Optional[SASRecRewardCalculator] = None,
    **kwargs
) -> List[float]:
    
    if calculator is None:
        raise ValueError("SASRecRewardCalculator must be provided!")
    
    rewards = []
    
    
    original_inputs = kwargs.get('original_input', None)
    
    for i, content in enumerate(completions):
        
        if isinstance(content, list) and len(content) > 0:
            generated_text = content[0].get("content", "")
        elif isinstance(content, str):
            generated_text = content
        else:
            rewards.append(0.0)
            continue
        
        
        if original_inputs is not None and i < len(original_inputs):
            original_input = original_inputs[i]
        elif prompts is not None and i < len(prompts):
            
            import re
            match = re.search(r'### Input:\n(.*?)\n\n### Response:', prompts[i], re.DOTALL)
            if match:
                original_input = match.group(1)
            else:
                rewards.append(0.0)
                continue
        else:
            rewards.append(0.0)
            continue
        
        
        try:
            reward, debug_info = calculator.compute_ranking_improvement_reward(
                original_input=original_input,
                generated_response=generated_text
            )
            rewards.append(reward)
            
            
            if i % 50 == 0:
                print(f"\n[SASRecRankingReward Sample {i}]")
                if 'error' not in debug_info:
                    print(f"  Rank before: {debug_info.get('rank_before', -1)}")
                    print(f"  Rank after:  {debug_info.get('rank_after', -1)}")
                    improvement = debug_info.get('improvement', 0)
                    was_clipped = debug_info.get('was_clipped', False)
                    if was_clipped:
                        improvement_clipped = debug_info.get('improvement_clipped', 0)
                        print(f"  Improvement: {improvement:.4f} → {improvement_clipped:.4f} (clipped) ✂️")
                    else:
                        print(f"  Improvement: {improvement:.4f}")
                    print(f"  Reward:      {reward:.4f}")
                else:
                    print(f"  Error: {debug_info.get('error', 'unknown')}")
                    
        except Exception as e:
            print(f"\n⚠️ Error computing ranking reward for sample {i}: {e}")
            rewards.append(0.0)
    
    return rewards


if __name__ == "__main__":
    
    calculator = SASRecRewardCalculator(
        sasrec_model_path="models/sasrec/Movie_denoise_user0.1_10000_1400_old/best_model.pth",
        id2name_path="data/Movie_denoise_user0.1_10000_1400_old/id2name.json",
        item_num=5101
    )
    
    
    test_input = '"Movie A: Action", "Movie B: Drama", "Movie C: Comedy"'
    test_response = 'Noise Items: "Movie B"\nSuggested Alternatives: "Movie D"'
    
    reward, debug_info = calculator.compute_reward_for_sample(test_input, test_response)
    print(f"Test reward: {reward}")
    print(f"Debug info: {debug_info}")
