import re
from typing import List, Set
import numpy as np


def extract_titles(text: str, section: str) -> List[str]:
    
    pattern = rf'{section}[:\s]+(.+?)(?=Suggested|$)'
    match = re.search(pattern, text, re.DOTALL | re.I)
    if not match:
        return []
    
    titles = []
    content = match.group(1)
    for match in re.finditer(r'"([^"]+)"', content):
        titles.append(match.group(1).strip())
    return titles


def precision_recall_reward_func(completions, gt_noises, **kwargs) -> List[float]:
    
    rewards = []
    
    for content, gt in zip(completions, gt_noises):
        res_text = content[0]["content"] if isinstance(content, list) else content
        pred_noises = extract_titles(res_text, "Noise Items")
        
        
        pred_set = set([p.lower().strip() for p in pred_noises])
        gt_set = set([g.lower().strip() for g in gt])
        
        
        if not gt_set:
            reward = 1.0 if not pred_set else 0.0
            rewards.append(reward)
            continue
        
        
        if not pred_set:
            rewards.append(0.0)
            continue
        
        
        tp = len(pred_set & gt_set)
        precision = tp / len(pred_set) if pred_set else 0.0
        recall = tp / len(gt_set) if gt_set else 0.0
        
        
        reward = 0.4 * precision + 0.6 * recall
        rewards.append(reward)
    
    return rewards


def shaped_f1_reward_func(completions, gt_noises, **kwargs) -> List[float]:
    
    rewards = []
    
    for content, gt in zip(completions, gt_noises):
        res_text = content[0]["content"] if isinstance(content, list) else content
        pred_noises = extract_titles(res_text, "Noise Items")
        
        
        pred_set = set([p.lower().strip() for p in pred_noises])
        gt_set = set([g.lower().strip() for g in gt])
        
        if not gt_set:
            rewards.append(1.0 if not pred_set else 0.0)
            continue
        
        
        if not pred_set:
            rewards.append(-0.1)
            continue
        
        tp = len(pred_set & gt_set)
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)
        
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
        
        
        shaped_reward = np.sqrt(f1)
        
        
        if tp > 0:
            shaped_reward += 0.1
        
        rewards.append(shaped_reward)
    
    return rewards


def partial_match_reward_func(completions, gt_noises, **kwargs) -> List[float]:
    
    rewards = []
    
    for content, gt in zip(completions, gt_noises):
        res_text = content[0]["content"] if isinstance(content, list) else content
        pred_noises = extract_titles(res_text, "Noise Items")
        
        
        pred_set = set([p.lower().strip() for p in pred_noises])
        gt_set = set([g.lower().strip() for g in gt])
        
        if not gt_set:
            rewards.append(1.0 if not pred_set else 0.0)
            continue
        
        if not pred_set:
            rewards.append(0.0)
            continue
        
        tp = len(pred_set & gt_set)
        
        if tp == len(gt_set) and len(pred_set) == len(gt_set):
            
            reward = 1.0
        elif tp > 0:
            
            recall = tp / len(gt_set)
            precision = tp / len(pred_set)
            reward = 0.3 + 0.7 * (recall * precision)
        else:
            
            reward = 0.1
        
        rewards.append(reward)
    
    return rewards


def curriculum_f1_reward_func(completions, gt_noises, current_step=0, total_steps=1000, **kwargs) -> List[float]:
    
    progress = current_step / total_steps if total_steps > 0 else 1.0
    rewards = []
    
    for content, gt in zip(completions, gt_noises):
        res_text = content[0]["content"] if isinstance(content, list) else content
        pred_noises = extract_titles(res_text, "Noise Items")
        
        pred_set = set(pred_noises)
        gt_set = set(gt)
        
        if not gt_set:
            rewards.append(1.0 if not pred_set else 0.0)
            continue
        
        if not pred_set:
            rewards.append(0.0)
            continue
        
        tp = len(pred_set & gt_set)
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)
        
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
        
        
        if progress < 0.3:
            
            reward = 0.5 + 0.5 * (tp / len(gt_set))
        elif progress < 0.7:
            
            reward = np.sqrt(f1)
        else:
            
            reward = f1
        
        rewards.append(reward)
    
    return rewards


def denoising_action_reward_func(completions, gt_noises, **kwargs) -> List[float]:
    
    rewards = []
    
    for content, gt in zip(completions, gt_noises):
        res_text = content[0]["content"] if isinstance(content, list) else content
        pred_noises = extract_titles(res_text, "Noise Items")
        
        
        pred_set = set([p.lower().strip() for p in pred_noises])
        gt_set = set([g.lower().strip() for g in gt])
        
        if not gt_set:
            rewards.append(1.0 if not pred_set else 0.0)
            continue
        
        
        if not pred_set:
            
            rewards.append(-0.2)
            continue
        
        
        action_reward = 0.2
        
        
        tp = len(pred_set & gt_set)
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
        
        
        total_reward = action_reward + 0.8 * f1
        rewards.append(total_reward)
    
    return rewards


def adaptive_f1_reward_func(completions, gt_noises, **kwargs) -> List[float]:
    
    rewards = []
    
    for content, gt in zip(completions, gt_noises):
        res_text = content[0]["content"] if isinstance(content, list) else content
        pred_noises = extract_titles(res_text, "Noise Items")
        
        
        pred_set = set([p.lower().strip() for p in pred_noises])
        gt_set = set([g.lower().strip() for g in gt])
        
        if not gt_set:
            rewards.append(1.0 if not pred_set else 0.0)
            continue
        
        if not pred_set:
            rewards.append(0.0)
            continue
        
        tp = len(pred_set & gt_set)
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)
        
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
        
        
        difficulty = len(gt_set)
        
        if difficulty == 1:
            
            reward = f1
        elif difficulty == 2:
            
            reward = 0.2 + 0.8 * f1
        else:
            
            reward = 0.4 + 0.6 * f1
        
        rewards.append(reward)
    
    return rewards


def multi_component_f1_reward(completions, gt_noises, **kwargs) -> List[float]:
    
    rewards = []
    
    for content, gt in zip(completions, gt_noises):
        res_text = content[0]["content"] if isinstance(content, list) else content
        pred_noises = extract_titles(res_text, "Noise Items")
        
        
        pred_set = set([p.lower().strip() for p in pred_noises])
        gt_set = set([g.lower().strip() for g in gt])
        
        if not gt_set:
            rewards.append(1.0 if not pred_set else 0.0)
            continue
        
        
        if not pred_set:
            action_reward = -0.2
            pr_reward = 0.0
            f1_reward = 0.0
        else:
            action_reward = 0.2
            
            tp = len(pred_set & gt_set)
            fp = len(pred_set - gt_set)
            fn = len(gt_set - pred_set)
            
            
            f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
            f1_reward = np.sqrt(f1)  
            
            
            precision = tp / len(pred_set) if pred_set else 0.0
            recall = tp / len(gt_set) if gt_set else 0.0
            pr_reward = 0.4 * precision + 0.6 * recall
        
        
        total_reward = 0.5 * f1_reward + 0.2 * action_reward + 0.3 * pr_reward
        rewards.append(max(0.0, total_reward))  
    
    return rewards
