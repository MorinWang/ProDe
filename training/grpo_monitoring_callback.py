

import numpy as np
from transformers import TrainerCallback
from typing import Dict, Any


class GRPOMonitoringCallback(TrainerCallback):
    
    
    def __init__(self):
        super().__init__()
        self.step_count = 0
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        
        if logs is None:
            return
        
        self.step_count += 1
        
        
        metric_mappings = {
            
            "rewards/mean": "episode_reward/mean",
            "rewards/std": "episode_reward/std",
            "rewards/min": "episode_reward/min",
            "rewards/max": "episode_reward/max",
            
            "loss": "total_loss",
            "policy_loss": "policy_loss",
            "value_loss": "value_loss",
            
            "kl": "kl_divergence/mean",
            "kl_div": "kl_divergence/mean",
            
            "entropy": "entropy/mean",
            "policy_entropy": "entropy/mean",
            
            "clip_fraction": "clip_fraction",
            "clipfrac": "clip_fraction",
            
            "response_length": "episode_length/mean",
            "sequence_length": "episode_length/mean",
        }
        
        
        enhanced_logs = {}
        for original_key, friendly_key in metric_mappings.items():
            if original_key in logs:
                enhanced_logs[friendly_key] = logs[original_key]
        
        
        reward_keys = [k for k in logs.keys() if "reward" in k.lower()]
        
        for key in reward_keys:
            if "sasrec" in key.lower():
                
                new_key = f"sasrec_reward/{key.split('/')[-1] if '/' in key else 'value'}"
                enhanced_logs[new_key] = logs[key]
            elif key not in ["rewards/mean", "rewards/std", "rewards/min", "rewards/max"]:
                
                enhanced_logs[f"reward_components/{key}"] = logs[key]
        
        
        if not hasattr(self, 'prev_reward'):
            self.prev_reward = None
        
        if "episode_reward/mean" in enhanced_logs:
            current_reward = enhanced_logs["episode_reward/mean"]
            if self.prev_reward is not None:
                enhanced_logs["episode_reward/delta"] = current_reward - self.prev_reward
            self.prev_reward = current_reward
        
        
        health_checks = {}
        
        
        if "kl_divergence/mean" in enhanced_logs:
            kl = enhanced_logs["kl_divergence/mean"]
            if kl < 0.01:
                health_checks["kl_status"] = 0.0  
            elif 0.01 <= kl <= 0.1:
                health_checks["kl_status"] = 1.0  
            else:
                health_checks["kl_status"] = -1.0  
        
        
        if "clip_fraction" in enhanced_logs:
            clip = enhanced_logs["clip_fraction"]
            if clip < 0.1:
                health_checks["clip_status"] = 0.0  
            elif 0.1 <= clip <= 0.3:
                health_checks["clip_status"] = 1.0  
            else:
                health_checks["clip_status"] = -1.0  
        
        
        if "entropy/mean" in enhanced_logs:
            entropy = enhanced_logs["entropy/mean"]
            if entropy > 3.0:
                health_checks["entropy_status"] = 0.5  
            elif 1.0 <= entropy <= 3.0:
                health_checks["entropy_status"] = 1.0  
            elif 0.5 <= entropy < 1.0:
                health_checks["entropy_status"] = 0.5  
            else:
                health_checks["entropy_status"] = -1.0  
        
        
        for key, value in health_checks.items():
            enhanced_logs[f"health_check/{key}"] = value
        
        
        logs.update(enhanced_logs)
        
        
        if self.step_count % 50 == 0:  
            print("\n" + "="*70)
            print(f"📊 Training Metrics Summary (Step {state.global_step})")
            print("="*70)
            
            if "episode_reward/mean" in enhanced_logs:
                print(f"🎁 Reward:      {enhanced_logs['episode_reward/mean']:.4f} "
                      f"(±{enhanced_logs.get('episode_reward/std', 0):.4f})")
            
            if "kl_divergence/mean" in enhanced_logs:
                kl = enhanced_logs['kl_divergence/mean']
                status = "✅" if 0.01 <= kl <= 0.1 else "⚠️"
                print(f"{status} KL Div:      {kl:.6f}")
            
            if "clip_fraction" in enhanced_logs:
                clip = enhanced_logs['clip_fraction']
                status = "✅" if 0.1 <= clip <= 0.3 else "⚠️"
                print(f"{status} Clip Frac:   {clip:.4f}")
            
            if "entropy/mean" in enhanced_logs:
                print(f"🎲 Entropy:     {enhanced_logs['entropy/mean']:.4f}")
            
            if "policy_loss" in enhanced_logs:
                print(f"📉 Policy Loss: {enhanced_logs['policy_loss']:.4f}")
            
            if "value_loss" in enhanced_logs:
                print(f"📉 Value Loss:  {enhanced_logs['value_loss']:.4f}")
            
            
            sasrec_keys = [k for k in enhanced_logs.keys() if "sasrec" in k.lower()]
            if sasrec_keys:
                print(f"\n🎯 SASRec Metrics:")
                for key in sasrec_keys:
                    print(f"   {key}: {enhanced_logs[key]:.4f}")
            
            print("="*70 + "\n")


class SASRecRewardMonitoringCallback(TrainerCallback):
    
    
    def __init__(self):
        super().__init__()
        self.batch_stats = {
            'rank_before': [],
            'rank_after': [],
            'improvement': [],
            'reward': []
        }
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        
        if logs is None:
            return
        
        
        if self.batch_stats['rank_before']:
            logs['sasrec/rank_before_mean'] = np.mean(self.batch_stats['rank_before'])
            logs['sasrec/rank_before_median'] = np.median(self.batch_stats['rank_before'])
            logs['sasrec/rank_before_min'] = np.min(self.batch_stats['rank_before'])
            
            logs['sasrec/rank_after_mean'] = np.mean(self.batch_stats['rank_after'])
            logs['sasrec/rank_after_median'] = np.median(self.batch_stats['rank_after'])
            logs['sasrec/rank_after_min'] = np.min(self.batch_stats['rank_after'])
            
            logs['sasrec/improvement_mean'] = np.mean(self.batch_stats['improvement'])
            logs['sasrec/improvement_std'] = np.std(self.batch_stats['improvement'])
            logs['sasrec/improvement_positive_rate'] = np.mean([x > 0 for x in self.batch_stats['improvement']])
            
            logs['sasrec/reward_mean'] = np.mean(self.batch_stats['reward'])
            logs['sasrec/reward_std'] = np.std(self.batch_stats['reward'])
            
            
            for key in self.batch_stats:
                self.batch_stats[key] = []
    
    def record_batch_reward(self, rank_before, rank_after, improvement, reward):
        
        self.batch_stats['rank_before'].append(rank_before)
        self.batch_stats['rank_after'].append(rank_after)
        self.batch_stats['improvement'].append(improvement)
        self.batch_stats['reward'].append(reward)


def create_monitoring_callbacks():
    
    return [
        GRPOMonitoringCallback(),
        
    ]
