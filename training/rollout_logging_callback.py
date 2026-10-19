

import os
import json
import torch
from transformers import TrainerCallback
from typing import List, Dict, Any
from datetime import datetime


class RolloutLoggingCallback(TrainerCallback):
    
    
    def __init__(
        self, 
        print_every_steps: int = 100,
        num_samples: int = 3,
        save_to_file: bool = True,
        output_dir: str = "./outputs",
        test_samples: List[str] = None
    ):
        super().__init__()
        self.print_every_steps = print_every_steps
        self.num_samples = num_samples
        self.save_to_file = save_to_file
        self.output_dir = output_dir
        self.test_samples = test_samples or []
        
        
        if self.save_to_file:
            self.rollout_dir = os.path.join(output_dir, "rollout_samples")
            os.makedirs(self.rollout_dir, exist_ok=True)
        
        self.last_print_step = 0
    
    def on_step_end(self, args, state, control, **kwargs):
        
        current_step = state.global_step
        
        
        if current_step - self.last_print_step >= self.print_every_steps:
            self.last_print_step = current_step
            self._print_rollout_samples(current_step, **kwargs)
    
    def _print_rollout_samples(self, step: int, model=None, train_dataset=None, processing_class=None, **kwargs):
        
        if model is None or processing_class is None:
            return
        
        print("\n" + "=" * 80)
        print(f"🎬 ROLLOUT SAMPLES at Step {step}")
        print("=" * 80)
        
        
        if self.test_samples:
            prompts = self.test_samples[:self.num_samples]
        
        elif train_dataset is not None:
            import random
            indices = random.sample(range(len(train_dataset)), min(self.num_samples, len(train_dataset)))
            prompts = []
            for idx in indices:
                sample = train_dataset[idx]
                
                if 'prompt' in sample:
                    prompts.append(sample['prompt'])
                elif 'messages' in sample:
                    
                    messages = sample['messages']
                    if isinstance(messages, list) and len(messages) > 0:
                        user_msg = [m for m in messages if m.get('role') == 'user']
                        if user_msg:
                            prompts.append(user_msg[0].get('content', ''))
        else:
            print("⚠️  No samples available for rollout")
            return
        
        
        samples_to_save = []
        
        model.eval()
        with torch.no_grad():
            for i, prompt in enumerate(prompts):
                print(f"\n{'─' * 80}")
                print(f"📝 Sample {i+1}/{len(prompts)}")
                print(f"{'─' * 80}")
                
                
                prompt_preview = prompt[:500] + "..." if len(prompt) > 500 else prompt
                print(f"\n💬 Prompt:\n{prompt_preview}")
                
                try:
                    
                    if hasattr(processing_class, 'apply_chat_template'):
                        messages = [{"role": "user", "content": prompt}]
                        full_prompt = processing_class.apply_chat_template(
                            messages, 
                            tokenize=False, 
                            add_generation_prompt=True
                        )
                    else:
                        full_prompt = prompt
                    
                    
                    inputs = processing_class(
                        full_prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512
                    ).to(model.device)
                    
                    
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=512,
                        temperature=0.7,
                        do_sample=True,
                        top_p=0.9,
                        pad_token_id=processing_class.pad_token_id,
                        eos_token_id=processing_class.eos_token_id,
                    )
                    
                    
                    generated_text = processing_class.decode(
                        outputs[0][inputs.input_ids.shape[1]:], 
                        skip_special_tokens=True
                    )
                    
                    print(f"\n🤖 Generated Response:\n{generated_text}")
                    
                    
                    quality_checks = self._analyze_response(generated_text)
                    print(f"\n📊 Quality Checks:")
                    for check, status in quality_checks.items():
                        icon = "✅" if status else "❌"
                        print(f"   {icon} {check}")
                    
                    
                    samples_to_save.append({
                        "step": step,
                        "sample_id": i + 1,
                        "prompt": prompt[:1000],  
                        "generated": generated_text,
                        "quality_checks": quality_checks
                    })
                    
                except Exception as e:
                    print(f"\n❌ Error generating sample: {e}")
        
        model.train()
        
        
        if self.save_to_file and samples_to_save:
            self._save_samples(step, samples_to_save)
        
        print("\n" + "=" * 80)
        print(f"✅ Rollout logging completed at step {step}")
        print("=" * 80 + "\n")
    
    def _analyze_response(self, response: str) -> Dict[str, bool]:
        
        checks = {}
        
        
        checks["Contains 'Clean Items'"] = "Clean Items" in response or "clean items" in response
        checks["Contains 'Noise Items'"] = "Noise Items" in response or "noise items" in response
        checks["Contains 'Explanation'"] = "Explanation" in response or "explanation" in response
        
        
        checks["Has bullet points or numbering"] = ("-" in response or "•" in response or 
                                                      any(f"{i}." in response for i in range(1, 10)))
        
        
        checks["Length reasonable (50-1000 chars)"] = 50 <= len(response) <= 1000
        
        
        lines = response.split('\n')
        unique_lines = set(line.strip() for line in lines if line.strip())
        checks["No major repetition"] = len(unique_lines) >= len(lines) * 0.7 if lines else True
        
        return checks
    
    def _save_samples(self, step: int, samples: List[Dict]):
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"rollout_step_{step}_{timestamp}.json"
        filepath = os.path.join(self.rollout_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Samples saved to: {filepath}")


def create_rollout_callback(
    print_every_steps: int = 100,
    num_samples: int = 3,
    save_to_file: bool = True,
    output_dir: str = "./outputs",
    test_samples: List[str] = None
) -> RolloutLoggingCallback:
    
    return RolloutLoggingCallback(
        print_every_steps=print_every_steps,
        num_samples=num_samples,
        save_to_file=save_to_file,
        output_dir=output_dir,
        test_samples=test_samples
    )
