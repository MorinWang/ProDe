

import json
import re
import torch
from typing import List, Dict, Set
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor
from tqdm import tqdm
import fire
from collections import defaultdict
import os
from vllm import LLM, SamplingParams

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


class ItemPoolTrie:
    
    
    def __init__(self):
        self.root = {}
        self.item_endings = {}  
    
    def insert(self, tokens: List[int], item_text: str):
        
        node = self.root
        for token in tokens:
            if token not in node:
                node[token] = {}
            node = node[token]
        
        node['<END>'] = item_text
        self.item_endings[tuple(tokens)] = item_text
    
    def get_valid_next_tokens(self, prefix: List[int]) -> Set[int]:
        
        node = self.root
        for token in prefix:
            if token in node:
                node = node[token]
            else:
                return set()  
        
        
        return set(k for k in node.keys() if k != '<END>')
    
    def is_complete_item(self, tokens: List[int]) -> bool:
        
        return tuple(tokens) in self.item_endings
    
    def get_item_text(self, tokens: List[int]) -> str:
        
        return self.item_endings.get(tuple(tokens))


class ItemPoolConstrainedLogitsProcessor(LogitsProcessor):
    
    
    def __init__(
        self, 
        tokenizer,
        item_pool: List[str],
        prompt_length: int,
        debug: bool = False
    ):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.debug = debug
        
        
        if self.debug:
            print(f"  Building Trie for {len(item_pool)} items...")
        
        self.trie = ItemPoolTrie()
        
        
        for item in item_pool:
            
            tokens = tokenizer.encode(item, add_special_tokens=False)
            if tokens:
                self.trie.insert(tokens, item)
            
            
            quoted_item = f'"{item}"'
            tokens_quoted = tokenizer.encode(quoted_item, add_special_tokens=False)
            if tokens_quoted:
                self.trie.insert(tokens_quoted, item)
        
        if self.debug:
            print(f"  ✅ Trie built")
        
        
        self.in_suggestion_phase = False
        self.in_quotes = False
        self.current_item_tokens = []
        
        
        self.quote_tokens = set(tokenizer.encode('"', add_special_tokens=False))
        self.comma_tokens = set(tokenizer.encode(',', add_special_tokens=False))
        self.newline_tokens = set(tokenizer.encode('\n', add_special_tokens=False))
        self.space_tokens = set(tokenizer.encode(' ', add_special_tokens=False))
        self.colon_tokens = set(tokenizer.encode(':', add_special_tokens=False))
        
        
        self.constraint_applied_count = 0
        self.total_steps = 0
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        
        self.total_steps += 1
        batch_size = input_ids.shape[0]
        
        for batch_idx in range(batch_size):
            generated_ids = input_ids[batch_idx, self.prompt_length:]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            
            if not self.in_suggestion_phase:
                if re.search(r"Suggested.*?(?:Alternatives|Replacements)", generated_text, re.I):
                    self.in_suggestion_phase = True
                    if self.debug:
                        print(f"\n  🔒 Constraint activated")
            
            
            if self.in_suggestion_phase:
                scores[batch_idx] = self._apply_constraint(scores[batch_idx], generated_ids)
        
        return scores
    
    def _apply_constraint(self, scores: torch.Tensor, generated_ids: torch.Tensor) -> torch.Tensor:
        
        
        recent_text = self.tokenizer.decode(generated_ids[-30:], skip_special_tokens=True)
        quote_count = recent_text.count('"')
        self.in_quotes = (quote_count % 2 == 1)
        
        if not self.in_quotes:
            
            mask = torch.full_like(scores, float('-inf'))
            allowed = (
                self.quote_tokens | self.comma_tokens | 
                self.newline_tokens | self.space_tokens | 
                self.colon_tokens | {self.tokenizer.eos_token_id}
            )
            mask[list(allowed)] = 0
            self.current_item_tokens = []
            return scores + mask
        
        
        self.constraint_applied_count += 1
        
        
        quote_positions = [i for i, tok in enumerate(generated_ids) if tok.item() in self.quote_tokens]
        
        if quote_positions:
            last_quote_pos = quote_positions[-1]
            tokens_in_quotes = generated_ids[last_quote_pos + 1:].tolist()
        else:
            tokens_in_quotes = []
        
        self.current_item_tokens = tokens_in_quotes
        
        
        valid_next_tokens = self.trie.get_valid_next_tokens(tokens_in_quotes)
        
        
        if self.trie.is_complete_item(tokens_in_quotes):
            valid_next_tokens.update(self.quote_tokens)
        
        
        if valid_next_tokens:
            mask = torch.full_like(scores, float('-inf'))
            mask[list(valid_next_tokens)] = 0
            return scores + mask
        else:
            
            mask = torch.full_like(scores, float('-inf'))
            mask[list(self.quote_tokens | {self.tokenizer.eos_token_id})] = 0
            return scores + mask


def extract_suggested_items(text: str) -> List[str]:
    
    
    pattern = r'Suggested.*?:\s*"(.*?)"'
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    return [m.strip() for m in matches if m.strip()]


def analyze_results(results: List[Dict], item_pool_set: Set[str]) -> Dict:
    
    stats = {
        'total_samples': len(results),
        'total_suggestions': 0,
        'valid_suggestions': 0,
        'ood_suggestions': 0,
        'ood_rate': 0.0,
        'empty_suggestions': 0,
    }
    
    for result in results:
        suggestions = result['generated_items']
        
        if not suggestions:
            stats['empty_suggestions'] += 1
            continue
        
        stats['total_suggestions'] += len(suggestions)
        
        for item in suggestions:
            item_clean = item.strip().lower()
            if item_clean in item_pool_set:
                stats['valid_suggestions'] += 1
            else:
                stats['ood_suggestions'] += 1
    
    if stats['total_suggestions'] > 0:
        stats['ood_rate'] = stats['ood_suggestions'] / stats['total_suggestions']
        stats['valid_rate'] = stats['valid_suggestions'] / stats['total_suggestions']
    
    return stats


def inference_with_constraints(
    model_path: str = "models/llama3_8B_Instruct_sft_denoised_merged",
    test_data_path: str = "data/Movie_noise_user0.1/TDMD/test_with_description.json",
    id2name_path: str = "data/Movie_noise_user0.1/id2name4Rec.json",
    output_path: str = "results/constrained_inference.json",
    use_constraints: bool = True,
    max_samples: int = 100,  
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    
    print(f"🚀 Starting constrained inference...")
    print(f"📦 Model: {model_path}")
    print(f"📊 Test data: {test_data_path}")
    print(f"🔒 Constraints: {'ENABLED' if use_constraints else 'DISABLED'}")
    
    
    print("\n[1/5] Loading item pool...")
    with open(id2name_path, 'r', encoding='utf-8') as f:
        id2name_dict = json.load(f)
    
    item_pool = list(id2name_dict.values())
    item_pool_set = set(item.strip().lower() for item in item_pool)
    
    print(f"✅ Item pool loaded: {len(item_pool)} items")
    print(f"   Sample: {item_pool[:3]}")
    
    
    print("\n[2/5] Loading model and tokenizer...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    
    print(f"✅ Model loaded")
    
    
    print(f"\n[3/5] Loading test data...")
    
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    
    if max_samples > 0:
        test_data = test_data[:max_samples]
    
    print(f"✅ Test data loaded: {len(test_data)} samples")
    
    
    print(f"\n[4/5] Running inference...")
    
    results = []
    
    for idx, sample in enumerate(tqdm(test_data, desc="Inference")):
        
        instruction = sample['instruction']
        user_input = sample['input']
        
        prompt = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "You must follow the following output format:\n\n"
            "Noise Items: \"item1\", \"item2\"\n"
            "Suggested Replacements: \"item1\", \"item2\"\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{user_input}\n\n"
            "### Response:\n"
        )
        
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_length = inputs.input_ids.shape[1]
        
        
        gen_kwargs = {
            'max_new_tokens': max_new_tokens,
            'temperature': temperature,
            'top_p': top_p,
            'do_sample': True,
            'pad_token_id': tokenizer.pad_token_id,
            'eos_token_id': tokenizer.eos_token_id,
        }
        
        
        if use_constraints:
            from transformers import LogitsProcessorList
            
            constraint_processor = ItemPoolConstrainedLogitsProcessor(
                tokenizer=tokenizer,
                item_pool=item_pool,
                prompt_length=prompt_length,
                debug=(idx == 0)  
            )
            
            gen_kwargs['logits_processor'] = LogitsProcessorList([constraint_processor])
        
        
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
        
        
        generated_text = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
        
        
        suggested_items = extract_suggested_items(generated_text)
        
        
        result = {
            'index': idx,
            'input': user_input,
            'ground_truth': sample.get('output', ''),
            'generated_text': generated_text,
            'generated_items': suggested_items,
        }
        results.append(result)
        
        
        if idx < 3:
            print(f"\n{'='*70}")
            print(f"Sample {idx + 1}:")
            print(f"Generated: {generated_text[:200]}...")
            print(f"Suggested items: {suggested_items}")
            print(f"{'='*70}")
    
    
    print(f"\n[5/5] Analyzing results...")
    
    stats = analyze_results(results, item_pool_set)
    
    print(f"\n{'='*70}")
    print(f"📊 RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"Total samples:       {stats['total_samples']}")
    print(f"Total suggestions:   {stats['total_suggestions']}")
    print(f"Valid suggestions:   {stats['valid_suggestions']}")
    print(f"OOD suggestions:     {stats['ood_suggestions']}")
    print(f"Empty suggestions:   {stats['empty_suggestions']}")
    print(f"")
    print(f"✅ Valid rate:       {stats.get('valid_rate', 0)*100:.2f}%")
    print(f"❌ OOD rate:         {stats['ood_rate']*100:.2f}%")
    print(f"{'='*70}")
    
    
    output_data = {
        'config': {
            'model_path': model_path,
            'use_constraints': use_constraints,
            'max_samples': max_samples,
            'temperature': temperature,
            'top_p': top_p,
        },
        'statistics': stats,
        'results': results,
    }
    
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: {output_path}")
    
    
    if stats['ood_suggestions'] > 0:
        print(f"\n⚠️  OOD Examples:")
        ood_count = 0
        for result in results:
            for item in result['generated_items']:
                if item.strip().lower() not in item_pool_set:
                    print(f"  - \"{item}\"")
                    ood_count += 1
                    if ood_count >= 10:  
                        break
            if ood_count >= 10:
                break
    
    return results, stats


def run_comparison():
    
    print("\n" + "="*70)
    print("🔬 Running comparison experiment")
    print("="*70)
    
    
    print("\n\n" + "="*70)
    print("📍 Experiment 1: WITHOUT Constraints")
    print("="*70)
    results_baseline, stats_baseline = inference_with_constraints(
        use_constraints=False,
        max_samples=50,
        output_path="results/baseline_no_constraints.json"
    )
    
    
    print("\n\n" + "="*70)
    print("📍 Experiment 2: WITH Constraints")
    print("="*70)
    results_constrained, stats_constrained = inference_with_constraints(
        use_constraints=True,
        max_samples=50,
        output_path="results/constrained.json"
    )
    
    
    print("\n\n" + "="*70)
    print("📊 COMPARISON")
    print("="*70)
    print(f"{'Metric':<30} {'Baseline':<15} {'Constrained':<15} {'Improvement'}")
    print("-"*70)
    
    baseline_ood = stats_baseline['ood_rate'] * 100
    constrained_ood = stats_constrained['ood_rate'] * 100
    improvement = baseline_ood - constrained_ood
    
    print(f"{'OOD Rate':<30} {baseline_ood:>6.2f}% {constrained_ood:>14.2f}% {improvement:>14.2f}%")
    
    baseline_valid = stats_baseline.get('valid_rate', 0) * 100
    constrained_valid = stats_constrained.get('valid_rate', 0) * 100
    improvement_valid = constrained_valid - baseline_valid
    
    print(f"{'Valid Rate':<30} {baseline_valid:>6.2f}% {constrained_valid:>14.2f}% {improvement_valid:>14.2f}%")
    print("="*70)


if __name__ == "__main__":
    fire.Fire({
        'inference': inference_with_constraints,
        'compare': run_comparison,
    })
