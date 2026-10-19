

import json
import re
import torch
import os
from typing import List, Dict, Set, Optional
from tqdm import tqdm
import fire


from vllm import LLM, SamplingParams
from vllm.sampling_params import LogitsProcessor as VLLMLogitsProcessor

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


class VLLMItemPoolConstrainedLogitsProcessor:
    
    
    def __init__(
        self, 
        tokenizer,
        item_pool: List[str],
        prompt_text: str,  
        debug: bool = False
    ):
        self.tokenizer = tokenizer
        self.prompt_text = prompt_text
        self.debug = debug
        
        
        if self.debug:
            print(f"  Building Trie for {len(item_pool)} items...")
        
        self.trie = ItemPoolTrie()
        
        
        for item in item_pool:
            
            tokens = tokenizer.encode(item, add_special_tokens=False)
            if tokens:
                self.trie.insert(tokens, item)
            
            
            for quote_variant in [f'"{item}"', f' "{item}"']:
                tokens_quoted = tokenizer.encode(quote_variant, add_special_tokens=False)
                if tokens_quoted:
                    self.trie.insert(tokens_quoted, item)
            
            
            tokens_with_space = tokenizer.encode(f" {item}", add_special_tokens=False)
            if tokens_with_space:
                self.trie.insert(tokens_with_space, item)
        
        if self.debug:
            print(f"  ✅ Trie built")
        
        
        self.in_suggestion_phase = False
        self.in_quotes = False
        
        
        self.quote_tokens = set(tokenizer.encode('"', add_special_tokens=False))
        self.comma_tokens = set(tokenizer.encode(',', add_special_tokens=False))
        self.newline_tokens = set(tokenizer.encode('\n', add_special_tokens=False))
        self.space_tokens = set(tokenizer.encode(' ', add_special_tokens=False))
        self.colon_tokens = set(tokenizer.encode(':', add_special_tokens=False))
        self.eos_token_id = tokenizer.eos_token_id
    
    def __call__(
        self, 
        token_ids: List[int],  
        logits: torch.Tensor   
    ) -> torch.Tensor:
        
        
        generated_text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        
        
        if not self.in_suggestion_phase:
            if re.search(r"Suggested.*?(?:Alternatives|Replacements)", generated_text, re.I):
                self.in_suggestion_phase = True
                if self.debug:
                    print(f"\n  🔒 Constraint activated")
        
        
        if not self.in_suggestion_phase:
            return logits
        
        
        return self._apply_constraint(logits, token_ids, generated_text)
    
    def _apply_constraint(
        self, 
        logits: torch.Tensor, 
        token_ids: List[int],
        generated_text: str
    ) -> torch.Tensor:
        
        
        match = re.search(r"Suggested.*?:\s*(.*)", generated_text, re.I | re.DOTALL)
        if match:
            after_suggested = match.group(1)
            quote_count = after_suggested.count('"')
            self.in_quotes = (quote_count % 2 == 1)
        else:
            self.in_quotes = False
        
        if not self.in_quotes:
            
            mask = torch.full_like(logits, float('-inf'))
            allowed = (
                self.quote_tokens | self.comma_tokens | 
                self.newline_tokens | self.space_tokens | 
                self.colon_tokens | {self.eos_token_id}
            )
            mask[list(allowed)] = 0
            return logits + mask
        
        
        quote_positions = []
        for i, tid in enumerate(token_ids):
            if tid in self.quote_tokens:
                quote_positions.append(i)
        
        if quote_positions:
            last_quote_pos = quote_positions[-1]
            tokens_in_quotes = token_ids[last_quote_pos + 1:]
        else:
            tokens_in_quotes = []
        
        
        valid_next_tokens = self.trie.get_valid_next_tokens(tokens_in_quotes)
        
        
        if self.trie.is_complete_item(tokens_in_quotes):
            valid_next_tokens.update(self.quote_tokens)
        
        
        if valid_next_tokens:
            mask = torch.full_like(logits, float('-inf'))
            mask[list(valid_next_tokens)] = 0
            return logits + mask
        else:
            
            if len(tokens_in_quotes) < 3:
                
                mask = torch.full_like(logits, float('-inf'))
                allowed = self.space_tokens | self.quote_tokens | {self.eos_token_id}
                mask[list(allowed)] = 0
                return logits + mask
            else:
                
                mask = torch.full_like(logits, float('-inf'))
                mask[list(self.quote_tokens | {self.eos_token_id})] = 0
                return logits + mask


def inference_with_vllm(
    model_path: str = "models/llama3_8B_Instruct_sft_denoised_merged",
    test_data_path: str = "data/Movie_noise_user0.1/TDMD/test_with_description.json",
    id2name_path: str = "data/Movie_noise_user0.1/id2name4Rec.json",
    output_path: str = "results/vllm_constrained_inference.json",
    use_constraints: bool = True,
    max_samples: int = 100,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    tensor_parallel_size: int = 4,  
    gpu_memory_utilization: float = 0.9,
):
    
    print(f"🚀 Starting vLLM constrained inference...")
    print(f"📦 Model: {model_path}")
    print(f"🎮 Tensor Parallel Size: {tensor_parallel_size}")
    print(f"🔒 Constraints: {'ENABLED' if use_constraints else 'DISABLED'}")
    
    
    print("\n[1/4] Loading item pool...")
    with open(id2name_path, 'r', encoding='utf-8') as f:
        id2name_dict = json.load(f)
    
    item_pool = list(id2name_dict.values())
    item_pool_set = set(item.strip().lower() for item in item_pool)
    
    print(f"✅ Item pool loaded: {len(item_pool)} items")
    
    
    print("\n[2/4] Initializing vLLM...")
    
    
    import os
    os.environ['VLLM_USE_V1'] = '0'
    
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=2048,
        enforce_eager=True,  
    )
    
    tokenizer = llm.get_tokenizer()
    
    print(f"✅ vLLM initialized")
    
    
    print(f"\n[3/4] Loading test data...")
    
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    if max_samples > 0:
        test_data = test_data[:max_samples]
    
    print(f"✅ Test data loaded: {len(test_data)} samples")
    
    
    print(f"\n[4/4] Running inference with vLLM...")
    
    
    prompts = []
    for sample in test_data:
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
        prompts.append(prompt)
    
    
    if use_constraints:
        
        def create_logits_processor(prompt_text: str):
            return VLLMItemPoolConstrainedLogitsProcessor(
                tokenizer=tokenizer,
                item_pool=item_pool,
                prompt_text=prompt_text,
                debug=False
            )
        
        
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            
            logits_processors=[create_logits_processor(p) for p in prompts],
        )
    else:
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )
    
    
    print(f"  Generating {len(prompts)} samples in batch...")
    outputs = llm.generate(prompts, sampling_params)
    
    
    print(f"\n[5/5] Processing results...")
    
    results = []
    for idx, (sample, output) in enumerate(zip(test_data, outputs)):
        generated_text = output.outputs[0].text
        
        
        pattern = r'Suggested.*?:\s*"(.*?)"'
        matches = re.findall(pattern, generated_text, re.DOTALL | re.IGNORECASE)
        suggested_items = [m.strip() for m in matches if m.strip()]
        
        result = {
            'index': idx,
            'input': sample['input'],
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
    
    
    stats = {
        'total_samples': len(results),
        'total_suggestions': 0,
        'valid_suggestions': 0,
        'ood_suggestions': 0,
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
    
    print(f"\n{'='*70}")
    print(f"📊 RESULTS SUMMARY (vLLM)")
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
            'vllm': True,
            'tensor_parallel_size': tensor_parallel_size,
        },
        'statistics': stats,
        'results': results,
    }
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else "results", exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: {output_path}")
    
    return results, stats


def benchmark_speed():
    
    import time
    
    print("\n" + "="*70)
    print("⚡ Speed Benchmark: vLLM vs Transformers")
    print("="*70)
    
    
    test_samples = 50
    
    
    print(f"\n📍 Testing vLLM ({test_samples} samples)...")
    start_time = time.time()
    _, stats_vllm = inference_with_vllm(
        max_samples=test_samples,
        use_constraints=True,
        output_path="results/vllm_benchmark.json"
    )
    vllm_time = time.time() - start_time
    
    print(f"\n⏱️  vLLM Time: {vllm_time:.2f}s ({vllm_time/test_samples:.2f}s per sample)")
    
    
if __name__ == "__main__":
    fire.Fire({
        'inference': inference_with_vllm,
        'benchmark': benchmark_speed,
    })
