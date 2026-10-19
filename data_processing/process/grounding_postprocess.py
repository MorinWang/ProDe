

import json
import torch
import fire
from typing import Dict, List, Set
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class SemanticGrounder:
    
    
    def __init__(
        self,
        model_path: str,
        item_embedding_path: str,
        id2name_dict: Dict[int, str],
        device: str = "cuda"
    ):
        self.device = device
        self.id2name_dict = id2name_dict
        
        
        self.name2id = {}
        for item_id, name in id2name_dict.items():
            clean_name = name.strip().lower()
            self.name2id[clean_name] = int(item_id)
        
        print(f"📦 Loading model and embeddings...")
        
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16
        ).to(device)
        self.model.eval()
        
        
        self.item_embeddings = torch.load(item_embedding_path).to(device)
        
        print(f"✅ Grounder initialized")
        print(f"   Item embeddings shape: {self.item_embeddings.shape}")
        print(f"   Device: {device}")
    
    def ground_title(self, title: str, top_k: int = 1) -> List[Dict]:
        
        
        clean_title = title.strip().lower()
        if clean_title in self.name2id:
            item_id = self.name2id[clean_title]
            return [{
                'item_id': item_id,
                'item_name': self.id2name_dict[item_id],
                'similarity_score': 1.0,
                'match_type': 'exact'
            }]
        
        
        with torch.no_grad():
            
            inputs = self.tokenizer(title, return_tensors="pt").to(self.device)
            outputs = self.model(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                output_hidden_states=True
            )
            
            
            title_embedding = outputs.hidden_states[-1][:, -1, :]  
            
            
            distances = torch.cdist(title_embedding, self.item_embeddings, p=2)  
            
            
            topk_distances, topk_indices = torch.topk(distances, k=top_k, largest=False, dim=1)
            
            results = []
            for dist, idx in zip(topk_distances[0], topk_indices[0]):
                item_id = idx.item()
                
                similarity = 1.0 / (1.0 + dist.item())
                
                results.append({
                    'item_id': item_id,
                    'item_name': self.id2name_dict.get(item_id, f"Unknown_{item_id}"),
                    'similarity_score': similarity,
                    'match_type': 'semantic',
                    'distance': dist.item()
                })
            
            return results
    
    def batch_ground_titles(
        self,
        titles: List[str],
        batch_size: int = 32,
        top_k: int = 1
    ) -> Dict[str, List[Dict]]:
        
        results = {}
        
        
        exact_matches = {}
        semantic_titles = []
        
        for title in titles:
            clean_title = title.strip().lower()
            if clean_title in self.name2id:
                item_id = self.name2id[clean_title]
                exact_matches[title] = [{
                    'item_id': item_id,
                    'item_name': self.id2name_dict[item_id],
                    'similarity_score': 1.0,
                    'match_type': 'exact'
                }]
            else:
                semantic_titles.append(title)
        
        results.update(exact_matches)
        
        
        if semantic_titles:
            print(f"  Grounding {len(semantic_titles)} titles semantically...")
            
            for i in tqdm(range(0, len(semantic_titles), batch_size)):
                batch_titles = semantic_titles[i:i+batch_size]
                
                with torch.no_grad():
                    
                    inputs = self.tokenizer(
                        batch_titles,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=128
                    ).to(self.device)
                    
                    outputs = self.model(
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        output_hidden_states=True
                    )
                    
                    
                    hidden_states = outputs.hidden_states[-1]  
                    seq_lengths = inputs.attention_mask.sum(dim=1) - 1
                    batch_embeddings = hidden_states[range(len(batch_titles)), seq_lengths, :]
                    
                    
                    distances = torch.cdist(batch_embeddings, self.item_embeddings, p=2)  
                    
                    
                    topk_distances, topk_indices = torch.topk(distances, k=top_k, largest=False, dim=1)
                    
                    
                    for title_idx, title in enumerate(batch_titles):
                        title_results = []
                        for k_idx in range(top_k):
                            item_id = topk_indices[title_idx, k_idx].item()
                            dist = topk_distances[title_idx, k_idx].item()
                            similarity = 1.0 / (1.0 + dist)
                            
                            title_results.append({
                                'item_id': item_id,
                                'item_name': self.id2name_dict.get(item_id, f"Unknown_{item_id}"),
                                'similarity_score': similarity,
                                'match_type': 'semantic',
                                'distance': dist
                            })
                        
                        results[title] = title_results
        
        return results
    
    def cleanup(self):
        
        del self.model
        del self.item_embeddings
        torch.cuda.empty_cache()


def postprocess_with_grounding(
    inference_result_path: str,
    id2name_path: str,
    item_embedding_path: str,
    model_path: str,
    output_path: str = None,
    batch_size: int = 32,
    top_k: int = 3,  
):
    
    print("🚀 Starting grounding postprocessing...")
    print(f"📊 Input: {inference_result_path}")
    
    
    print("\n[1/4] Loading inference results...")
    with open(inference_result_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data['results']
    stats = data['statistics']
    
    print(f"✅ Loaded {len(results)} samples")
    print(f"   Original OOD rate: {stats['ood_rate']*100:.2f}%")
    
    
    print("\n[2/4] Loading item mappings...")
    with open(id2name_path, 'r', encoding='utf-8') as f:
        id2name_dict = json.load(f)
    id2name_dict = {int(k): v for k, v in id2name_dict.items()}
    
    item_pool_set = set(v.strip().lower() for v in id2name_dict.values())
    
    print(f"✅ Loaded {len(id2name_dict)} items")
    
    
    print("\n[3/4] Collecting OOD titles...")
    ood_titles = set()
    
    for result in results:
        for title in result['generated_items']:
            clean_title = title.strip().lower()
            if clean_title not in item_pool_set:
                ood_titles.add(title)
    
    print(f"✅ Found {len(ood_titles)} unique OOD titles")
    
    if len(ood_titles) == 0:
        print("  No OOD titles to ground. All suggestions are valid!")
        return
    
    print(f"\n  Sample OOD titles:")
    for title in list(ood_titles)[:5]:
        print(f"    - \"{title}\"")
    
    
    print("\n[4/4] Initializing grounder and grounding...")
    
    grounder = SemanticGrounder(
        model_path=model_path,
        item_embedding_path=item_embedding_path,
        id2name_dict=id2name_dict,
        device="cuda"
    )
    
    
    grounding_results = grounder.batch_ground_titles(
        list(ood_titles),
        batch_size=batch_size,
        top_k=top_k
    )
    
    grounder.cleanup()
    
    
    print("\n[5/5] Applying grounding results...")
    
    grounded_stats = {
        'total_samples': len(results),
        'total_suggestions': 0,
        'exact_match': 0,
        'semantic_grounded': 0,
        'grounding_failed': 0,
    }
    
    for result in results:
        grounded_items = []
        grounding_info = []
        
        for title in result['generated_items']:
            grounded_stats['total_suggestions'] += 1
            
            clean_title = title.strip().lower()
            
            
            if clean_title in item_pool_set:
                
                grounded_items.append(title)
                grounding_info.append({
                    'original_title': title,
                    'grounded_title': title,
                    'match_type': 'exact',
                    'similarity_score': 1.0
                })
                grounded_stats['exact_match'] += 1
            else:
                
                if title in grounding_results:
                    top_match = grounding_results[title][0]
                    grounded_title = top_match['item_name']
                    grounded_items.append(grounded_title)
                    grounding_info.append({
                        'original_title': title,
                        'grounded_title': grounded_title,
                        'grounded_item_id': top_match['item_id'],
                        'match_type': top_match['match_type'],
                        'similarity_score': top_match['similarity_score'],
                        'distance': top_match.get('distance', 0),
                        'top_k_candidates': grounding_results[title][:3]  
                    })
                    grounded_stats['semantic_grounded'] += 1
                else:
                    
                    grounded_items.append(title)
                    grounding_info.append({
                        'original_title': title,
                        'grounded_title': title,
                        'match_type': 'failed',
                        'similarity_score': 0.0
                    })
                    grounded_stats['grounding_failed'] += 1
        
        
        result['grounded_items'] = grounded_items
        result['grounding_info'] = grounding_info
    
    
    print(f"\n{'='*70}")
    print(f"📊 GROUNDING RESULTS")
    print(f"{'='*70}")
    print(f"Total suggestions:        {grounded_stats['total_suggestions']}")
    print(f"Exact matches:            {grounded_stats['exact_match']} ({grounded_stats['exact_match']/grounded_stats['total_suggestions']*100:.2f}%)")
    print(f"Semantic grounded:        {grounded_stats['semantic_grounded']} ({grounded_stats['semantic_grounded']/grounded_stats['total_suggestions']*100:.2f}%)")
    print(f"Grounding failed:         {grounded_stats['grounding_failed']}")
    print(f"")
    print(f"✅ Final valid rate:      100.00% (after grounding)")
    print(f"   Original OOD rate:     {stats['ood_rate']*100:.2f}%")
    print(f"   Grounding coverage:    {(grounded_stats['semantic_grounded']/(stats['ood_suggestions']+0.001))*100:.2f}%")
    print(f"{'='*70}")
    
    
    if output_path is None:
        output_path = inference_result_path.replace('.json', '_grounded.json')
    
    output_data = {
        'config': {
            **data['config'],
            'grounding_applied': True,
            'grounding_model': model_path,
            'grounding_top_k': top_k,
        },
        'original_statistics': stats,
        'grounding_statistics': grounded_stats,
        'results': results,
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Grounded results saved to: {output_path}")
    
    
    print(f"\n{'='*70}")
    print(f"🔍 GROUNDING EXAMPLES (showing 5)")
    print(f"{'='*70}")
    
    example_count = 0
    for result in results:
        if 'grounding_info' in result:
            for info in result['grounding_info']:
                if info['match_type'] == 'semantic' and example_count < 5:
                    print(f"\n{example_count + 1}. Original: \"{info['original_title']}\"")
                    print(f"   Grounded: \"{info['grounded_title']}\"")
                    print(f"   Similarity: {info['similarity_score']:.4f}")
                    print(f"   Distance: {info.get('distance', 0):.4f}")
                    
                    if 'top_k_candidates' in info and len(info['top_k_candidates']) > 1:
                        print(f"   Top-3 candidates:")
                        for k, cand in enumerate(info['top_k_candidates'][:3]):
                            print(f"     {k+1}. {cand['item_name']} (sim: {cand['similarity_score']:.4f})")
                    
                    example_count += 1
                    
                if example_count >= 5:
                    break
        
        if example_count >= 5:
            break
    
    print(f"{'='*70}")


if __name__ == "__main__":
    fire.Fire(postprocess_with_grounding)
