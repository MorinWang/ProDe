

import json
import torch
import fire
from typing import Dict, List, Set
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class SemanticGrounderWithDesc:
    
    
    def __init__(
        self,
        model_path: str,
        item_embedding_path: str,
        id2name_dict: Dict[int, str],
        id2desc_dict: Dict[int, str],
        device: str = "cuda"
    ):
        self.device = device
        self.id2name_dict = id2name_dict
        self.id2desc_dict = id2desc_dict
        
        
        self.name2id = {}
        for item_id, name in id2name_dict.items():
            clean_name = name.strip().lower()
            self.name2id[clean_name] = int(item_id)
        
        print(f"📦 Loading model and embeddings...")
        
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        
        self.tokenizer.padding_side = "left"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16
        ).to(device)
        self.model.eval()
        
        
        self.item_embeddings = torch.load(item_embedding_path).to(device)
        
        print(f"✅ Grounder initialized")
        print(f"   Item embeddings shape: {self.item_embeddings.shape}")
        print(f"   Padding side: {self.tokenizer.padding_side}")
        print(f"   Device: {device}")
    
    def ground_title(self, title: str, top_k: int = 1) -> List[Dict]:
        
        
        clean_title = title.strip().lower()
        if clean_title in self.name2id:
            item_id = self.name2id[clean_title]
            return [{
                'item_id': item_id,
                'item_name': self.id2name_dict[item_id],
                'item_desc': self.id2desc_dict.get(item_id, ""),
                'similarity_score': 1.0,
                'match_type': 'exact',
                'distance': 0.0
            }]
        
        
        with torch.no_grad():
            inputs = self.tokenizer(title, return_tensors="pt").to(self.device)
            outputs = self.model(inputs.input_ids, output_hidden_states=True)
            
            
            emb = outputs.hidden_states[-1][:, -1, :]
            
            
            dist = torch.cdist(emb, self.item_embeddings, p=2)
            
            
            topk_distances, topk_indices = torch.topk(dist, k=top_k, largest=False, dim=1)
            
            results = []
            for distance, idx in zip(topk_distances[0], topk_indices[0]):
                item_id = idx.item()
                dist_val = distance.item()
                
                
                similarity = torch.exp(-distance / 10).item()  
                
                results.append({
                    'item_id': item_id,
                    'item_name': self.id2name_dict.get(item_id, f"Unknown_{item_id}"),
                    'item_desc': self.id2desc_dict.get(item_id, ""),
                    'similarity_score': similarity,
                    'match_type': 'semantic',
                    'distance': dist_val
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
                    'item_desc': self.id2desc_dict.get(item_id, ""),
                    'similarity_score': 1.0,
                    'match_type': 'exact',
                    'distance': 0.0
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
                    batch_embeddings = hidden_states[:, -1, :]  
                    
                    
                    distances = torch.cdist(batch_embeddings, self.item_embeddings, p=2)  
                    
                    
                    topk_distances, topk_indices = torch.topk(distances, k=top_k, largest=False, dim=1)
                    
                    
                    for title_idx, title in enumerate(batch_titles):
                        title_results = []
                        for k_idx in range(top_k):
                            item_id = topk_indices[title_idx, k_idx].item()
                            dist = topk_distances[title_idx, k_idx].item()
                            
                            
                            similarity = torch.exp(torch.tensor(-dist / 10)).item()
                            
                            title_results.append({
                                'item_id': item_id,
                                'item_name': self.id2name_dict.get(item_id, f"Unknown_{item_id}"),
                                'item_desc': self.id2desc_dict.get(item_id, ""),
                                'similarity_score': similarity,
                                'match_type': 'semantic',
                                'distance': dist
                            })
                        
                        results[title] = title_results
        
        return results


def postprocess_with_grounding(
    inference_result_path: str,
    id2name_path: str,
    id2desc_path: str,
    item_embedding_path: str,
    model_path: str,
    output_path: str,
    batch_size: int = 32,
    top_k: int = 3,
    device: str = "cuda"
):
    
    print("🚀 Starting grounding postprocessing with description enhancement...")
    print(f"📊 Input: {inference_result_path}")
    print()
    
    
    print("[1/5] Loading inference results...")
    with open(inference_result_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    
    if isinstance(data, dict) and 'results' in data:
        raw_results = data['results']
        print(f"✅ Loaded {len(raw_results)} samples from 'results' field")
    else:
        raw_results = data
        print(f"✅ Loaded {len(raw_results)} samples")
    
    
    print()
    
    
    print("[2/5] Loading item mappings...")
    with open(id2name_path, 'r', encoding='utf-8') as f:
        id2name_raw = json.load(f)
    id2name_dict = {int(k): v for k, v in id2name_raw.items()}
    
    with open(id2desc_path, 'r', encoding='utf-8') as f:
        id2desc_raw = json.load(f)
    id2desc_dict = {int(k): v for k, v in id2desc_raw.items()}
    
    name2id = {v.strip().lower(): int(k) for k, v in id2name_raw.items()}
    
    print(f"✅ Loaded {len(id2name_dict)} items")
    print(f"   With descriptions: {len(id2desc_dict)}")
    print()
    
    
    print("[2.5/5] Converting data format and identifying OOD items...")
    samples = []
    total_suggestions = 0
    ood_count = 0
    
    for raw_result in raw_results:
        
        generated_items = raw_result.get('generated_items', [])
        if not generated_items:
            
            generated_items = raw_result.get('suggested_items', [])
        if not generated_items:
            generated_items = raw_result.get('suggested_titles', [])
        
        
        is_ood = []
        for item in generated_items:
            item_clean = item.strip().lower()
            is_out = item_clean not in name2id
            is_ood.append(is_out)
            if is_out:
                ood_count += 1
        
        total_suggestions += len(generated_items)
        
        
        sample = {
            'index': raw_result.get('index', len(samples)),
            'input': raw_result.get('input', ''),
            'ground_truth': raw_result.get('ground_truth', ''),
            'generated_text': raw_result.get('generated_text', ''),
            'suggested_titles': generated_items,
            'is_ood': is_ood
        }
        samples.append(sample)
    
    original_ood_rate = ood_count / total_suggestions if total_suggestions > 0 else 0
    print(f"✅ Converted {len(samples)} samples")
    print(f"   Total suggestions: {total_suggestions}")
    print(f"   OOD items: {ood_count}")
    print(f"   OOD rate: {original_ood_rate:.2%}")
    print()
    
    
    print("[3/5] Collecting OOD titles...")
    ood_titles = set()
    
    for sample in samples:
        for title, is_ood in zip(sample.get('suggested_titles', []), sample.get('is_ood', [])):
            if is_ood:
                ood_titles.add(title)
    
    print(f"✅ Found {len(ood_titles)} unique OOD titles")
    if ood_titles:
        print(f"\n  Sample OOD titles:")
        for title in list(ood_titles)[:5]:
            print(f"    - \"{title}\"")
    print()
    
    
    print("[4/5] Initializing grounder and grounding...")
    grounder = SemanticGrounderWithDesc(
        model_path=model_path,
        item_embedding_path=item_embedding_path,
        id2name_dict=id2name_dict,
        id2desc_dict=id2desc_dict,
        device=device
    )
    
    grounding_results = grounder.batch_ground_titles(
        list(ood_titles),
        batch_size=batch_size,
        top_k=top_k
    )
    print()
    
    
    print("[5/5] Applying grounding to samples...")
    grounded_samples = []
    
    for sample in tqdm(samples, desc="Processing samples"):
        new_sample = sample.copy()
        suggested_titles = sample.get('suggested_titles', [])
        is_ood = sample.get('is_ood', [])
        
        grounded_titles = []
        grounded_ids = []
        grounding_info = []
        
        for title, ood in zip(suggested_titles, is_ood):
            if ood and title in grounding_results:
                
                best_match = grounding_results[title][0]
                grounded_titles.append(best_match['item_name'])
                grounded_ids.append(best_match['item_id'])
                grounding_info.append({
                    'original_title': title,
                    'grounded_title': best_match['item_name'],
                    'grounded_id': best_match['item_id'],
                    'grounded_desc': best_match['item_desc'],
                    'similarity': best_match['similarity_score'],
                    'distance': best_match['distance'],
                    'match_type': best_match['match_type'],
                    'top_k_candidates': grounding_results[title][:top_k]
                })
            else:
                
                grounded_titles.append(title)
                item_id = name2id.get(title.strip().lower(), -1)
                grounded_ids.append(item_id)
                grounding_info.append(None)
        
        new_sample['grounded_titles'] = grounded_titles
        new_sample['grounded_ids'] = grounded_ids
        new_sample['grounding_info'] = grounding_info
        
        grounded_samples.append(new_sample)
    
    
    print(f"\n💾 Saving grounded results to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(grounded_samples, f, indent=2, ensure_ascii=False)
    
    
    print("\n" + "="*70)
    print("📊 GROUNDING STATISTICS")
    print("="*70)
    print(f"Total samples: {len(grounded_samples)}")
    print(f"Total suggestions: {total_suggestions}")
    print(f"Unique OOD titles: {len(ood_titles)}")
    print(f"Total OOD occurrences: {ood_count}")
    print(f"OOD rate: {original_ood_rate:.2%}")
    print()
    
    
    print("="*70)
    print("🔍 GROUNDING EXAMPLES (showing 5)")
    print("="*70)
    
    examples_shown = 0
    for sample in grounded_samples:
        if examples_shown >= 5:
            break
        for info in sample.get('grounding_info', []):
            if info and examples_shown < 5:
                examples_shown += 1
                print(f"\n{examples_shown}. Original: \"{info['original_title']}\"")
                print(f"   Grounded: \"{info['grounded_title']}\"")
                print(f"   Description: {info['grounded_desc']}")
                print(f"   Similarity: {info['similarity']:.4f}")
                print(f"   Distance: {info['distance']:.4f}")
                print(f"   Top-{top_k} candidates:")
                for i, cand in enumerate(info['top_k_candidates'], 1):
                    print(f"     {i}. {cand['item_name']} (sim: {cand['similarity']:.4f})")
    
    print("="*70)
    print("✅ Grounding postprocessing completed!")


if __name__ == "__main__":
    fire.Fire(postprocess_with_grounding)
