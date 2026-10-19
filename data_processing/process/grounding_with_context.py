

import json
import torch
import fire
from typing import Dict, List
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class ContextBasedGrounder:
    
    
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
    
    def encode_context(self, context_text: str) -> torch.Tensor:
        
        with torch.no_grad():
            inputs = self.tokenizer(
                context_text, 
                return_tensors="pt",
                truncation=True,
                max_length=2048  
            ).to(self.device)
            
            outputs = self.model(
                inputs.input_ids, 
                output_hidden_states=True
            )
            
            
            emb = outputs.hidden_states[-1][:, -1, :]
            
            return emb
    
    def ground_ood_items_with_context(
        self,
        context_text: str,
        ood_titles: List[str],
        top_k: int = 5
    ) -> Dict[str, List[Dict]]:
        
        if not ood_titles:
            return {}
        
        
        context_emb = self.encode_context(context_text)
        
        
        context_emb_norm = context_emb / torch.norm(context_emb, p=2, dim=1, keepdim=True)
        item_embs_norm = self.item_embeddings / torch.norm(self.item_embeddings, p=2, dim=1, keepdim=True)
        
        
        cosine_similarities = torch.matmul(context_emb_norm, item_embs_norm.T)  
        
        
        cosine_similarities[0, 0] = -1.0
        
        
        topk_similarities, topk_indices = torch.topk(
            cosine_similarities, 
            k=min(top_k * 2, cosine_similarities.shape[1]), 
            largest=True,  
            dim=1
        )
        
        
        candidates = []
        for k_idx in range(topk_similarities.shape[1]):
            item_id = topk_indices[0, k_idx].item()
            similarity = topk_similarities[0, k_idx].item()
            
            
            if item_id == 0 or item_id not in self.id2name_dict:
                continue
            
            
            if similarity < 0:
                continue
            
            candidates.append({
                'item_id': item_id,
                'item_name': self.id2name_dict.get(item_id, f"Unknown_{item_id}"),
                'item_desc': self.id2desc_dict.get(item_id, ""),
                'similarity_score': similarity,
                'cosine_similarity': similarity,
                'match_type': 'context_based_cosine'
            })
            
            
            if len(candidates) >= top_k:
                break
        
        
        results = {}
        for ood_title in ood_titles:
            
            clean_title = ood_title.strip().lower()
            if clean_title in self.name2id:
                item_id = self.name2id[clean_title]
                results[ood_title] = [{
                    'item_id': item_id,
                    'item_name': self.id2name_dict[item_id],
                    'item_desc': self.id2desc_dict.get(item_id, ""),
                    'similarity_score': 1.0,
                    'cosine_similarity': 1.0,
                    'match_type': 'exact'
                }]
            else:
                
                results[ood_title] = candidates
        
        return results


def postprocess_with_context_grounding(
    inference_result_path: str,
    id2name_path: str,
    id2desc_path: str,
    item_embedding_path: str,
    model_path: str,
    output_path: str,
    top_k: int = 5,
    device: str = "cuda"
):
    
    print("🚀 Starting context-based grounding postprocessing...")
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
    
    
    print("[3/5] Converting data format and identifying OOD items...")
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
    
    
    print("[4/5] Initializing grounder and grounding with context...")
    grounder = ContextBasedGrounder(
        model_path=model_path,
        item_embedding_path=item_embedding_path,
        id2name_dict=id2name_dict,
        id2desc_dict=id2desc_dict,
        device=device
    )
    print()
    
    
    print("Processing samples with context-based grounding...")
    grounded_samples = []
    
    for sample in tqdm(samples, desc="Grounding samples"):
        context_text = sample['input']
        suggested_titles = sample.get('suggested_titles', [])
        is_ood = sample.get('is_ood', [])
        
        
        ood_titles_in_sample = [title for title, ood in zip(suggested_titles, is_ood) if ood]
        
        
        if ood_titles_in_sample:
            grounding_results = grounder.ground_ood_items_with_context(
                context_text=context_text,
                ood_titles=ood_titles_in_sample,
                top_k=top_k
            )
        else:
            grounding_results = {}
        
        
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
                    'cosine_similarity': best_match.get('cosine_similarity', best_match['similarity_score']),
                    'match_type': best_match['match_type'],
                    'top_k_candidates': grounding_results[title][:top_k]
                })
            else:
                
                grounded_titles.append(title)
                item_id = name2id.get(title.strip().lower(), -1)
                grounded_ids.append(item_id)
                grounding_info.append(None)
        
        new_sample = sample.copy()
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
    print(f"Total OOD occurrences: {ood_count}")
    print(f"OOD rate: {original_ood_rate:.2%}")
    print(f"Grounding method: Context-based (using full user interaction sequence)")
    print()
    
    
    print("="*70)
    print(f"🔍 GROUNDING EXAMPLES (showing 5)")
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
                print(f"   Description: {info['grounded_desc'][:100]}...")
                print(f"   Cosine Similarity: {info['cosine_similarity']:.4f}")
                print(f"   Match type: {info['match_type']}")
                print(f"   Context: {sample['input'][:150]}...")
                print(f"   Top-{top_k} candidates:")
                for i, cand in enumerate(info['top_k_candidates'], 1):
                    cos_sim = cand.get('cosine_similarity', cand['similarity_score'])
                    print(f"     {i}. {cand['item_name']} (cosine_sim: {cos_sim:.4f})")
    
    print("="*70)
    print("✅ Context-based grounding completed!")


if __name__ == "__main__":
    fire.Fire(postprocess_with_context_grounding)
