

import os
import sys
import json
import math
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import argparse


original_argv = sys.argv.copy()


script_dir = os.path.dirname(os.path.abspath(__file__))
sasrec_code_dir = os.path.join(script_dir, "SASRec_code")
sys.path.insert(0, sasrec_code_dir)


sys.argv = [sys.argv[0]]

from model import SASRec


sys.argv = original_argv


class TestDataset(Dataset):
    
    def __init__(self, test_df, maxlen=10):
        
        self.test_df = test_df
        self.maxlen = maxlen
        
    def __len__(self):
        return len(self.test_df)
    
    def __getitem__(self, idx):
        row = self.test_df.iloc[idx]
        
        
        item_ids = eval(row['item_ids'])
        
        
        if len(item_ids) > self.maxlen + 1:
            item_ids = item_ids[-(self.maxlen + 1):]
        else:
            
            item_ids = [0] * (self.maxlen + 1 - len(item_ids)) + item_ids
        
        return torch.tensor(item_ids, dtype=torch.long)


def evaluate_sasrec(
    model: SASRec, 
    dataloader: DataLoader, 
    sample_total: int, 
    topk_list=[1, 5, 10, 20, 50],
    device='cuda'
):
    
    model.eval()
    
    NDCG_total = torch.zeros(size=(len(topk_list),), device=device)
    HR_total = torch.zeros(size=(len(topk_list),), device=device)
    
    with torch.no_grad():
        for seq_items_id in tqdm(dataloader, desc="Evaluating"):
            seq_items_id = seq_items_id.to(device)
            
            
            logits = -model.predict(seq_items_id[:, :-1])  
            target_item_ids = seq_items_id[:, -1]
            
            
            rank_tensor = logits.argsort(dim=-1).argsort(dim=-1)
            target_item_ranks = rank_tensor[torch.arange(rank_tensor.size(0)), target_item_ids]
            
            
            for i, k in enumerate(topk_list):
                
                Hit_num = (target_item_ranks < k).sum().item()
                HR_total[i] += Hit_num
                
                
                mask = target_item_ranks < k
                if mask.sum() > 0:
                    NDCG_num = 1 / torch.log2(target_item_ranks[mask].float() + 2)
                    NDCG_total[i] += NDCG_num.sum().item()
    
    
    NDCG = NDCG_total.cpu() / (sample_total * (1.0 / math.log(2)))
    HR = HR_total.cpu() / sample_total
    
    
    result_dict = {}
    for i, k in enumerate(topk_list):
        result_dict[f"NDCG@{k}"] = round(NDCG[i].item(), 4)
        result_dict[f"HR@{k}"] = round(HR[i].item(), 4)
    
    return result_dict


def inference(
    model_path: str = "models/sasrec/Movie_denoise_user0.1_10000_1400_old/best_model.pth",
    test_data_path: str = "data/Movie_denoise_user0.1_10000_1400_old/test_5000.csv",
    item_num: int = 5101,
    maxlen: int = 10,
    hidden_units: int = 64,
    num_blocks: int = 2,
    num_heads: int = 1,
    dropout_rate: float = 0.5,
    batch_size: int = 256,
    topk: int = 50,
    device: str = 'cuda'
):
    
    print("=" * 80)
    print("SASRec Inference Script")
    print("=" * 80)
    print(f"Model Path: {model_path}")
    print(f"Test Data: {test_data_path}")
    print(f"Item Num: {item_num}")
    print(f"Max Length: {maxlen}")
    print(f"Batch Size: {batch_size}")
    print(f"Device: {device}")
    print("=" * 80)
    
    
    print("\n[1/4] Loading test data...")
    test_df = pd.read_csv(test_data_path)
    print(f"✅ Loaded {len(test_df)} test samples")
    print(f"   Columns: {test_df.columns.tolist()}")
    print(f"\n   Sample row:")
    print(f"   User ID: {test_df.iloc[0]['user_id']}")
    print(f"   Item IDs: {test_df.iloc[0]['item_ids']}")
    
    
    print("\n[2/4] Creating dataset and dataloader...")
    test_dataset = TestDataset(test_df, maxlen=maxlen)
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4
    )
    print(f"✅ Dataset created: {len(test_dataset)} samples")
    print(f"   Batch size: {batch_size}")
    print(f"   Number of batches: {len(test_dataloader)}")
    
    
    print("\n[3/4] Initializing SASRec model...")
    
    
    import world
    world.config = {
        "hidden_units": hidden_units,
        "maxlen": maxlen,
        "dropout_rate": dropout_rate,
        "num_blocks": num_blocks,
        "num_heads": num_heads
    }
    
    model = SASRec(item_num).to(device) 
    
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict) 
    model.eval()
    
    print(f"✅ Model loaded from: {model_path}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    
    print(f"\n[4/4] Running inference and evaluation...")
    
    
    topk_list = [1, 5, 10, 20]
    if topk not in topk_list:
        topk_list.append(topk)
    topk_list = sorted(topk_list)
    
    results = evaluate_sasrec(
        model=model,
        dataloader=test_dataloader,
        sample_total=len(test_dataset),
        topk_list=topk_list,
        device=device
    )
    
    
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    for metric, value in results.items():
        print(f"{metric:12s}: {value:.4f}")
    print("=" * 80)
    
    
    output_dir = os.path.dirname(model_path)
    output_file = os.path.join(output_dir, "inference_results.json")
    
    results_with_meta = {
        "model_path": model_path,
        "test_data_path": test_data_path,
        "num_samples": len(test_dataset),
        "item_num": item_num,
        "maxlen": maxlen,
        "batch_size": batch_size,
        "metrics": results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_with_meta, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Results saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SASRec Inference Script")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to test CSV file")
    parser.add_argument("--item_num", type=int, required=True, help="Total number of items")
    parser.add_argument("--maxlen", type=int, default=10, help="Maximum sequence length")
    parser.add_argument("--hidden_units", type=int, default=64, help="Hidden units dimension")
    parser.add_argument("--num_blocks", type=int, default=2, help="Number of transformer blocks")
    parser.add_argument("--num_heads", type=int, default=1, help="Number of attention heads")
    parser.add_argument("--dropout_rate", type=float, default=0.5, help="Dropout rate")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--topk", type=int, default=50, help="Maximum top-k for evaluation")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    
    args = parser.parse_args()
    
    inference(
        model_path=args.model_path,
        test_data_path=args.test_data_path,
        item_num=args.item_num,
        maxlen=args.maxlen,
        hidden_units=args.hidden_units,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        dropout_rate=args.dropout_rate,
        batch_size=args.batch_size,
        topk=args.topk,
        device=args.device
    )
