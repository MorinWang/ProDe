import json
import pandas as pd
import random
import ast
from collections import Counter
from tqdm import tqdm
from collections import defaultdict

def sample_sequences_by_popularity(
    all_sequences_path: str,
    id2popularity_path: str,
    output_path: str,
    target_size = 10000,
    min_occurreences = 3
):
    
    def parse_and_stringify_mask(x):
        try:
            if isinstance(x, str):
                items = ast.literal_eval(x)
            
            else:
                items = x
            return [str(item) for item in items]
        except:
            return []
        
    df = pd.read_csv(all_sequences_path)
    all_sequences = df['item_ids'].apply(parse_and_stringify_mask).tolist()
    
    
    with open(id2popularity_path, 'r') as f:
        id2popularity = json.load(f)
        
    id2popularity = {str(k): v for k, v in id2popularity.items()}
    
        
    item_to_seq_indices = defaultdict(list)
    
    items_in_datasets = set()
    
    for i,seq in enumerate(tqdm(all_sequences)):
        for item in seq:
            item_to_seq_indices[item].append(i)
            items_in_datasets.add(item)
            
            
    sorted_items = sorted(
        list(items_in_datasets),
        key=lambda x: id2popularity.get(x, 0),
    )
    
    
    selected_sequences = set()
    current_item_coverage = Counter()
    
    
    for item in tqdm(sorted_items):
        
        if len(selected_sequences) >= target_size:
            break
        
        
        current_count = current_item_coverage[item]
        
        if current_count < min_occurreences:
            needed = min_occurreences - current_count
            
            
            candidate_indices = item_to_seq_indices[item]
            
            
            available_indices = [i for i in candidate_indices if i not in selected_sequences]
            
            
            random.shuffle(available_indices)
            
            
            to_add = available_indices[:needed]
            
            for i in to_add:
                selected_sequences.add(i)
                
                for seq_item in all_sequences[i]:
                    current_item_coverage[seq_item] += 1
                    
                if len(selected_sequences) >= target_size:
                    break
                
    sampled_sequences = [all_sequences[i] for i in selected_sequences]
    sampled_df = pd.DataFrame({'item_ids': sampled_sequences})
    sampled_df.to_csv(output_path, index=False)
    
    if len(selected_sequences) < target_size:
        remaining_size = target_size - len(selected_sequences)
        
        all_indices = set(range(len(all_sequences)))
        
        unselected_indices = list(all_indices - selected_sequences)
        
        if len(unselected_indices) > remaining_size:
            extras = random.sample(unselected_indices, remaining_size)
            selected_sequences.update(extras)
        else:
            selected_sequences.update(unselected_indices)
            
        
        final_coverage_count = Counter()
        
        for idx in selected_sequences:
            for item in all_sequences[idx]:
                final_coverage_count[item] += 1
                
        
        under_represented = 0
        zero_represented = 0
        for item in items_in_datasets:
            if final_coverage_count[item] == 0:
                zero_represented += 1
            elif final_coverage_count[item] < min_occurreences:
                under_represented += 1
                
        selected_df = df.iloc[list(selected_sequences)].copy()
        selected_df.to_csv(f"{output_path}_10000.csv", index=False)
    return selected_df
if __name__ == "__main__":
    
    all_sequences_path = 'data/Movie_noise_user0.1/train.csv'
    id2popularity_path = 'data/Movie_noise_user0.1/id2popularity.json'
    output_path = 'data/Movie_noise_user0.1/train_sampled.csv'
    target_size = 10000
    min_occurreences = 3
    
    sampled_df = sample_sequences_by_popularity(all_sequences_path, id2popularity_path, output_path, target_size, min_occurreences)
    
    print(sampled_df.head())
