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
    
    
    def parse_list(x):
        try:
            if isinstance(x, str):
                return ast.literal_eval(x)
            return x
        except:
            return []
            
    def stringify_ids(items):
        return [str(item) for item in items]

    df = pd.read_csv(all_sequences_path)
    df['parsed_ids'] = df['item_ids'].apply(parse_list).apply(stringify_ids)
    df['parsed_mask'] = df['mask'].apply(parse_list)
    noisy_indices = df[df['parsed_mask'].apply(lambda x: any(v != 0 for v in x))].index.tolist()
    
    with open(id2popularity_path, 'r') as f:
        id2popularity = json.load(f)
    id2popularity = {str(k): v for k, v in id2popularity.items()}     

    
    item_to_seq_indices = defaultdict(list)
    items_in_datasets = set()
    
    
    all_parsed_ids = df['parsed_ids'].tolist()
    
    for idx in tqdm(noisy_indices):
        seq = all_parsed_ids[idx]
        for item in seq:
            item_to_seq_indices[item].append(idx)
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
            
            for idx in to_add:
                selected_sequences.add(idx)
                
                for seq_item in all_parsed_ids[idx]:
                    current_item_coverage[seq_item] += 1
                    
                if len(selected_sequences) >= target_size:
                    break
                
    if len(selected_sequences) < target_size:
        remaining_size = target_size - len(selected_sequences)
        candidate_pool = set(noisy_indices) - selected_sequences
        candidate_list = list(candidate_pool)
        
        if len(candidate_list) > remaining_size:
            extras = random.sample(candidate_list, remaining_size)
            selected_sequences.update(extras)
        else:
            selected_sequences.update(candidate_list)
            
    
    final_coverage_count = Counter()
    for idx in selected_sequences:
        for item in all_parsed_ids[idx]:
            final_coverage_count[item] += 1
    
    
    selected_list = sorted(list(selected_sequences))
    result_df = df.iloc[selected_list].copy()
    result_df.to_csv(output_path, index=False)
    
    return result_df
            

if __name__ == "__main__":
    
    all_sequences_path = '../data/Movie_noise_user0.1/train.csv'
    id2popularity_path = '../data/Movie_noise_user0.1/id2popularity.json'
    
    output_path = '../data/Movie_noise_user0.1/train_sampled_noisy_only_10k.csv'
    target_size = 10000
    min_occurreences = 3
    
    sampled_df = sample_sequences_by_popularity(all_sequences_path, id2popularity_path, output_path, target_size, min_occurreences)
