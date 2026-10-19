import json
import re
from improved_reward_funcs import extract_titles


with open('diagnosis/analysis_5k_20260107_000600/prepared_data.jsonl', 'r') as f:
    lines = [json.loads(line) for line in f]


stats = {
    'total': len(lines),
    'has_noise_items_field': 0,
    'has_suggested_field': 0,
    'correctly_parsed': 0,
    'empty_response': 0,
    'format_variants': {}
}

for i, item in enumerate(lines[:100]):  
    response = item.get('response', '')
    
    if not response or len(response) < 10:
        stats['empty_response'] += 1
        continue
    
    
    if re.search(r'Noise Items', response, re.I):
        stats['has_noise_items_field'] += 1
    if re.search(r'Suggested', response, re.I):
        stats['has_suggested_field'] += 1
    
    
    pred_noises = extract_titles(response, "Noise Items")
    if pred_noises:
        stats['correctly_parsed'] += 1
    
    
    first_line = response.split('\n')[0][:80]
    stats['format_variants'][first_line] = stats['format_variants'].get(first_line, 0) + 1
    
    
    pass
