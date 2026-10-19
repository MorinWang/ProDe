import json
import os
import ast
import random
from typing import List, Dict, Any

import pandas as pd
from tqdm import tqdm


def _parse_list(x):
    
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return []
    return []


def _build_item_text(
    item_id: str,
    id2name: Dict[str, str],
    id2title_desc: Dict[str, str],
    max_desc_len: int = 256,
) -> str:
    
    sid = str(item_id)
    title = id2name.get(sid, f"Item {sid}")
    desc = id2title_desc.get(sid, "").strip()

    if desc:
        
        if len(desc) > max_desc_len:
            desc = desc[:max_desc_len].rsplit(" ", 1)[0].rstrip() + "..."
        return f"{title}: {desc}"
    else:
        return title


def build_test_with_description(
    test_sequences_path: str,
    id2name_path: str,
    id2title_description_path: str,
    output_json_path: str,
    target_size: int = 1000,
    random_seed: int = 42,
) -> None:
    
    df = pd.read_csv(test_sequences_path)
    
    def has_noise(mask_str):
        try:
            mask = ast.literal_eval(mask_str) if isinstance(mask_str, str) else mask_str
            return any(m != 0 for m in mask)
        except:
            return False
    
    noisy_df = df[df['mask'].apply(has_noise)].copy()
    
    if len(noisy_df) > target_size:
        random.seed(random_seed)
        sampled_df = noisy_df.sample(n=target_size, random_state=random_seed)
    else:
        sampled_df = noisy_df

    with open(id2name_path, "r", encoding="utf-8") as f:
        id2name = json.load(f)

    with open(id2title_description_path, "r", encoding="utf-8") as f:
        id2title_desc = json.load(f)
    records: List[Dict[str, Any]] = []

    
    instruction_text = (
        "You are to analyze a list of item titles provided by a user. "
        "Your task is to identify any item(s) that do not align with the main interests "
        "reflected by the majority of the items. After identifying these noise items, "
        "suggest alternative items that better match the user's interests. "
        "If there is no noise item, please output None."
    )

    for _, row in tqdm(sampled_df.iterrows(), total=len(sampled_df)):
        item_ids = _parse_list(row.get("item_ids", []))
        mask_list = _parse_list(row.get("mask", []))
        item_titles = _parse_list(row.get("item_titles", []))
        replaced_titles = _parse_list(row.get("replaced_item_titles", []))

        
        item_texts: List[str] = []
        for iid in item_ids:
            item_texts.append(_build_item_text(iid, id2name, id2title_desc))

        
        quoted_items = [f"\"{t}\"" for t in item_texts]
        input_text = "User Interaction Sequence: " + ", ".join(quoted_items)

        
        noise_titles: List[str] = []
        for idx, v in enumerate(mask_list):
            if v != 0 and idx < len(item_titles):
                noise_titles.append(str(item_titles[idx]).strip())

        
        suggest_titles: List[str] = [str(t).strip() for t in replaced_titles if t]
        
        
        if len(noise_titles) != len(suggest_titles) and len(noise_titles) > 0:
            if len(suggest_titles) > len(noise_titles):
                suggest_titles = suggest_titles[:len(noise_titles)]

        def titles_to_str(titles: List[str]) -> str:
            titles = [t for t in titles if t]  
            if not titles:
                return "None"
            if len(titles) == 1:
                return f"\"{titles[0]}\""
            return ", ".join(f"\"{t}\"" for t in titles)

        noise_part = titles_to_str(noise_titles)
        suggest_part = titles_to_str(suggest_titles)

        output_text = f"Noise Items: {noise_part}\nSuggested Replacements: {suggest_part}"

        record = {
            "instruction": instruction_text,
            "input": input_text,
            "output": output_text,
        }
        records.append(record)

    
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_root = os.path.join(project_root, "data", "Movie_noise_user0.1")

    
    test_sequences_path = os.path.join(dataset_root, "test.csv")
    id2name_path = os.path.join(dataset_root, "id2name.json")

    
    id2title_description_path = os.path.join(
        project_root, "data", "Movie_noise_user0.0", "id2title_description.json"
    )

    
    output_json_path = os.path.join(
        dataset_root, "TDMD", "test_with_description_5k.json"
    )

    build_test_with_description(
        test_sequences_path=test_sequences_path,
        id2name_path=id2name_path,
        id2title_description_path=id2title_description_path,
        output_json_path=output_json_path,
        target_size=5000,  
        random_seed=42,    
    )
