import json
import os
import ast
from typing import List, Dict, Any
import shutil

import pandas as pd
from tqdm import tqdm

from training_set_filter_without_natural_seqs import sample_sequences_by_popularity


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
    desc_raw = id2title_desc.get(sid, "")
    
    if isinstance(desc_raw, str):
        desc = desc_raw.strip()
    else:
        desc = ""

    if desc:
        
        if len(desc) > max_desc_len:
            desc = desc[:max_desc_len].rsplit(" ", 1)[0].rstrip() + "..."
        return f"{title}: {desc}"
    else:
        return title


def build_toy_sft_data(
    all_sequences_path: str,
    id2popularity_path: str,
    id2name_path: str,
    id2title_description_path: str,
    output_json_path: str,
    target_size: int = 10000,
    min_occurrences: int = 3,
) -> None:
    
    
    tmp_sampled_csv = os.path.splitext(all_sequences_path)[0] + "_sampled_tmp.csv"
    
    
    _ = sample_sequences_by_popularity(
        all_sequences_path=all_sequences_path,
        id2popularity_path=id2popularity_path,
        output_path=tmp_sampled_csv,
        target_size=target_size,
        min_occurreences=min_occurrences,
    )
    
    
    sampled_df = pd.read_csv(tmp_sampled_csv)

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

    if os.path.exists(tmp_sampled_csv):
        os.remove(tmp_sampled_csv)


def prepare_description_file(user0_0_path: str, user0_1_path: str):
    
    if not os.path.exists(user0_1_path):
        if os.path.exists(user0_0_path):
            shutil.copy2(user0_0_path, user0_1_path)


if __name__ == "__main__":
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    
    dataset_name = "Toy_noise_user0.1"
    dataset_root = os.path.join(project_root, "data", dataset_name)
    
    
    all_sequences_path = os.path.join(dataset_root, "train.csv")
    id2popularity_path = os.path.join(dataset_root, "id2popularity.json")
    id2name_path = os.path.join(dataset_root, "id2name.json")
    id2title_description_path = os.path.join(dataset_root, "id2title_description.json")
    
    
    user0_0_desc_path = os.path.join(
        project_root, "data", "Toy_noise_user0.0", "id2title_description.json"
    )
    prepare_description_file(user0_0_desc_path, id2title_description_path)
    
    
    output_json_path = os.path.join(
        dataset_root, "TDMD", "train_with_description.json"
    )
    
    
    required_files = [
        all_sequences_path,
        id2popularity_path,
        id2name_path,
        id2title_description_path,
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        exit(1)
    
    
    build_toy_sft_data(
        all_sequences_path=all_sequences_path,
        id2popularity_path=id2popularity_path,
        id2name_path=id2name_path,
        id2title_description_path=id2title_description_path,
        output_json_path=output_json_path,
        target_size=10000,
        min_occurrences=3,
    )
