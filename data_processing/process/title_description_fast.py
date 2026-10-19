from openai import OpenAI
import os
import json
import json_repair
import time
from tqdm import tqdm
from prompts import (
    system_prompt_for_Movies_Title_Description,
    system_prompts_for_Toys_Title_Description,
)


client = OpenAI(api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1")


def get_item_context(item_id, title, meta_db, category):
    
    
    context = f"Title: {title}\n"
    
    
    item_meta = meta_db.get(str(item_id))
    
    if not item_meta or item_meta.get('match_type') == 'none':
        return context + "Reference Info: None (Please use your internal knowledge based on Title)."

    full_meta = item_meta.get('full_metadata', {})
    
    
    if category == "Movie":
        
        desc_list = item_meta.get('description', '')
        if isinstance(desc_list, list):
            desc_text = desc_list[0] if desc_list else ""
        else:
            desc_text = str(desc_list)
        if len(desc_text) > 400: 
            desc_text = desc_text[:400] + "..."
        
        
        details = full_meta.get('details', {})
        credits = []
        if details:
            if 'Director' in details: credits.append(f"Director: {details['Director']}")
            if 'Actors' in details: credits.append(f"Stars: {details['Actors']}")
        credits_text = " | ".join(credits)
        
        
        cats = full_meta.get('categories', [])
        cat_text = ", ".join(cats[-2:]) if cats else ""

        ref_str = ""
        if cat_text: ref_str += f"Categories: {cat_text}\n"
        if credits_text: ref_str += f"Credits: {credits_text}\n"
        if desc_text: ref_str += f"Plot Summary: {desc_text}"
        
        if not ref_str: ref_str = "None available."
    elif category == "Toy":
        
        
        desc_text = ""
        
        
        outer_desc = item_meta.get('description', '')
        if isinstance(outer_desc, list) and outer_desc:
            desc_text = outer_desc[0]
        elif isinstance(outer_desc, str):
            desc_text = outer_desc
        
        
        if not desc_text:
            inner_desc = full_meta.get('description', '')
            if isinstance(inner_desc, list) and inner_desc:
                desc_text = inner_desc[0]
            elif isinstance(inner_desc, str):
                desc_text = inner_desc
        
        
        if len(desc_text) > 300:
            desc_text = desc_text[:300] + "..."
        
        
        cats = full_meta.get('categories', [])
        cat_text = ", ".join(cats[-3:]) if cats else ""  
        
        
        details = full_meta.get('details', {})
        info_parts = []
        
        
        recommended_age = details.get('Manufacturer recommended age', '')
        if recommended_age:
            info_parts.append(f"Recommended Age: {recommended_age}")
        
        
        info_text = " | ".join(info_parts)
        
        
        features = full_meta.get('features', [])
        feature_text = ""
        if features and len(features) > 0:
            
            top_features = []
            for i, feature in enumerate(features[:3]):
                if len(feature) > 60:
                    feature = feature[:57] + "..."
                top_features.append(feature)
            feature_text = "Key Features: " + "; ".join(top_features)
    
        
        ref_str = ""
        
        
        if cat_text:
            ref_str += f"Categories: {cat_text}\n"
        
        
        if info_text:
            ref_str += f"Details: {info_text}\n"
        
        
        if feature_text:
            ref_str += f"{feature_text}\n"
        
        
        if desc_text:
            ref_str += f"Description: {desc_text}"
        
        
        if not ref_str.strip():
            ref_str = "None available (Please use your internal knowledge)."

    return context + f"Reference Info:\n{ref_str}"

def generate_title_description(batch_dict, category):
    
    batch_str = json.dumps(batch_dict, ensure_ascii=False, indent=2)
    
    
    if category == "Movie":
        system_prompt = system_prompt_for_Movies_Title_Description()
    elif category == "Toy":
        system_prompt = system_prompts_for_Toys_Title_Description()
    else:
        raise ValueError(f"Unknown category: {category}")
    
    
    user_prompt = f"Here is the {category} list with reference info:\n{batch_str}\n\nProvide the JSON output:"
    
    try:
        response = client.chat.completions.create(
            model="deepseek/deepseek-v3.2", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating title and description: {e}")
        return None
    
    
def main():
    import sys
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    if len(sys.argv) > 1:
        CATEGORY = sys.argv[1]
    else:
        CATEGORY = "Movie"
    
    print(f"Processing {CATEGORY}...")
        
    
    if CATEGORY == "Movie":
        INPUT_PATH = "../data/Movie_noise_user0.0/id2name4Rec.json"
        OUTPUT_PATH = "../data/Movie_noise_user0.0/id2title_description.json"
        INPUT_METADATA = "../data/Movie_noise_user0.1/matched_metadata.json"
    elif CATEGORY == "Toy":
        INPUT_PATH = "../data/Toy_noise_user0.0/id2name4Rec.json"
        OUTPUT_PATH = "../data/Toy_noise_user0.0/id2title_description.json"
        INPUT_METADATA = "../data/Toy_noise_user0.1/matched_metadata.json"
    
    BATCH_SIZE = 10
    MAX_WORKERS = 5  
    
    
    with open(INPUT_PATH, "r") as f:
        id2name = json.load(f)
    items = list(id2name.items())
    
    meta_db = {}
    if os.path.exists(INPUT_METADATA):
        with open(INPUT_METADATA, "r", encoding='utf-8') as f:
            meta_db = json.load(f)

    
    knowledge_base = {}
    JSONL_PATH = OUTPUT_PATH.replace(".json", ".jsonl")
    
    
    if os.path.exists(OUTPUT_PATH):
        try:
            with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
                knowledge_base.update(json.load(f))
        except: pass
        
    
    if os.path.exists(JSONL_PATH):
        with open(JSONL_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        knowledge_base.update(json.loads(line))
                    except: continue
    
    pending_items = [item for item in items if item[0] not in knowledge_base]

    
    def process_one_batch(batch_slice):
        batch_input = {item_id: get_item_context(item_id, title, meta_db, CATEGORY) 
                       for item_id, title in batch_slice}
        result_str = generate_title_description(batch_input, CATEGORY)
        if result_str:
            try:
                result_dict = json_repair.loads(result_str)
                
                return {k: v for k, v in result_dict.items() if k in batch_input}
            except:
                return None
        return None

    
    with open(JSONL_PATH, 'a', encoding='utf-8') as jf:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            batches = [pending_items[i:i + BATCH_SIZE] for i in range(0, len(pending_items), BATCH_SIZE)]
            futures = {executor.submit(process_one_batch, b): b for b in batches}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                valid_updates = future.result()
                if valid_updates:
                    knowledge_base.update(valid_updates)
                    
                    jf.write(json.dumps(valid_updates, ensure_ascii=False) + "\n")
                    jf.flush() 

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(knowledge_base, f, ensure_ascii=False, indent=4)
    
if __name__ == "__main__":
    main()
