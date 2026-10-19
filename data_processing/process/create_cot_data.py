import pandas as pd
import json
import os
import ast
import time
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("process.log"), 
        logging.StreamHandler()             
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()
os.environ["OPENROUTER_API_KEY"] = "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"
client = OpenAI(api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1")



BASE_DIR = "../data/Movie_noise_user0.0/"
SAMPLED_DATA_PATH = os.path.join(BASE_DIR, "train_sampled_noisy_only_10k.csv")
ID2NAME_PATH = os.path.join(BASE_DIR, "id2name.json")
ID2TITLE_DESCRIPTION_PATH = os.path.join(BASE_DIR, "id2title_description.json")
OUTPUT_PATH = os.path.join(BASE_DIR, "cot_data_sampled_noisy_only_10k.jsonl") 

def build_prompt_inputs(row, id2name, knowledge_base):
    try:
        seq_ids = ast.literal_eval(row['item_ids'])
        mask = ast.literal_eval(row['mask'])
        replaced_list = ast.literal_eval(row["replaced_item_titles"])
        target_title = replaced_list[0] if replaced_list else None 
    except Exception as e:
        print(f"Error building prompt inputs: {e}")
        return None, None, None
    
    teacher_seq_lines = []
    student_seq_lines = []
    noise_title = "None"
    
    for i, item_id in enumerate(seq_ids):
        item_id_str = str(item_id)
        title = id2name.get(item_id_str, "Unknown")
        item_info = knowledge_base.get(item_id_str, "Unknown")
        
        teacher_seq_lines.append(f"{i+1}. \"{title}\"\n   [Context: {item_info}]")
        student_seq_lines.append(f"\"{title}\"")
        
        if mask[i] == 1:
            noise_title = title

    teacher_context = "\n".join(teacher_seq_lines)
    
    teacher_content = f"""
**User Interaction Sequence:**
{teacher_context}

**Ground Truth (The Correct Answer):**
- Noise Item: "{noise_title}"
- Suggested Replacement: "{target_title}"

**Requirement:**
Analyze the sequence. Explain why '{noise_title}' conflicts with the user's core interests (Genre/Style/Theme) defined by the other items. Explain why '{target_title}' is a good fit.
"""
    logger.info(f"Teacher Content: {teacher_content}")
    student_input_str = f"User Interaction Sequence: {', '.join(student_seq_lines)}"
    
    return teacher_content, student_input_str, noise_title, target_title

def generate_cot_analysis(user_content):
    
    system_prompt = 
    
    try:
        response = client.chat.completions.create(
            model="deepseek/deepseek-v3.2", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.1,
            max_tokens=512
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API Error: {e}")
        return None

def load_data():
    df = pd.read_csv(SAMPLED_DATA_PATH)
    
    with open(ID2NAME_PATH, 'r') as f:
        id2name = json.load(f)
        
    if not os.path.exists(ID2TITLE_DESCRIPTION_PATH):
        raise FileNotFoundError(f"Knowledge base file not found: {ID2TITLE_DESCRIPTION_PATH}")
        
    with open(ID2TITLE_DESCRIPTION_PATH, 'r') as f:
        knowledge_base = json.load(f)
        
    return df, id2name, knowledge_base

def main():
    df, id2name, knowledge_base = load_data()
    
    
    processed_count = 0
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
            for _ in f:
                processed_count += 1

    
    with open(OUTPUT_PATH, 'a', encoding='utf-8') as f_out:
        
        for index, row in tqdm(df.iterrows(), total=len(df), mininterval=30):
            if index < processed_count:
                continue
            
            teacher_content, student_input, noise_title, target_title = build_prompt_inputs(row, id2name, knowledge_base)
            
            if not teacher_content: 
                continue 
                
            cot_output = generate_cot_analysis(teacher_content)
            
            if not cot_output:
                time.sleep(2)
                continue
            if cot_output.strip().startswith("<analysis>"):
                cot_output = cot_output.strip().replace("<analysis>", "", 1) 
            if cot_output.strip().endswith("</analysis>"):
                cot_output = cot_output.strip().replace("</analysis>", "", 1)
            final_output = f"<analysis>{cot_output}</analysis>\nNoise Items: \"{noise_title}\"\nSuggested Replacements: \"{target_title}\""
            
            sft_sample = {
                "instruction": "You are to analyze a list of item titles provided by a user. Identify any noise item that does not align with the main interests and suggest a replacement.",
                "input": student_input,
                "output": final_output
            }
            
            f_out.write(json.dumps(sft_sample, ensure_ascii=False) + "\n")
            f_out.flush()
            
    pass
            
if __name__ == "__main__":
    main()