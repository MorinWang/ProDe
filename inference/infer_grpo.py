import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,6,7,8"
os.environ["VLLM_USE_V1"] = "0" 
import json
import pandas as pd
import ast
import re
import random
from tqdm import tqdm
from vllm import LLM, SamplingParams


MODEL_PATH = "models/llama3_8B_Instruct_grpo_denoised_merged"







ID2NAME_PATH = "data/Movie_denoise_user0.1_10000_1400/id2name4Rec.json"
ID2DESC_PATH = "data/Movie_noise_user0.0/id2title_description.json"
TEST_DATA_PATH = "data/Movie_noise_user0.1/TDMD/test_with_description.json"
OUTPUT_PATH = "bad_case_analysis/vllm_grpo_results_ranking_v1.jsonl"





def generate_inference_prompt(item_ids, id2name, id2desc):
    instruction = "You are to analyze a list of item titles and descriptions provided by a user. Your task is to identify any item(s) that do not align with the main interests reflected by the majority of the items. After identifying these noise items, suggest replaced items that better match the user's interests."
    
    example = 
    formatted_items = []
    for idx in item_ids:
        title = id2name.get(idx, "Unknown Title")
        desc = id2desc.get(idx, "")
        formatted_items.append(f'"{title}: {desc}"' if desc else f'"{title}"')
            
    history = "User Interaction Sequence: " + ", ".join(formatted_items)
    
    return f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n{example}\n### Input:\n{history}\n\n### Response:\nNoise Items:"

def main():
    
    print("🚀 Initializing vLLM Engine...")
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=4, 
        gpu_memory_utilization=0.8,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=2048,
        enforce_eager=True
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=256,
        repetition_penalty=1.1,
        stop=["<|endoftext|>", "###"]
    )

    
    print("📂 Loading Metadata...")
    with open(ID2NAME_PATH, "r", encoding="utf-8") as f:
        id2name = {int(k): v for k, v in json.load(f).items()}
    with open(ID2DESC_PATH, "r", encoding="utf-8") as f:
        id2desc = {int(k): v for k, v in json.load(f).items()}
    name2id = {v.lower(): k for k, v in id2name.items()}

    
    print(f"📄 Loading Data from {TEST_DATA_PATH}...")
    if TEST_DATA_PATH.endswith(".json"):
        with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
        
        
        test_samples = random.sample(all_data, min(len(all_data), 1000))
    else:
        df = pd.read_csv(TEST_DATA_PATH)
        test_samples = df.to_dict(orient='records')

    
    prompts = []
    gt_noises_list = []
    for item in test_samples:
        if TEST_DATA_PATH.endswith(".json"):
            
            if 'prompt' in item:
                prompts.append(item['prompt'] + "Noise Items:")
            else:
                instruction = item.get('instruction', "")
                
                prompt = (
                    "Below is an instruction that describes a task, paired with an input that provides further context. "
                    "Write a response that appropriately completes the request.\n\n"
                    f"### Instruction:\n{instruction}\n\n"
                    f"### Input:\n{item.get('input', '')}\n\n"
                    "### Response:\nNoise Items:"
                )
                prompts.append(prompt)
            
            
            output_text = item.get('output', "")
            noise_match = re.search(r"Noise Items:\s*(.*?)(?=\n|Suggested|Replacement|$)", output_text, re.S | re.I)
            if noise_match:
                gt = re.findall(r'"(.*?)"', noise_match.group(1))
                gt_noises_list.append([t.split(":")[0].strip() for t in gt])
            else:
                gt_noises_list.append([])
        else:
            item_ids = ast.literal_eval(item["item_ids"])
            mask = ast.literal_eval(item["mask"])
            prompts.append(generate_inference_prompt(item_ids, id2name, id2desc))
            gt_noises_list.append([id2name.get(item_ids[idx], "Unknown").split(":")[0].strip() for idx, m in enumerate(mask) if m == 1])

    
    print(f"🔥 Generating Responses for {len(prompts)} samples...")
    outputs = llm.generate(prompts, sampling_params)

    
    results = []
    for output, gt_noises in zip(outputs, gt_noises_list):
        resp = "Noise Items:" + output.outputs[0].text
        
        
        noise_match = re.search(r"Noise Items:\s*(.*?)(?=\n?(?:Suggested|Alternative|Recommended|Alternate|Replacement|Replacing)|$)", resp, re.S | re.I)
        noise_part = noise_match.group(1).strip() if noise_match else ""
        
        suggest_match = re.search(r"(?:Suggested|Alternative|Recommended|Alternate|Recommended Replacing)\s*(?:Alternatives|Replacements|Replacement|Items|Item|Replacing Item)?:\s*(.*)", resp, re.S | re.I)
        suggest_part = suggest_match.group(1).strip() if suggest_match else ""

        pred_noise_titles = re.findall(r'"(.*?)"', noise_part)
        if not pred_noise_titles:
            pred_noise_titles = [n.strip().strip("'\"").split(":")[0].strip() for n in noise_part.split(",") if n.strip()]
        else:
            pred_noise_titles = [t.split(":")[0].strip() for t in pred_noise_titles]

        hit_count = sum(1 for gt_n in gt_noises if any(gt_n.lower() in pn.lower() or pn.lower() in gt_n.lower() for pn in pred_noise_titles))
        
        results.append({
            "gt_noises": gt_noises,
            "pred_noise_raw": noise_part,
            "pred_suggest_raw": suggest_part,
            "hit_any": hit_count > 0,
            "hit_all": hit_count == len(gt_noises) if gt_noises else True,
            "raw_response": resp
        })

    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = len(results)
    hr_any = sum(1 for r in results if r["hit_any"]) / total
    hr_all = sum(1 for r in results if r["hit_all"]) / total
    print(f"\n✅ Done! Report saved to {OUTPUT_PATH}")
    print(f"Total: {total}, Hit Rate (Any): {hr_any:.2%}")
    print(f"Hit Rate (All): {hr_all:.2%}")

if __name__ == "__main__":
    main()

