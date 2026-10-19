

import torch
import argparse
from model import SASRec, Bert4Rec, GRU


def convert(state_dict_path, output_path, model_type, item_num):
    
    
    print(f"Loading state_dict from: {state_dict_path}")
    state_dict = torch.load(state_dict_path, map_location='cpu')
    
    
    print(f"Initializing {model_type} model with item_num={item_num}")
    if model_type == "SASRec":
        model = SASRec(item_num)
    elif model_type == "Bert4Rec":
        model = Bert4Rec(item_num)
    elif model_type == "GRU":
        model = GRU(item_num)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    
    print("Loading state_dict into model...")
    model.load_state_dict(state_dict)
    
    
    print(f"Saving full model to: {output_path}")
    torch.save(model, output_path)
    
    print("✓ Conversion completed successfully!")
    print(f"\nYou can now use this model in LLaRA:")
    print(f"  --rec_model_path {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--state_dict_path", type=str, required=True,
                       help="Path to the state_dict file (.pth)")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Output path for the full model (.pt)")
    parser.add_argument("--model_type", type=str, default="SASRec",
                       choices=["SASRec", "Bert4Rec", "GRU"],
                       help="Model type")
    parser.add_argument("--item_num", type=int, required=True,
                       help="Number of items in the dataset")
    
    args = parser.parse_args()
    
    convert(
        state_dict_path=args.state_dict_path,
        output_path=args.output_path,
        model_type=args.model_type,
        item_num=args.item_num
    )
