

import pandas as pd
import numpy as np
import json
import ast
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


def parse_list_column(col):
    
    if isinstance(col, str):
        return ast.literal_eval(col)
    return col


def evaluate_denoise_performance(csv_path, threshold_percentile=0.6):
    
    df = pd.read_csv(csv_path)
    
    
    df['mask'] = df['mask'].apply(parse_list_column)
    df['noise_prob'] = df['noise_prob'].apply(parse_list_column)
    df['item_ids'] = df['item_ids'].apply(parse_list_column)
    
    
    all_true_labels = []
    all_pred_probs = []
    all_item_ids = []
    
    for idx, row in df.iterrows():
        mask = row['mask']
        noise_prob = row['noise_prob']
        item_ids = row['item_ids']
        
        if len(item_ids) == len(noise_prob) + 1:
            mask = mask[:-1]
            item_ids = item_ids[:-1]
        
        if len(mask) != len(noise_prob) or len(mask) != len(item_ids):
            raise ValueError(f"mask: {len(mask)}, noise_prob: {len(noise_prob)}, item_ids: {len(item_ids)}")
            
        all_true_labels.extend(mask)
        all_pred_probs.extend(noise_prob)
        all_item_ids.extend(item_ids)
    
    all_true_labels = np.array(all_true_labels)
    all_pred_probs = np.array(all_pred_probs)
    all_item_ids = np.array(all_item_ids)
    
    threshold = np.percentile(all_pred_probs,  threshold_percentile * 100)
    all_pred_labels = (all_pred_probs >= threshold).astype(int)
    
    accuracy = accuracy_score(all_true_labels, all_pred_labels)
    precision = precision_score(all_true_labels, all_pred_labels, zero_division=0)
    recall = recall_score(all_true_labels, all_pred_labels, zero_division=0)
    f1 = f1_score(all_true_labels, all_pred_labels, zero_division=0)
    
    try:
        auc = roc_auc_score(all_true_labels, all_pred_probs)
        ap = average_precision_score(all_true_labels, all_pred_probs)
    except Exception as e:
        pass
    
    cm = confusion_matrix(all_true_labels, all_pred_labels)
    tn, fp, fn, tp = cm.ravel()
    
    best_f1 = 0
    best_threshold_pct = 0
    
    for pct in [0.5, 0.6, 0.7, 0.8, 0.9]:
        thresh = np.percentile(all_pred_probs,  pct * 100)
        pred_labels = (all_pred_probs >= thresh).astype(int)
        
        acc = accuracy_score(all_true_labels, pred_labels)
        prec = precision_score(all_true_labels, pred_labels, zero_division=0)
        rec = recall_score(all_true_labels, pred_labels, zero_division=0)
        f1_tmp = f1_score(all_true_labels, pred_labels, zero_division=0)
        
        if f1_tmp > best_f1:
            best_f1 = f1_tmp
            best_threshold_pct = pct
    
    noise_probs = all_pred_probs[all_true_labels == 1]
    normal_probs = all_pred_probs[all_true_labels == 0]
    
    seq_correct = 0
    seq_total = 0
    seq_with_noise = 0
    seq_noise_detected = 0
    
    for idx, row in df.iterrows():
        mask = row['mask']
        noise_prob = row['noise_prob']
        
        if len(mask) == len(noise_prob) + 1:
            mask = mask[:-1]
        
        if len(mask) != len(noise_prob):
            raise ValueError(f"mask: {len(mask)}, noise_prob: {len(noise_prob)}")
        
        seq_total += 1
        has_noise = any(mask)
        
        if has_noise:
            seq_with_noise += 1
            
        
        pred_labels = [1 if p >= threshold else 0 for p in noise_prob]
        
        
        if mask == pred_labels:
            seq_correct += 1
        
        
        if has_noise and any(pred_labels):
            seq_noise_detected += 1
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'best_threshold_pct': best_threshold_pct,
        'best_f1': best_f1,
        'all_true_labels': all_true_labels,
        'all_pred_probs': all_pred_probs,
        'threshold': threshold
    }


def compare_train_test(train_path, test_path, threshold_percentile=0.6):
    train_results = evaluate_denoise_performance(train_path, threshold_percentile)
    test_results = evaluate_denoise_performance(test_path, threshold_percentile)


def analyze_error_cases(csv_path, threshold_percentile=0.6, num_examples=10):
    
    df = pd.read_csv(csv_path)
    df['mask'] = df['mask'].apply(parse_list_column)
    df['noise_prob'] = df['noise_prob'].apply(parse_list_column)
    df['item_ids'] = df['item_ids'].apply(parse_list_column)
    df['item_titles'] = df['item_titles'].apply(parse_list_column)
    
    
    all_probs = []
    for probs in df['noise_prob']:
        all_probs.extend(probs)
    threshold = np.percentile(all_probs,  threshold_percentile * 100)
    
    false_positives = []  
    false_negatives = []  
    
    for idx, row in df.iterrows():
        mask = row['mask']
        noise_prob = row['noise_prob']
        item_titles = row['item_titles']
        
        if len(mask) != len(noise_prob):
            continue
        
        for i, (true_label, pred_prob, title) in enumerate(zip(mask, noise_prob, item_titles)):
            pred_label = 1 if pred_prob >= threshold else 0
            
            if true_label == 0 and pred_label == 1:
                
                false_positives.append({
                    'user_id': row['user_id'],
                    'position': i,
                    'title': title,
                    'prob': pred_prob,
                    'sequence': item_titles
                })
            elif true_label == 1 and pred_label == 0:
                
                false_negatives.append({
                    'user_id': row['user_id'],
                    'position': i,
                    'title': title,
                    'prob': pred_prob,
                    'sequence': item_titles
                })
    
    pass


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, 
                       default="./data/Movie_denoise_user0.1_5000_3950/train.csv")
    parser.add_argument("--test_path", type=str,
                       default="./data/Movie_denoise_user0.1_5000_3950/test_5000.csv")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--mode", type=str, default="both",
                       choices=["train", "test", "both", "compare", "error"],
                       help="Evaluation mode")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        evaluate_denoise_performance(args.train_path, args.threshold)
    elif args.mode == "test":
        evaluate_denoise_performance(args.test_path, args.threshold)
    elif args.mode == "both":
        evaluate_denoise_performance(args.train_path, args.threshold)
        evaluate_denoise_performance(args.test_path, args.threshold)
    elif args.mode == "compare":
        compare_train_test(args.train_path, args.test_path, args.threshold)
    elif args.mode == "error":
        analyze_error_cases(args.train_path, args.threshold, num_examples=5)
        analyze_error_cases(args.test_path, args.threshold, num_examples=5)
