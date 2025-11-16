# 1_predict_on_unlabeled.py

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from datasets import Dataset
from tqdm import tqdm
import json

# --- Configuration ---
TEACHER_MODEL_PATH = "censored"
LISTING_FILE = 'censored'
PREDICTIONS_OUTPUT_FILE = "censored"
UNLABELED_DATA_RANGE = (5001, 999999)
BATCH_SIZE = 32

def load_unlabeled_data(file_path: str, data_range: tuple) -> Dataset:
    print(f"Loading unlabeled data from: {file_path}")
    column_names = ['Record Number', 'Category Id', 'Title']
    # Added low_memory=False to handle the DtypeWarning on large files
    listing_df = pd.read_csv(
        file_path, sep='\t', header=None, names=column_names,
        keep_default_na=False, na_values=None, low_memory=False
    )
    
    listing_df['Record Number'] = pd.to_numeric(listing_df['Record Number'], errors='coerce')
    listing_df['Category Id'] = pd.to_numeric(listing_df['Category Id'], errors='coerce')
    
    listing_df.dropna(subset=['Record Number', 'Category Id'], inplace=True)
    
    listing_df['Record Number'] = listing_df['Record Number'].astype(int)
    listing_df['Category Id'] = listing_df['Category Id'].astype(int)
    
    subset_df = listing_df[
        (listing_df['Record Number'] >= data_range[0]) & 
        (listing_df['Record Number'] < data_range[1])
    ].copy()
    
    print(f"Loaded and cleaned {len(subset_df)} unlabeled records.")
    return Dataset.from_pandas(subset_df)

def main():
    print(f"Loading teacher model from {TEACHER_MODEL_PATH}")
    device = 0 if torch.cuda.is_available() else -1
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_PATH, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(TEACHER_MODEL_PATH).to(device)
    model.eval()

    unlabeled_dataset = load_unlabeled_data(LISTING_FILE, UNLABELED_DATA_RANGE)
    id2tag = model.config.id2label

    with open(PREDICTIONS_OUTPUT_FILE, 'w') as writer:
        for i in tqdm(range(0, len(unlabeled_dataset), BATCH_SIZE)):
            batch = unlabeled_dataset[i:i+BATCH_SIZE]
            titles = batch["Title"]
            
            inputs = tokenizer(titles, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)

            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            top_probs, top_preds = torch.max(probabilities, dim=-1)

            for j in range(len(titles)):
                input_ids = inputs["input_ids"][j]
                word_ids = inputs.word_ids(batch_index=j)
                preds = top_preds[j].cpu().numpy()
                probs = top_probs[j].cpu().numpy()
                
                aligned_predictions = []
                current_word_subwords = []
                last_word_id = -1

                for token_idx, word_id in enumerate(word_ids):
                    if word_id != last_word_id and last_word_id != -1:
                        word_text = tokenizer.decode(current_word_subwords, skip_special_tokens=True)
                        if word_text.strip():
                             aligned_predictions.append({
                                 "token": word_text,
                                 "tag": id2tag[preds[token_idx - 1]],
                                 "score": float(probs[token_idx - 1])
                             })
                        current_word_subwords = []
                    
                    if word_id is None:
                        last_word_id = -1
                        continue
                    
                    current_word_subwords.append(input_ids[token_idx])
                    last_word_id = word_id

                if current_word_subwords:
                    word_text = tokenizer.decode(current_word_subwords, skip_special_tokens=True)
                    if word_text.strip():
                        aligned_predictions.append({
                            "token": word_text,
                            "tag": id2tag[preds[len(word_ids) - 2]],
                            "score": float(probs[len(word_ids) - 2])
                        })

                result = {
                    "record_number": batch["Record Number"][j],
                    "tokens": [p["token"] for p in aligned_predictions],
                    "predicted_tags": [p["tag"] for p in aligned_predictions],
                    "confidence_scores": [p["score"] for p in aligned_predictions]
                }
                writer.write(json.dumps(result) + '\n')

    print(f"Finished. Raw predictions with scores saved to {PREDICTIONS_OUTPUT_FILE}")

if __name__ == "__main__":
    main()
