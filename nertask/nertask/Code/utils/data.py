
import pandas as pd
from datasets import Dataset

def load_and_process_training_data(file_path: str):
    train_df = pd.read_csv(file_path, sep='\t', keep_default_na=False, na_values=None)
    
    all_tags = sorted(list(train_df['Tag'].unique()))
    if '' in all_tags:
        all_tags.remove('')
        
    tag2id = {tag: i for i, tag in enumerate(all_tags)}
    id2tag = {i: tag for i, tag in enumerate(all_tags)}

    processed_data = []
    for record_num, group in train_df.groupby('Record Number'):
        tokens = group['Token'].tolist()
        tags = group['Tag'].tolist()
        
        for i in range(len(tags)):
            if tags[i] == '':
                tags[i] = tags[i-1]
        
        processed_data.append({
            "tokens": tokens,
            "ner_tags": [tag2id[tag] for tag in tags]
        })
        
    print(f"loaded {len(processed_data)} training examples")
    print(f"found {len(all_tags)} unique tags: {list(tag2id.keys())}")
    
    return processed_data, tag2id, id2tag

def load_quiz_data(file_path: str) -> pd.DataFrame:
    print(f"Loading quiz data from: {file_path}")
    column_names = ['Record Number', 'Category Id', 'Title']
    listing_df = pd.read_csv(
        file_path, sep='\t', header=None, names=column_names,
        keep_default_na=False, na_values=None
    )
    listing_df['Record Number'] = pd.to_numeric(listing_df['Record Number'], errors='coerce')
    listing_df.dropna(subset=['Record Number'], inplace=True)
    listing_df['Record Number'] = listing_df['Record Number'].astype(int)
    listing_df['Category Id'] = listing_df['Category Id'].astype(int)
    quiz_df = listing_df[
        (listing_df['Record Number'] >= 5001) & (listing_df['Record Number'] <= 30000)
    ].copy()
    return quiz_df

def tokenize_and_align_labels(examples, tokenizer, label_all_tokens=True):
    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []

    for i in range(len(examples["tokens"])):
        tokens = examples["tokens"][i]
        ner_tags = examples["ner_tags"][i]

        text = ""
        token_offsets = []
        for token in tokens:
            start = len(text)
            text += token
            end = len(text)
            token_offsets.append((start, end))
            text += " "

        tokenized_inputs = tokenizer(
            text,
            truncation=True,
            return_offsets_mapping=True,
            padding=False
        )
        
        subword_labels = [-100] * len(tokenized_inputs["input_ids"])
        original_token_idx = 0
        
        for subword_idx, (sub_start, sub_end) in enumerate(tokenized_inputs["offset_mapping"]):
            if sub_start == sub_end == 0:
                continue
            
            while original_token_idx < len(token_offsets) and sub_start >= token_offsets[original_token_idx][1]:
                original_token_idx += 1
                
            if original_token_idx < len(token_offsets) and \
               sub_start >= token_offsets[original_token_idx][0] and \
               sub_end <= token_offsets[original_token_idx][1]:
                
                subword_labels[subword_idx] = ner_tags[original_token_idx]

        batch_input_ids.append(tokenized_inputs["input_ids"])
        batch_attention_mask.append(tokenized_inputs["attention_mask"])
        batch_labels.append(subword_labels)

    return {
        "input_ids": batch_input_ids,
        "attention_mask": batch_attention_mask,
        "labels": batch_labels
    }
