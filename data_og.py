import pandas as pd
from datasets import Dataset

def load_and_process_training_data(file_path: str):
    """
    Loads and processes the training data from the TSV file.
    """
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
        
    print(f"Loaded {len(processed_data)} training examples.")
    print(f"Found {len(all_tags)} unique tags: {list(tag2id.keys())}")
    
    return processed_data, tag2id, id2tag

def load_quiz_data(file_path: str) -> pd.DataFrame:
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
    print(f"loaded and cleaned {len(quiz_df)} records for the quiz set")
    return quiz_df

def tokenize_and_align_labels(examples, tokenizer, label_all_tokens=True):
    is_luke_tokenizer = "LukeTokenizer" in tokenizer.__class__.__name__

    if is_luke_tokenizer:
        tokenized_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
        for i, (tokens, ner_tags) in enumerate(zip(examples["tokens"], examples["ner_tags"])):
            tokenized_example = tokenizer(tokens, truncation=True, is_split_into_words=True)
            word_ids = tokenized_example.word_ids()
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(ner_tags[word_idx])
                else:
                    label_ids.append(ner_tags[word_idx] if label_all_tokens else -100)
                previous_word_idx = word_idx
            
            tokenized_inputs["input_ids"].append(tokenized_example["input_ids"])
            tokenized_inputs["attention_mask"].append(tokenized_example["attention_mask"])
            tokenized_inputs["labels"].append(label_ids)
    else:
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(label[word_idx] if label_all_tokens else -100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels

    return tokenized_inputs
