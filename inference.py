import data
import torch
from transformers import pipeline
from tqdm import tqdm
import pandas as pd

MODEL_PATH = "./ner_model_xlm_roberta_simplified"
LISTING_FILE = 'censored'

def run_inference(model_path: str, quiz_data: pd.DataFrame):
    print(f"Loading model from {model_path}")
    device = 0 if torch.cuda.is_available() else -1
    ner_pipeline = pipeline(
        "ner",
        model=model_path,
        tokenizer=model_path,
        device=device,
        aggregation_strategy="simple"
    )
    
    all_predictions = []
    for _, row in tqdm(quiz_data.iterrows(), total=quiz_data.shape[0]):
        title = row['Title']
        if not title or not isinstance(title, str):
            entities = []
        else:
            entities = ner_pipeline(title)
        
        all_predictions.append({
            "Record Number": row['Record Number'],
            "Category Id": row['Category Id'],
            "Title": title,
            "entities": entities
        })
        
    return all_predictions

if __name__ == '__main__':
    quiz_df = data.load_quiz_data(LISTING_FILE)
    predictions = run_inference(MODEL_PATH, quiz_df)
