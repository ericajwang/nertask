
import pandas as pd
import torch
from transformers import pipeline
from tqdm import tqdm
import csv


model = 'xlm-roberta-baseline'
MODEL_PATH = f"censored"
MODEL_PATH = f"censored"

LISTING_FILE = 'censored'
SUBMISSION_FILE = f'censored'


ASPECT_CATEGORY_RULES = {
    'Anwendung': [2],
    'Anzahl_Der_Einheiten': [1, 2],
    'Besonderheiten': [1, 2],
    'Breite': [2],
    'Bremsscheiben-Aussendurchmesser': [1],
    'Bremsscheibenart': [1],
    'Einbauposition': [1, 2],
    'Farbe': [1],
    'Größe': [1, 2],
    'Hersteller': [1, 2],
    'Herstellernummer': [1, 2],
    'Herstellungsland_Und_-Region': [1],
    'Im_Lieferumfang_Enthalten': [1, 2],
    'Kompatible_Fahrzeug_Marke': [1, 2],
    'Kompatibles_Fahrzeug_Jahr': [1, 2],
    'Kompatibles_Fahrzeug_Modell': [1, 2],
    'Länge': [2],
    'Material': [1],
    'Maßeinheit': [1, 2],
    'Menge': [2],
    'Modell': [1, 2],
    'Oberflächenbeschaffenheit': [1],
    'Oe/Oem_Referenznummer(N)': [1, 2],
    'Produktart': [1, 2],
    'Produktlinie': [1],
    'SAE_Viskosität': [2],
    'Stärke': [1],
    'Technologie': [1],
    'Zähnezahl': [2]
}

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

def run_inference(model_path: str, quiz_data: pd.DataFrame):
    print(f"Loading model from {model_path}")
    device = 0 if torch.cuda.is_available() else -1
    ner_pipeline = pipeline(
        "ner", model=model_path, tokenizer=model_path,
        device=device, aggregation_strategy="simple"
    )
    all_predictions = []
    for _, row in tqdm(quiz_data.iterrows(), total=quiz_data.shape[0]):
        title = row['Title']
        entities = ner_pipeline(title) if title and isinstance(title, str) else []
        all_predictions.append({
            "Record Number": row['Record Number'],
            "Category Id": row['Category Id'],
            "Title": title,
            "entities": entities
        })
    return all_predictions

def create_submission_file(predictions: list, output_file: str):
    submission_records = []
    for record in predictions:
        record_num = record['Record Number']
        cat_id = record['Category Id']
        for entity in record['entities']:
            if entity['entity_group'] == 'O':
                continue
            
            aspect_name = entity['entity_group']
            aspect_value = entity['word']
            
            if aspect_name == "Aussendurchmesser":
                aspect_name = "Bremsscheiben-Aussendurchmesser"
            
            allowed_categories = ASPECT_CATEGORY_RULES.get(aspect_name)
            if allowed_categories and cat_id not in allowed_categories:
                continue

            sanitized_aspect_value = aspect_value.replace('\t', ' ')
            
            if not sanitized_aspect_value.strip():
                continue
            
            submission_records.append({
                "Record Number": record_num,
                "Category Id": cat_id,
                "Aspect Name": aspect_name,
                "Aspect Value": sanitized_aspect_value
            })
            
    submission_df = pd.DataFrame(submission_records)
    submission_df.to_csv(
        output_file, sep='\t', index=False, header=True,
        encoding='utf-8', quoting=csv.QUOTE_NONE
    )

if __name__ == '__main__':
    
    quiz_df = load_quiz_data(LISTING_FILE)
    predictions = run_inference(MODEL_PATH, quiz_df)
    create_submission_file(predictions, SUBMISSION_FILE)

    print("finished successfully")
