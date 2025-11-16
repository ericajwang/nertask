
import pandas as pd
import json

PREDICTIONS_INPUT_FILE = "censored"
ORIGINAL_TRAIN_FILE = 'censored'
AUGMENTED_TRAIN_FILE = "censored"
CONFIDENCE_THRESHOLD = 0.99

def main():    
    new_records = []
    with open(PREDICTIONS_INPUT_FILE, 'r') as f:
        for line in f:
            data = json.loads(line)
            record_num = data["record_number"]
            
            for i in range(len(data["tokens"])):
                token = data["tokens"][i]
                score = data["confidence_scores"][i]
                predicted_tag = data["predicted_tags"][i]
                
                # apply the confidence filter
                # if confidence is high enough, use the predicted tag. otherwise, label as 'O'.
                final_tag = predicted_tag if score >= CONFIDENCE_THRESHOLD else 'O'
                
                new_records.append([record_num, 1, token, final_tag])

    pseudo_df = pd.DataFrame(new_records, columns=['Record Number', 'Category Id', 'Token', 'Tag'])
    print(f"created {len(pseudo_df)} pseudo-labeled token records")
    
    original_df = pd.read_csv(ORIGINAL_TRAIN_FILE, sep='\t')
    original_df = original_df[['Record Number', 'Category', 'Token', 'Tag']]
    augmented_df = pd.concat([original_df, pseudo_df], ignore_index=True)
    augmented_df.to_csv(AUGMENTED_TRAIN_FILE, sep='\t', index=False)

if __name__ == "__main__":
    main()
