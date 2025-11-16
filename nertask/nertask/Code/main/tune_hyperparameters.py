# manual_tune.py

from Code.utils import data, metrics
import torch
import os
import shutil
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer
)

MODEL_CHECKPOINT = "studio-ousia/luke-large"
TRAIN_FILE = 'censored'

LEARNING_RATES_TO_TEST = [1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5] 

def main():
    processed_data, tag2id, id2tag = data.load_and_process_training_data(TRAIN_FILE)
    num_labels = len(tag2id)
    
    full_dataset = Dataset.from_list(processed_data)
    split_dataset = full_dataset.train_test_split(test_size=0.1, seed=42)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    train_tokenized = split_dataset["train"].map(
        lambda x: data.tokenize_and_align_labels(x, tokenizer), batched=True
    )
    eval_tokenized = split_dataset["test"].map(
        lambda x: data.tokenize_and_align_labels(x, tokenizer), batched=True
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
    results = []

    print(f"hyperparameters search starting")
    
    for i, lr in enumerate(LEARNING_RATES_TO_TEST):
        print(f"\nTRIAL {i+1}/{len(LEARNING_RATES_TO_TEST)}: Testing Learning Rate = {lr}")
        
        trial_output_dir = f"./manual_tuning_lr_{lr}"

        model = AutoModelForTokenClassification.from_pretrained(
            MODEL_CHECKPOINT, 
            num_labels=num_labels,
            id2label=id2tag,
            label2id=tag2id
        )

        # Use the "bare-bones" TrainingArguments
        training_args = TrainingArguments(
            output_dir=trial_output_dir,
            num_train_epochs=3, 
            learning_rate=lr,   
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            weight_decay=0.01,
            logging_steps=50,
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_tokenized,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
        
        # Train and then evaluate separately
        trainer.train()
        
        predictions, labels, _ = trainer.predict(eval_tokenized)
        eval_metrics = metrics.compute_metrics((predictions, labels), id2tag)
        f02_score = eval_metrics.get("overall_f0.2", 0.0)

        print(f"result for LR = {lr}: F0.2 score = {f02_score:.4f}")
        
        results.append({"learning_rate": lr, "f0.2_score": f02_score})

        shutil.rmtree(trial_output_dir)

    best_result = max(results, key=lambda x: x["f0.2_score"])
    
    print(f"best lr found: {best_result['learning_rate']} with F0.2 Score: {best_result['f0.2_score']:.4f}")

if __name__ == "__main__":
    main()
