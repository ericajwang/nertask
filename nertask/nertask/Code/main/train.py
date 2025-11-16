# train.py

from Code.utils import data, metrics
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer
)
import os
import shutil
import torch

MODEL_CHECKPOINT = "xlm-roberta-large"
TRAIN_FILE = 'censored'
OUTPUT_DIR = "./Code/models/xlm-roberta-baseline" # A clear output directory for this model
BATCH_SIZE = 8
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 15
EARLY_STOPPING_PATIENCE = 3


def main():
    processed_data, tag2id, id2tag = data.load_and_process_training_data(TRAIN_FILE)
    num_labels = len(tag2id)
    
    dataset = Dataset.from_list(processed_data)
    tokenized_datasets = dataset.train_test_split(test_size=0.1, seed=42)

    print(f"Loading tokenizer and model for: {MODEL_CHECKPOINT}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, use_fast=True)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=num_labels,
        id2label=id2tag,
        label2id=tag2id
    )
    
    train_tokenized = tokenized_datasets["train"].map(
        lambda x: data.tokenize_and_align_labels(x, tokenizer),
        batched=True
    )
    eval_tokenized = tokenized_datasets["test"].map(
        lambda x: data.tokenize_and_align_labels(x, tokenizer),
        batched=True
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True)
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=1,
        weight_decay=WEIGHT_DECAY,
        logging_dir='./logs',
        disable_tqdm=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: metrics.compute_metrics(p, id2tag),
    )
    
    print(f"--- Starting training for {MODEL_CHECKPOINT} with MANUAL early stopping ---")
    best_f_beta_score = -1.0
    patience_counter = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{NUM_EPOCHS} ---")
        
        trainer.train()
        
        eval_metrics = trainer.evaluate()
        current_f_beta_score = eval_metrics.get("eval_overall_f0.2", 0.0)
        
        print(f"Epoch {epoch}: Validation F0.2-Score = {current_f_beta_score:.4f}, Best F0.2-Score = {best_f_beta_score:.4f}")

        if current_f_beta_score > best_f_beta_score:
            print(f"F0.2-Score improved. Saving model to {OUTPUT_DIR}")
            best_f_beta_score = current_f_beta_score
            patience_counter = 0
            trainer.save_model(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
        else:
            patience_counter += 1
            print(f"F0.2-Score did not improve. Patience counter: {patience_counter}/{EARLY_STOPPING_PATIENCE}")

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print("\nEarly stopping triggered. Training finished.")
            break

    if os.path.exists(OUTPUT_DIR):
        for item in os.listdir(OUTPUT_DIR):
            if item.startswith("checkpoint-"):
                shutil.rmtree(os.path.join(OUTPUT_DIR, item))

    print(f"\n--- Final model saved successfully in {OUTPUT_DIR}! ---")
    print(f"Best Validation F0.2-Score achieved: {best_f_beta_score:.4f}")

if __name__ == "__main__":
    main()
