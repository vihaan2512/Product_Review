import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments
)

logger.add("logs/transformer_training.log", rotation="10 MB")

class ReviewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
    acc = accuracy_score(labels, predictions)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

def main():
    dataset_path = Path("data/raw/amazon_fake/deceptive_reviews.csv")
    model_save_dir = Path("models/fake_review_transformer")
    
    if not dataset_path.exists():
        logger.error(f"Dataset not found at {dataset_path}. Run extract_amazon_reviews.py first.")
        return
        
    logger.info(f"Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)
    
    df = df.rename(columns={"clean_text": "text", "true_label": "label"})
    if df["label"].dtype == object:
        df["label"] = df["label"].str.lower().map({"truthful": 0, "deceptive": 1, "real": 0, "fake": 1}).fillna(0)
    df["label"] = df["label"].astype(int)

    df = df.dropna(subset=["text"]).reset_index(drop=True)
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["text"].tolist(), 
        df["label"].tolist(), 
        test_size=0.2, 
        random_state=42, 
        stratify=df["label"].tolist()
    )
    
    logger.info(f"Dataset split: {len(train_texts)} train, {len(val_texts)} validation samples.")

    model_name = "distilbert-base-uncased"
    logger.info(f"Loading pre-trained tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    logger.info("Tokenizing datasets...")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)
    
    train_dataset = ReviewsDataset(train_encodings, train_labels)
    val_dataset = ReviewsDataset(val_encodings, val_labels)

    logger.info(f"Loading pre-trained model: {model_name}...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training will run on device: {device.upper()}")
    model = model.to(device)
    
    training_args = TrainingArguments(
        output_dir="logs/transformer_runs",
        num_train_epochs=2,
        per_device_train_batch_size=16 if device == "cuda" else 8,
        per_device_eval_batch_size=16 if device == "cuda" else 8,
        warmup_steps=200,
        weight_decay=0.01,
        logging_dir="logs/transformer_runs_logs",
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=(device == "cuda"),
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    
    logger.info("Starting Transformer fine-tuning...")
    trainer.train()
    
    logger.info("Evaluating model on validation split...")
    eval_results = trainer.evaluate()
    
    metrics_summary = (
        f"\nTransformer Training Complete:\n"
        f"  Validation Accuracy:  {eval_results.get('eval_accuracy', 0):.4f}\n"
        f"  Validation F1-Score:  {eval_results.get('eval_f1', 0):.4f}\n"
        f"  Validation Precision: {eval_results.get('eval_precision', 0):.4f}\n"
        f"  Validation Recall:    {eval_results.get('eval_recall', 0):.4f}"
    )
    
    # Print directly to stdout and log
    print(metrics_summary)
    logger.success(metrics_summary)
    
    save_msg = f"Saving best model to {model_save_dir}..."
    print(save_msg)
    logger.info(save_msg)
    
    model_save_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(model_save_dir))
    tokenizer.save_pretrained(str(model_save_dir))
    
    done_msg = f"Best model state successfully saved to: {model_save_dir}"
    print(done_msg)
    logger.success(done_msg)

if __name__ == "__main__":
    main()