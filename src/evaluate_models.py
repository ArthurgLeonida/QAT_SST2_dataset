from src.utils import compute_metrics
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm 
import time
import numpy as np
import os

def evaluate_pytorch_model(model_path, eval_dataset, batch_size=16, tokenizer=None):
    """
    Evaluate a PyTorch model on the evaluation dataset.
    
    Args:
        model_path (str): Path to the saved PyTorch model.
        eval_dataset (DatasetDict): The evaluation dataset.
        batch_size (int): Batch size for evaluation.
        tokenizer: Tokenizer used for preprocessing the dataset (needed for DataCollator).
    Returns:
        dict: Evaluation metrics.
    """

    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    preds = []
    labels = []
    inference_times = []

    # Use DataLoader for batching
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False, # Evaluation should not shuffle
        collate_fn=data_collator
    )

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            batch_labels = batch['labels'].to(device)

            start_time = time.time()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            end_time = time.time()
            inference_times.append(end_time - start_time)

            preds.append(logits.cpu().numpy())
            labels.append(batch_labels.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    labels = np.concatenate(labels, axis=0)

    metrics = compute_metrics((preds, labels))
    avg_inference_time = np.mean(inference_times)

    print(f"PyTorch Model Accuracy: {metrics['accuracy']:.4f}")
    print(f"Average Inference Time per Batch: {avg_inference_time:.4f} seconds")

    # Get model size in MB
    model_size_bytes = os.path.getsize(os.path.join(model_path, "model.safetensors"))
    model_size_mb = model_size_bytes / (1024 * 1024)
    print(f"Model Size: {model_size_mb:.2f} MB")

    return metrics, avg_inference_time, model_size_mb