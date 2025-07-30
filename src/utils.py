from transformers import AutoTokenizer
import os
import numpy as np
import evaluate

def get_tokenizer(model_name="distilbert-base-uncased", save_path=None):
    """
    Loads and returns a pre-trained tokenizer.
    If save_path is provided and the tokenizer files exist, it loads from there.
    Otherwise, it loads from the Hugging Face Hub.
    """
    if save_path and os.path.isdir(save_path):
        print(f"Loading tokenizer from local path: {save_path}")
        return AutoTokenizer.from_pretrained(save_path)
    else:
        print(f"Loading tokenizer from Hugging Face Hub: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if save_path:
            print(f"Saving tokenizer to local path: {save_path}")
            tokenizer.save_pretrained(save_path)
        return tokenizer

def compute_metrics(eval_pred):
    """
    Computes accuracy for the evaluation predictions.
    """
    accuracy_metric = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy_metric.compute(predictions=predictions, references=labels)