from src.utils import compute_metrics
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from torch.utils.data import DataLoader
from tqdm.auto import tqdm 
import torch
import time
import numpy as np
import os
import onnxruntime as ort

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
    print(f"Evaluation device: {device}")
    model.to(device)
    model.eval()

    preds = []
    labels = []
    inference_times = []

    # Use DataLoader for batching
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=2*batch_size, # Use double batch size for evaluation to optimize performance
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

def evaluate_onnx_model(onnx_model_path, tokenizer, eval_dataset, use_gpu=False, batch_size=16):
    '''
    Evaluate an ONNX model on the evaluation dataset.
    Args:
        onnx_model_path (str): Path to the ONNX model.
        tokenizer: Tokenizer used for preprocessing the dataset.
        eval_dataset (DatasetDict): The evaluation dataset.
        use_gpu (bool): Whether to use GPU for inference.
        batch_size (int): Batch size for evaluation.
    Returns:
        dict: Evaluation metrics.
    '''

    print("\nEvaluating ONNX model...")
    if use_gpu and 'onnxruntime-gpu' not in ort.get_available_providers():
        print("GPU support not available in ONNX Runtime. Using CPU instead.")
        use_gpu = False

    if use_gpu:
        session_options = ort.SessionOptions()
        providers = [('CUDAExecutionProvider', {'cudnn_conv_algo_search': 'DEFAULT'}), 'CPUExecutionProvider']
        session = ort.InferenceSession(onnx_model_path, sess_options=session_options, providers=providers)
    else:
        session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])

    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]

    preds = []
    labels = []
    inference_times = []

    # Use DataLoader for batching
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=2*batch_size, # Use double batch size for evaluation to optimize performance
        shuffle=False, # Evaluation should not shuffle
        collate_fn=data_collator
    )

    for batch in tqdm(eval_dataloader, desc="Evaluating ONNX Model"):
        input_ids = batch['input_ids'].numpy()
        attention_mask = batch['attention_mask'].numpy()
        batch_labels = batch['labels'].numpy()

        # Prepare inputs for ONNX model
        onnx_inputs = {
            input_names[0]: input_ids, 
            input_names[1]: attention_mask
        }

        start_time = time.time()
        onnx_outputs = session.run(output_names, onnx_inputs)
        end_time = time.time()
        inference_times.append(end_time - start_time)

        preds.append(onnx_outputs[0])
        labels.append(batch_labels)

    preds = np.concatenate(preds, axis=0)
    labels = np.concatenate(labels, axis=0)

    metrics = compute_metrics((preds, labels))
    avg_inference_time = np.mean(inference_times)

    device_str = "GPU" if use_gpu else "CPU"
    print(f"ONNX Model Accuracy on {device_str}: {metrics['accuracy']:.4f}")
    print(f"Average Inference Time per Batch on {device_str}: {avg_inference_time:.4f} seconds")

    model_size_bytes = os.path.getsize(onnx_model_path)
    model_size_mb = model_size_bytes / (1024 * 1024)
    print(f"ONNX Model Size: {model_size_mb:.2f} MB")

    return metrics, avg_inference_time, model_size_mb