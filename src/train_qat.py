from src.utils import compute_metrics
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import DatasetDict
from optimum.onnxruntime.configuration import ORTConfig, AutoQuantizationConfig
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTTrainer, ORTQuantizer, ORTTrainingArguments
import torch

def train_qat_model(
    baseline_model_path="./fine_tuned_baseline_model",
    output_dir="./results_qat",
    num_train_epochs_qat=5, # Lower than baseline
    learning_rate_qat=5e-6, # Lower than baseline
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    train_ds: DatasetDict = None,
    eval_ds: DatasetDict = None,
    tokenizer = None,
    save_path="./qat_model"
):
    '''
    Train a Quantization-Aware Training (QAT) model based on a fine-tuned baseline model.
    '''
    print("\nLoading baseline model for QAT...")
    model_for_qat = AutoModelForSequenceClassification.from_pretrained(baseline_model_path)

    ort_config = ORTConfig(
        quantization=AutoQuantizationConfig.for_sequence_classification(
            is_static=True, 
            format="QDQ", # (Quantize-Dequantize)
            weight_type="int8",
        )
    )

    qat_training_args = ORTTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs_qat,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate_qat,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        report_to="tensorboard",
        save_onnx_model=True,
        save_onnx_model_path=save_path,
        feature='sequence-classification'
    )

    qat_trainer = ORTTrainer(
        model=model_for_qat,
        args=qat_training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        feature='sequence-classification',
        ort_config=ort_config
    )

    print("\nStarting QAT training...")
    qat_trainer.train()