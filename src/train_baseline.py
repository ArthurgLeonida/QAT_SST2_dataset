from src.utils import compute_metrics
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import DatasetDict
import torch

def train_baseline_model(
    model_name="distilbert-base-uncased",
    output_dir="./results_baseline",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    train_ds: DatasetDict = None,
    eval_ds: DatasetDict = None,
    save_path="./fine_tuned_baseline_model",
    tokenizer=None
):
    """
    Train a baseline model using the preprocessed SST-2 dataset.
    Args:
        model_checkpoint (str): The model checkpoint to use for training.
        output_dir (str): Directory to save the trained model and results.
        num_train_epochs (int): Number of training epochs.
        per_device_train_batch_size (int): Batch size for training.
        per_device_eval_batch_size (int): Batch size for evaluation.
        learning_rate (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay for the optimizer.
        train_ds (DatasetDict): The training dataset.
        eval_ds (DatasetDict): The evaluation dataset.
        save_path (str): Path to save the fine-tuned model.
        tokenizer: Tokenizer used for preprocessing the dataset.
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        report_to="tensorboard"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    print(f"\nSaving fine-tuned baseline model to: {save_path}")
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print("Fine-tuned baseline model and tokenizer saved.")

