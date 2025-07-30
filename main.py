import multiprocessing
import os
from functools import partial
import torch
# Import modules from src directory
from src.data_preparation import load_and_preprocess_data, get_subsetted_datasets
from src.train_baseline import train_baseline_model
from src.evaluate_models import evaluate_pytorch_model

# Import configuration from config.py
from config import (
    MODEL_NAME,
    MAX_SEQUENCE_LENGTH,
    TOKENIZER_SAVE_PATH,
    TOKENIZED_DATASET_SAVE_PATH,
    NUM_PROCESSES_FOR_MAP,
    SUBSET_SIZE,
    BASELINE_OUTPUT_DIR,
    PER_DEVICE_EVAL_BATCH_SIZE,
    PER_DEVICE_TRAIN_BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_TRAIN_EPOCHS,
    FINE_TUNED_MODEL_SAVE_PATH
)

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

    # --- Verify GPU/Device ---
    if torch.cuda.is_available():
        print("CUDA (NVIDIA GPU) is available! Using GPU for acceleration.")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA (NVIDIA GPU) is NOT available. Training will run on CPU.")

    # --- Load and Prepare Data ---
    sst2_ds, tokenized_ds, parent_tokenizer = load_and_preprocess_data(
            model_name=MODEL_NAME,
            tokenizer_save_path=TOKENIZER_SAVE_PATH,
            tokenized_dataset_save_path=TOKENIZED_DATASET_SAVE_PATH,
            max_length=MAX_SEQUENCE_LENGTH,
            num_processes_for_map=NUM_PROCESSES_FOR_MAP
        )
    
    # print("\nFinal preprocessed dataset splits:", tokenized_ds.keys())
    # print("Shape of train set:", tokenized_ds['train'].shape)
    # print("Shape of validation set:", tokenized_ds['validation'].shape)
    # print("Shape of test set:", tokenized_ds['test'].shape)

    # --- Get Subsetted Datasets ---
    tok_train_ds, tok_val_ds = get_subsetted_datasets(
        tokenized_ds=tokenized_ds,
        train_subset_size=SUBSET_SIZE,
    )

    ################################## Fine-Tuned Baseline Model Training ##################################
    '''
    print("\nStarting baseline model training...")
    train_baseline_model(
        model_name=MODEL_NAME,
        output_dir=BASELINE_OUTPUT_DIR,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        train_ds=tok_train_ds,
        eval_ds=tok_val_ds,
        save_path=FINE_TUNED_MODEL_SAVE_PATH,
        tokenizer=parent_tokenizer
    )
    print("Baseline model Fine-Tuning complete!")
    '''
    print("\nStarting evaluation of the fine-tuned model...")
    evaluate_pytorch_model(
        model_path=FINE_TUNED_MODEL_SAVE_PATH,
        eval_dataset=tok_val_ds,
        batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        tokenizer=parent_tokenizer
    )
    print("Evaluation complete!")
    
    ######################################### QAT Model Training ###########################################






    