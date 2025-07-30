from datasets import load_dataset, load_from_disk, DatasetDict, Dataset 
from transformers import AutoTokenizer
from src.utils import get_tokenizer
import os
from functools import partial

def load_and_preprocess_data(model_name: str, max_length: int, tokenizer_save_path: str, tokenized_dataset_save_path:str, num_processes_for_map: int):
    """
    Load and preprocess the dataset for training or evaluation.

    Args:
        model_name (str): The name of the model to use for tokenization.
        dataset_name (str): The name of the dataset to load.
        split (str): The split of the dataset to load (default is 'train').

    Returns:
        Dataset: A preprocessed dataset ready for training or evaluation.
    """

    def tokenize_function_for_map(examples, tokenizer_path, model_name, max_length):
        # Tokenizer must be loaded INSIDE this function for num_proc > 1
        local_tokenizer = get_tokenizer(model_name=model_name, save_path=tokenizer_path)
        return local_tokenizer(
            examples['sentence'],
            truncation=True,
            padding='max_length',
            max_length=max_length
        )

    print(f"Loading SST-2 dataset...")
    sst2_dataset = load_dataset("glue", "sst2")
    
    parent_tokenizer = get_tokenizer(model_name=model_name, save_path=tokenizer_save_path)
    tokenized_dataset = None
    
    if os.path.isdir(tokenized_dataset_save_path):
        print(f"Loading tokenized dataset from: {tokenized_dataset_save_path}")
        tokenized_dataset = load_from_disk(tokenized_dataset_save_path)
    else:
        print("Tokenized dataset not found locally. Starting tokenization...")
        tokenize_with_args = partial(
            tokenize_function_for_map,
            tokenizer_path=tokenizer_save_path,
            model_name=model_name,
            max_length=max_length
        )
        tokenized_dataset = sst2_dataset.map(
            tokenize_with_args,
            batched=True,
            num_proc=num_processes_for_map, 
            remove_columns=["sentence", "idx"]
        )
        print(f"Tokenization complete. Saving to {tokenized_dataset_save_path}")
        tokenized_dataset.save_to_disk(tokenized_dataset_save_path)
        print("Tokenized dataset saved successfully.")
    
    # print("Example from original train set:", sst2_dataset['train'][0])
    # print("Example from tokenized train set:", tokenized_dataset['train'][0])
    
    return sst2_dataset, tokenized_dataset, parent_tokenizer
    
def get_subsetted_datasets(
    tokenized_ds: DatasetDict,
    train_subset_size: int,
    eval_subset_ratio: float = 0.1,
    seed: int = 42
) -> tuple[Dataset, Dataset]:
    """
    Applies subsetting logic to the tokenized training/validation datasets for SST-2.

    Args:
        tokenized_dataset_dict (DatasetDict): The full tokenized dataset (with "train" and "validation" splits).
        train_subset_size (int): The desired number of samples for the training subset.
        eval_subset_ratio (float): The ratio of train_subset_size to use for the evaluation subset.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: (train_dataset_for_trainer, eval_dataset_for_trainer)
    """
    train_dataset_for_trainer = tokenized_ds["train"]
    eval_dataset_for_trainer = tokenized_ds["validation"]

    if train_subset_size > 0 and train_subset_size < len(train_dataset_for_trainer):
        print(f"\nUsing a SUBSET for training (Train size: {train_subset_size}).")
    else:
        print("\nUsing full train dataset for training.")
        train_subset_size = len(train_dataset_for_trainer)

    eval_subset_size = int(train_subset_size * eval_subset_ratio)
    eval_subset_size = min(eval_subset_size, len(eval_dataset_for_trainer))

    if eval_subset_size == 0 and len(tokenized_ds["validation"]) > 0:
        eval_subset_size = 1

    train_dataset_for_trainer = train_dataset_for_trainer.shuffle(seed=seed).select(range(train_subset_size))
    eval_dataset_for_trainer = eval_dataset_for_trainer.shuffle(seed=seed).select(range(eval_subset_size))
    
    print(f"Final subset sizes: Train={len(train_dataset_for_trainer)}, Eval={len(eval_dataset_for_trainer)}")
    
    return train_dataset_for_trainer, eval_dataset_for_trainer
    