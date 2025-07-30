# Model and Tokenizer Settings
MODEL_NAME = "distilbert-base-uncased"
MAX_SEQUENCE_LENGTH = 128

# Data Paths
TOKENIZER_SAVE_PATH = "./distilbert_tokenizer_local"
TOKENIZED_DATASET_SAVE_PATH = "./SST2_tokenized_dataset"
FINE_TUNED_MODEL_SAVE_PATH = "./fine_tuned_baseline_model"
BASELINE_OUTPUT_DIR = "./results_baseline"
QUANTIZED_QAT_MODEL_SAVE_PATH = "./QAT_model"


QUANTIZED_MODEL_SAVE_PATH = "./PTQ_model"
PRUNED_MODEL_SAVE_PATH = "./PTUP_model"

# Training Hyperparameters
NUM_TRAIN_EPOCHS = 8
PER_DEVICE_TRAIN_BATCH_SIZE = 16
PER_DEVICE_EVAL_BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
USE_FP16 = True # Set to True if your GPU supports it (my RTX 3050 does)

# Quantization and Pruning Settings
PRUNING_AMOUNT = 0.2 # Percentage of weights to prune (20%)
NUM_QAT_EPOCHS = 3 # Fewer epochs for QAT fine-tuning
QAT_LEARNING_RATE = 2e-5

# Multiprocessing
NUM_PROCESSES_FOR_MAP = 6 # Number of processes for datasets.map() (adjust based on your CPU cores)

# Subset for quick testing (set to -1 or a very large number for full dataset)
SUBSET_SIZE = -1
