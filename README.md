# Quantization-Aware Training (QAT) for Text Classification with Hugging Face Optimum & ONNX

## Project Overview

This project demonstrates the implementation and evaluation of Quantization-Aware Training (QAT) on a Transformer model for a text classification task (Sentiment Analysis). The goal is to optimize model size and inference speed (on both CPU and GPU) while maintaining high accuracy, leveraging the Hugging Face Optimum library and ONNX Runtime. This project aims to achieve a high-performing quantized model deployable on various devices.

## Objective

To implement and evaluate Quantization-Aware Training (QAT) on a Transformer model for a text classification task, demonstrating the trade-offs between model size, inference speed (on both CPU and GPU), and accuracy using the Hugging Face Optimum library and ONNX Runtime.

### Task & Dataset

* **Task:** Sentiment Analysis (Binary Classification) - classifying text as positive or negative.
* **Dataset:** SST-2 (Stanford Sentiment Treebank v2), available via Hugging Face `datasets` library (`load_dataset("glue", "sst2")`). This relatively small dataset allows for faster iteration and QAT retraining.

### Model

* **Model:** DistilBERT (`distilbert-base-uncased`), loaded using `AutoModelForSequenceClassification.from_pretrained()`.

## Prerequisites

To set up and run this project, you will need to have the following installed:

* **Python:** Version `3.11.9` (highly recommended to use this specific version for compatibility with project libraries).

## Environment Setup and Dependency Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/ArthurgLeonida/AI_QuantizationAndPruning.git](https://github.com/ArthurgLeonida/AI_QuantizationAndPruning.git)
    cd AI_QuantizationAndPruning
    ```

2.  **Create and Activate the Virtual Environment:**
    It is crucial to create an isolated virtual environment for this project to prevent dependency conflicts.

    * Create the virtual environment (this will create a folder named `venv` inside your project):
        ```bash
        python -m venv venv
        ```

    * Activate the virtual environment:
        * In **PowerShell**:
            ```bash
            .\venv\Scripts\Activate.ps1
            ```
        * In **Git Bash / Command Prompt (CMD)**:
            ```bash
            source venv/Scripts/activate
            ```
        *(You will know the environment is active when `(venv)` appears at the beginning of your command prompt line.)*

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Ensure you install the CUDA-enabled PyTorch version compatible with your GPU for optimal performance, as per instructions at PyTorch.org.)*

4.  **Configure `config.py`:**
    Review and adjust hyperparameters and paths in `config.py` as needed (e.g., `NUM_TRAIN_EPOCHS`, `SUBSET_SIZE`, `PRUNING_AMOUNT`).

## Usage

To run the full project pipeline (data preparation, baseline training, and optimization evaluations):

```bash
python main.py