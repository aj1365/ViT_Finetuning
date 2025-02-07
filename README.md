# Satellite Scene Classification using a Pre-trained Vision Transformer

This project fine-tune a pretrained Vision Transformer (ViT) model on the NWPU-RESISC45 dataset for scene classification using the HuggingFace Transformers library.

## Installation

1. Clone this repository:
    ```bash
    git clone <https://github.com/aj1365/ViT_Finetuning>
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the NWPU-RESISC45 dataset from Kaggle and place it in your preferred directory:
 ```bash
https://www.kaggle.com/datasets/aqibrehmanpirzada/nwpuresisc45
 ```

## Usage

Run the training script with command line arguments to specify settings:


--data_dir: Path to the NWPU-RESISC45 dataset directory.

--batch_size: Batch size for training.

--eval_batch_size: Batch size for evaluation.

--epochs: Number of training epochs.

```bash
python main.py --data_dir /path/to/dataset --batch_size 16 --eval_batch_size 16 --epochs 10
