### This code is written by Ali Jamali (ali.jamali.65@gmail.com)

import argparse
import torch
from transformers import ViTForImageClassification, TrainingArguments, Trainer
from utils import load_and_preprocess_dataset, plot_predicted_samples

def main(args):
    # Load and preprocess the dataset
    dataset, feature_extractor = load_and_preprocess_dataset(args.data_dir)
    
    # Define the model
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224-in21k',
        num_labels=len(dataset['train'].features['label'].names)
    )

    # Move model to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        logging_dir='./logs',
    )

    # Define the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        tokenizer=feature_extractor,
    )

    # Function to train the model
    def train_model(trainer):
        print("Starting training...")
        trainer.train()
        print("Training completed!")

    # Call the training function
    train_model(trainer)

    # Optional: Evaluate the model
    trainer.evaluate()

    # Plot some predictions
    plot_predicted_samples(trainer, dataset['test'], num_samples=20)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Vision Transformer on the NWPU-RESISC45 dataset")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the NWPU-RESISC45 dataset directory")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")

    args = parser.parse_args()
    main(args)
