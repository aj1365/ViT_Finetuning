import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import ViTImageProcessor
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

def load_and_preprocess_dataset(data_dir):
    from datasets import load_dataset
    
    # Load the dataset
    dataset = load_dataset('imagefolder', data_dir=data_dir)
    
    # Preprocess the dataset
    feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    
    def transform(examples):
        examples['pixel_values'] = [feature_extractor(image.convert("RGB"), return_tensors="pt").pixel_values[0] for image in examples['image']]
        return examples
    
    # Apply the transformation to a copy of the dataset
    transformed_dataset = dataset.map(transform, batched=True)
    
    return transformed_dataset, feature_extractor

def plot_predicted_samples(trainer, eval_dataset, num_samples=20):
    model = trainer.model
    feature_extractor = trainer.tokenizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Randomly select a subset of the evaluation dataset
    sampled_indices = np.random.choice(len(eval_dataset), num_samples, replace=False)
    
    plt.figure(figsize=(20, 10))
    for i, idx in enumerate(sampled_indices):
        idx = int(idx)  # Convert numpy integer to Python integer
        image = eval_dataset[idx]['image']
        pixel_values = feature_extractor(image.convert("RGB"), return_tensors="pt").pixel_values.to(device)
        
        with torch.no_grad():
            outputs = model(pixel_values)
            logits = outputs.logits
            predicted_label = logits.argmax(-1).item()
        
        true_label = eval_dataset[idx]['label']
        plt.subplot(4, 5, i + 1)
        plt.imshow(image)
        color = 'green' if predicted_label == true_label else 'red'
        plt.title(f"True: {eval_dataset.features['label'].int2str(true_label)}\nPred: {eval_dataset.features['label'].int2str(predicted_label)}", color=color)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Compute confusion matrix
def compute_confusion_matrix(trainer, eval_dataset):
    # Get predictions
    predictions = trainer.predict(eval_dataset)
    preds = np.argmax(predictions.predictions, axis=1)
    labels = predictions.label_ids

    # Compute confusion matrix
    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=(12, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=eval_dataset.features['label'].names,
                yticklabels=eval_dataset.features['label'].names)
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()