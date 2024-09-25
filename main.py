import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTForImageClassification, ViTFeatureExtractor, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from PIL import Image
import matplotlib.pyplot as plt
import zipfile

# Params
img_wid = 224
img_hght = 224
batch_size = 16 #already trained in colab
epochs = 15
learning_rate = 0.01
momentum = 0.9

# model import and feature extractor
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

# unzipping data
with zipfile.ZipFile("D:/Repos/Soil_test/soil_cls/data/Train.zip") as zObject:
    zObject.extractall("D:/Repos/Soil_test/soil_cls/data")

# Load Dataset from folder structure
train_dataset = datasets.ImageFolder(root='D:/Repos/Soil_test/soil_cls/data/Train')
val_dataset = datasets.ImageFolder(root='D:/Repos/Soil_test/soil_cls/data/Valid')

# Data loader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# preprocess function
def preprocess_images(image, label):
    # Extract pixel values using the feature extractor
    pixel_values = feature_extractor(image, return_tensors="pt").pixel_values
    return {"pixel_values": pixel_values.squeeze(), "labels": label}

# Custom Dataset class to integrate with Hugging Face's Trainer API
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        processed_img = preprocess_images(img, label)
        return processed_img
    
# Convert train/val datasets to Hugging Face compatible datasets
train_dataset = CustomDataset(train_loader.dataset)
val_dataset = CustomDataset(val_loader.dataset)

# Evaluation metrics function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}

# Training arguments
training_args = TrainingArguments(
    output_dir="D:/Repos/Soil_test/soil_cls/model/base_model/vit_b_16_9class",  # Output directory
    eval_strategy="epoch",     # Evaluate every epoch
    per_device_train_batch_size=batch_size,  # Training batch size
    per_device_eval_batch_size=batch_size,   # Validation batch size
    num_train_epochs=epochs,             # Number of epochs
    save_strategy="epoch",           # Save model every epoch
    load_best_model_at_end=True,     # Load the best model when finished
    logging_dir="D:/Repos/Soil_test/soil_cls/logs",            # Log directory
)

# Trainer API to train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)


# Fine-tune the model
trainer.train()

# Save the model
model.save_pretrained('D:/Repos/Soil_test/soil_cls/model/updated_model/fine_tuned_vit_base_9_classes')

print("/n/nX========================= Main.py Compeleted =========================X")