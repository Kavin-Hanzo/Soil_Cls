import os

# Define the directory structure for the project
PROJECT_NAME = "soil_tester_project"

structure = {
    f"{PROJECT_NAME}/": [
        "app.py",            # Flask application
        "main.py",           # Main script for training and evaluation
        "setup.py",          # For packaging the project
        "requirements.txt",  # Python dependencies
        "data_ingestion.py", # Data ingestion module
        "data_preprocessing.py", # Data preprocessing module
        "model_preparation.py",  # Model preparation module
        "train_eval.py",     # Training and evaluation functions
        "visualization.py",  # Visualization functions
    ],
    f"{PROJECT_NAME}/data/": [],   # Directory for dataset images
    f"{PROJECT_NAME}/templates/": [
        "index.html",        # HTML template for Flask app
    ],
    f"{PROJECT_NAME}/results/": [], # Directory for storing results
}

# Boilerplate content for the main files
file_content = {
    "app.py": """from flask import Flask, request, jsonify, render_template
from PIL import Image
import torch
from data_preprocessing import get_transform
from model_preparation import load_pretrained_model

app = Flask(__name__)

MODEL_NAME = 'google/vit-base-patch16-224'
model, feature_extractor = load_pretrained_model(MODEL_NAME)
model.eval()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    if file:
        try:
            img = Image.open(file)
            transform = get_transform()
            img = transform(img).unsqueeze(0)
            
            with torch.no_grad():
                outputs = model(img)
                logits = outputs.logits
                predicted_class_idx = logits.argmax(-1).item()
            
            result = {"predicted_class": predicted_class_idx}
            return jsonify(result)
        
        except Exception as e:
            return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
    """,
    
    "index.html": """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Soil Tester</title>
</head>
<body>
    <h1>Upload a Soil Image</h1>
    <form action="/predict" method="POST" enctype="multipart/form-data">
        <input type="file" name="file">
        <button type="submit">Predict</button>
    </form>
</body>
</html>
    """,

    "setup.py": """from setuptools import setup, find_packages

setup(
    name="soil_tester_project",
    version="1.0",
    description="A soil image classifier using Vision Transformers (ViT)",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        'Flask',
        'torch',
        'torchvision',
        'transformers',
        'scikit-learn',
        'matplotlib',
        'pillow',
    ],
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'train_soil_model=main:main',
        ],
    },
    python_requires='>=3.6',
)
    """,
    
    "main.py": """import os
import torch
from transformers import TrainingArguments, Trainer
from data_ingestion import load_images
from data_preprocessing import get_transform
from model_preparation import load_pretrained_model
from train_eval import train_model, evaluate_model, compute_metrics
from visualization import plot_training_curve
from torch.utils.data import DataLoader

DATA_DIR = './data/Soil_types'
MODEL_NAME = 'google/vit-base-patch16-224'
BATCH_SIZE = 8
EPOCHS = 5
OUTPUT_DIR = './results'

def main():
    transform = get_transform()
    dataset = load_images(DATA_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model, feature_extractor = load_pretrained_model(MODEL_NAME)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        save_steps=10_000,
        save_total_limit=2,
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
        compute_metrics=compute_metrics
    )

    train_model(trainer)
    eval_metrics = evaluate_model(trainer)
    print(f"Evaluation Metrics: {eval_metrics}")

    if 'loss' in eval_metrics:
        plot_training_curve([trainer.state.log_history[i]['loss'] for i in range(EPOCHS)], [eval_metrics['eval_loss']])

if __name__ == "__main__":
    main()
    """
}

# Function to create directories and files
def create_project_structure(structure, file_content):
    for folder, files in structure.items():
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created directory: {folder}")

        for file in files:
            file_path = os.path.join(folder, file)
            with open(file_path, 'w') as f:
                content = file_content.get(file, "")
                f.write(content)
                print(f"Created file: {file_path}")

# Create the project structure
create_project_structure(structure, file_content)

print("Project structure created successfully.")
