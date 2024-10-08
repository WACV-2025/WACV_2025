from time import process_time_ns
import torch
import clip
from PIL import Image
import os
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, average_precision_score, f1_score
import torch.nn.functional as F

import torch.nn as nn
from torch.utils.data import random_split
from torchvision import transforms
import sys
import argparse
import time
from tqdm import tqdm
from sklearn import metrics
from sklearn.metrics import roc_curve
from blipmodels import blip_decoder



class NeuralNet(nn.Module):
    """
    A neural network for classification with three fully connected layers and dropout.
    
    Attributes:
        dropout2 (nn.Dropout): Dropout layer with a probability of 0.5.
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
        fc3 (nn.Linear): Third fully connected layer (output layer).
    """
    
    def __init__(self, input_size, hidden_size_list, num_classes):
        """
        Initialize the NeuralNet model with given input size, hidden sizes, and number of classes.
        
        Args:
            input_size (int): The number of input features.
            hidden_size_list (list of int): List containing sizes of hidden layers.
            num_classes (int): The number of output classes.
        """
        super(NeuralNet, self).__init__()
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(input_size, hidden_size_list[0])
        self.fc2 = nn.Linear(hidden_size_list[0], hidden_size_list[1])
        self.fc3 = nn.Linear(hidden_size_list[1], num_classes)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (Tensor): Input tensor.
        
        Returns:
            Tensor: Output predictions.
        """
        out = self.fc1(x)
        out = F.relu(out)
        out = self.dropout2(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        return out

def preprocess_image(img_path, image_size=224):
    """
    Preprocess an image for model input.
    
    Args:
        img_path (str): Path to the image file.
        image_size (int): Desired size of the image.
        
    Returns:
        Tensor: Preprocessed image tensor.
    """
    img = Image.open(img_path).convert('RGB')
    img = img.resize((image_size, image_size))
    return preprocess(img)

def calculate_metrics(true_labels, predictions):
    """
    Calculate various metrics for the given true labels and predictions.
    
    Args:
        true_labels (list of int): List of true labels.
        predictions (list of int): List of predicted labels.
        
    Returns:
        dict: Dictionary containing confusion matrix, accuracy, precision, and recall.
    """
    metrics_dict = {}
    if len(true_labels) == 0 or len(predictions) == 0:
        return {"confusion_matrix": [], "accuracy": np.nan, "precision": np.nan, "recall": np.nan , "f1_score":np.nan}
    
    conf_matrix = confusion_matrix(true_labels, predictions)
    metrics_dict['confusion_matrix'] = conf_matrix.tolist()
    metrics_dict['accuracy'] = accuracy_score(true_labels, predictions)
    metrics_dict['precision'] = precision_score(true_labels, predictions)
    metrics_dict['recall'] = recall_score(true_labels, predictions)
    metrics_dict['f1_score'] = f1_score(true_labels, predictions)
    return metrics_dict

# Argument parsing
parser = argparse.ArgumentParser(description='Finetune the classifier to wash the backdoor')
parser.add_argument('--image_folder', default='images/', type=str, help='Path to the folder containing images.')
parser.add_argument('--gpu', default='0', type=str, help='GPU device index.')
parser.add_argument('--metric_dir', default="real", type=str, help='Directory to save metrics for real or fake images.')
args = parser.parse_args()

# Set device and load models
device = "cuda" if torch.cuda.is_available() else "cpu"
model2, preprocess = clip.load("ViT-B/32")

image_size = 224

blip_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'
blip = blip_decoder(pretrained=blip_url, image_size=image_size, vit='base')
blip.eval()
blip = blip.to(device)

model = torch.load("finetune_clip.pt").to(device)
linear = NeuralNet(1024, [512, 256], 2).to(device)
linear = torch.load('clip_linear.pt')

true_labels_real = []
all_predictions_real = []

true_labels_fake = []
all_predictions_fake = []

# Ensure the output directory exists
real_metric_path = "metrics_real"
fake_metric_path = "metrics_fake"
os.makedirs(real_metric_path, exist_ok=True)
os.makedirs(fake_metric_path, exist_ok=True)

# Process images
for img_name in tqdm(os.listdir(args.image_folder)):
    img_path = os.path.join(args.image_folder, img_name)
    img = Image.open(img_path).convert('RGB')
    tform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    img = tform(img)
    img = img.unsqueeze(0).to(device)

    caption = blip.generate(img, sample=False, num_beams=1, max_length=60, min_length=5) 
    text = clip.tokenize(list(caption)).to(device)

    image = preprocess_image(img_path, image_size).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        emb = torch.cat((image_features, text_features), 1)
        output = linear(emb.float())
        predict = output.argmax(1).cpu().numpy()[0]
        
        # Assign true label based on the directory
        if "real" in img_path:
            true_label = 0
        elif "fake" in img_path:
            true_label = 1
        else:
            # Skip images that don't fit the criteria
            continue
        
        # Collect metrics separately for real and fake images
        if args.metric_dir == "real":
            true_labels_real.append(true_label)
            all_predictions_real.append(predict)
        elif args.metric_dir == "fake":
            true_labels_fake.append(true_label)
            all_predictions_fake.append(predict)
        
        # Save metrics for individual images
        image_metrics = calculate_metrics([true_label], [predict])
        output_dir = real_metric_path if args.metric_dir == "real" else fake_metric_path
        metrics_file = os.path.join(output_dir, f"{img_name}.json")
        with open(metrics_file, 'w') as f:
            json.dump(image_metrics, f)
        
        print(f"Image: {img_name}, Prediction: {predict}")

def print_metrics(title, metrics):
    """
    Print metrics with a title and formatted output.
    """
    print(f"\n{'='*40}")
    print(f"{title}")
    print(f"{'='*40}")
    print("Confusion Matrix:\n", metrics['confusion_matrix'])
    print("Accuracy:", metrics['accuracy'])
    print("Precision:", metrics['precision'])
    print("Recall:", metrics['recall'])
    print("F1 Score:", metrics['f1_score'])
    print(f"{'='*40}")

# Calculate final metrics for real images
metrics_real = calculate_metrics(true_labels_real, all_predictions_real)
print_metrics("Metrics for Real Images", metrics_real)

# Calculate final metrics for fake images
metrics_fake = calculate_metrics(true_labels_fake, all_predictions_fake)
print_metrics("Metrics for Fake Images", metrics_fake)

# Combine true labels and predictions for both real and fake images
true_labels_combined = true_labels_real + true_labels_fake
all_predictions_combined = all_predictions_real + all_predictions_fake

# Calculate combined metrics
metrics_combined = calculate_metrics(true_labels_combined, all_predictions_combined)
avg_precision_combined = average_precision_score(true_labels_combined, all_predictions_combined, average='weighted')

print(f"\n{'='*40}")
print("Combined Metrics for Real and Fake Images")
print(f"{'='*40}")
print("Confusion Matrix:\n", metrics_combined['confusion_matrix'])
print("Accuracy:", metrics_combined['accuracy'])
print("Precision:", metrics_combined['precision'])
print("Recall:", metrics_combined['recall'])
print("F1 Score:", metrics_combined['f1_score'])
print("Average Precision:", avg_precision_combined)
print(f"{'='*40}")

# Save final metrics to JSON file
final_metrics = {
    "confusion_matrix": metrics_combined['confusion_matrix'],
    "accuracy": metrics_combined['accuracy'],
    "precision": metrics_combined['precision'],
    "recall": metrics_combined['recall'],
    "f1_score": metrics_combined['f1_score'],
    "avg_precision_combined": avg_precision_combined
}
with open("final_metrics.json", "w") as f:
    json.dump(final_metrics, f, indent=4)
