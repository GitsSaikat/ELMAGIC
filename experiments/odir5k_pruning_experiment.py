# Pruning Experiment for ODIR-5K

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import numpy as np
import copy  # For deepcopying models

# --- Dataset and DataLoader (same as before) ---
odir_image_dir = "/kaggle/input/ocular-disease-recognition-odir5k/preprocessed_images/"
odir_label_path = "/kaggle/input/ocular-disease-recognition-odir5k/full_df.csv"

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class ODIRDataset(Dataset): # ... (ODIRDataset class - same as before)
    def __init__(self, csv_path, image_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for _, row in self.df.iterrows():
            if pd.notna(row["Left-Fundus"]):
                self.image_paths.append(os.path.join(image_dir, row["Left-Fundus"]))
                self.labels.append(row["Left-Diagnostic Keywords"])
            if pd.notna(row["Right-Fundus"]):
                self.image_paths.append(os.path.join(image_dir, row["Right-Fundus"]))
                self.labels.append(row["Right-Diagnostic Keywords"])

        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.labels)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"File not found: {img_path}")
            return None, None

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


odir_dataset = ODIRDataset(odir_label_path, odir_image_dir, transform=transform)
valid_indices = []
for idx in range(len(odir_dataset)):
    img_path = odir_dataset.image_paths[idx]
    if os.path.exists(img_path):
        valid_indices.append(idx)
valid_dataset = torch.utils.data.Subset(odir_dataset, valid_indices)
train_size = int(0.8 * len(valid_dataset))
test_size = len(valid_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(valid_dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Pruning Functions ---
def traditional_pruning(model, prune_percent):
    """Applies traditional magnitude pruning to the model."""
    parameters_to_prune = []
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            parameters_to_prune.append((m, 'weight'))

    for module, name in parameters_to_prune:
        torch.nn.utils.prune.l1_unstructured(module, name=name, amount=prune_percent)

def iterative_magnitude_pruning(model, prune_percent_per_iteration, num_iterations, train_loader, test_loader, epochs_per_iteration=1):
    """Applies iterative magnitude pruning with fine-tuning."""
    original_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    f1_scores = []

    for iteration in range(num_iterations):
        print(f"\n--- Iteration {iteration+1} of Iterative Pruning ---")
        traditional_pruning(model, prune_percent_per_iteration) # Prune in each iteration

        # Fine-tune pruned model
        train_model_for_pruning(model, train_loader, epochs=epochs_per_iteration)
        results = evaluate_model(model, test_loader, model_name=f"Pruned Model Iteration {iteration+1}")
        f1_scores.append(results['F1 Score'])
        current_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        remaining_weight_ratio = current_params / original_params * 100
        print(f"Remaining Weight Ratio: {remaining_weight_ratio:.2f}%")

    return f1_scores


# --- Training Function for Pruning Fine-tuning ---
def train_model_for_pruning(model, train_loader, epochs=1): # Reduced epochs for fine-tuning
    optimizer = optim.Adam(model.parameters(), lr=1e-4) # Or smaller LR for fine-tuning
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"  Fine-tuning Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}")


# --- Evaluation Function (same as before) ---
def evaluate_model(model, test_loader, model_name="Model"):
    model.eval()
    correct = 0
    total = 0
    all_predicted_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_predicted_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    f1 = f1_score(all_labels, predicted.cpu().numpy(), average='weighted')

    print(f"\n--- {model_name} Evaluation ---")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Test F1 Score: {f1:.4f}")
    return {"Accuracy": accuracy, "F1 Score": f1}


# --- Experiment Parameters ---
prune_percents = np.arange(0.1, 0.6, 0.1) # Prune amounts: 10%, 20%, 30%, 40%, 50%
weight_remain_percentages = [100, 90, 80, 70, 60, 50] # Weight percentages to evaluate
num_iterative_prune_iterations = 5
prune_percent_per_iteration_iterative = 0.1 # Prune 10% weights per iteration


# --- Models to Prune ---
model_names = ["ResNet18", "VGG19"]
pruning_methods = ["Traditional Pruning", "Iterative Magnitude Pruning"]

results_dict = {} # Store results


for model_name in model_names:
    results_dict[model_name] = {}
    for prune_method_name in pruning_methods:
        results_dict[model_name][prune_method_name] = []

        print(f"\n--- Starting Experiment: {model_name} - {prune_method_name} ---")

        for weight_remain_percent in weight_remain_percentages:
            print(f"\n--- Pruning to {weight_remain_percent:.0f}% Weight Remaining ---")
            prune_percent = 1.0 - (weight_remain_percent / 100.0) # Convert weight remaining % to prune %

            # Load fresh pretrained model for each sparsity level
            if model_name == "ResNet18":
                model = models.resnet18(pretrained=True)
                model.fc = nn.Linear(model.fc.in_features, 8) # 8 classes for ODIR
            elif model_name == "VGG19":
                model = models.vgg19(pretrained=True)
                model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 8) # 8 classes for ODIR
            model.to(device)
            initial_model = copy.deepcopy(model) # Deepcopy for iterative pruning to reset each time


            if prune_method_name == "Traditional Pruning":
                model = copy.deepcopy(initial_model) # Start from fresh model for each prune level
                traditional_pruning(model, prune_percent)
                results = evaluate_model(model, test_loader, model_name=f"{model_name} - Traditional Pruning - {weight_remain_percent:.0f}%")
                results_dict[model_name][prune_method_name].append({'weight_remain': weight_remain_percent, 'f1_score': results['F1 Score']})


            elif prune_method_name == "Iterative Magnitude Pruning":
                model = copy.deepcopy(initial_model) # Start from fresh model for each prune level
                f1_scores_iterative = iterative_magnitude_pruning(
                    model, prune_percent_per_iteration_iterative, num_iterative_prune_iterations, train_loader, test_loader
                ) # Iterative pruning returns list of F1 scores for each iteration

                # Get F1 score after the last iteration (most pruned state)
                final_f1_score_iterative = f1_scores_iterative[-1] if f1_scores_iterative else np.nan # Handle empty list case

                results_dict[model_name][prune_method_name].append({'weight_remain': weight_remain_percent, 'f1_score': final_f1_score_iterative})


# --- Print Results ---
print("\n--- Pruning Experiment Results ---")
for model_name in model_names:
    for prune_method_name in pruning_methods:
        print(f"\n--- {model_name} - {prune_method_name} ---")
        for result in results_dict[model_name][prune_method_name]:
            print(f"Weight Remaining: {result['weight_remain']:.0f}%, F1 Score: {result['f1_score']:.4f}")


