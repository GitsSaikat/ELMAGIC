# Pruning Experiment for HAM10000

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
import copy

# --- Dataset and DataLoader for HAM10000 ---
ham_image_dir = "/kaggle/input/skin-cancer-mnist-ham10000/HAM10000_images_part_1/"
ham_label_path = "/kaggle/input/skin-cancer-mnist-ham10000/HAM10000_metadata.csv"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class HAMDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for _, row in self.df.iterrows():
            img_name = row["image_id"] + ".jpg"
            img_path = os.path.join(image_dir, img_name)
            if os.path.exists(img_path):
                self.image_paths.append(img_path)
                self.labels.append(row["dx"])

        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.labels)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {img_path}: {str(e)}")
            return None, None

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


ham_dataset = HAMDataset(ham_label_path, ham_image_dir, transform=transform)
train_size = int(0.8 * len(ham_dataset))
test_size = len(ham_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(ham_dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Pruning Functions 
def traditional_pruning(model, prune_percent):
    parameters_to_prune = []
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            parameters_to_prune.append((m, 'weight'))

    for module, name in parameters_to_prune:
        torch.nn.utils.prune.l1_unstructured(module, name=name, amount=prune_percent)

def iterative_magnitude_pruning(model, prune_percent_per_iteration, num_iterations, train_loader, test_loader, epochs_per_iteration=1):
    original_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    f1_scores = []

    for iteration in range(num_iterations):
        print(f"\n--- Iteration {iteration+1} of Iterative Pruning ---")
        traditional_pruning(model, prune_percent_per_iteration)

        train_model_for_pruning(model, train_loader, epochs=epochs_per_iteration)
        results = evaluate_model(model, test_loader, model_name=f"Pruned Model Iteration {iteration+1}")
        f1_scores.append(results['F1 Score'])
        current_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        remaining_weight_ratio = current_params / original_params * 100
        print(f"Remaining Weight Ratio: {remaining_weight_ratio:.2f}%")

    return f1_scores


def train_model_for_pruning(model, train_loader, epochs=1):
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
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


# --- Experiment Parameters (same as odir5k_pruning_experiment.py) ---
prune_percents = np.arange(0.1, 0.6, 0.1)
weight_remain_percentages = [100, 90, 80, 70, 60, 50]
num_iterative_prune_iterations = 5
prune_percent_per_iteration_iterative = 0.1

# --- Models to Prune (same as odir5k_pruning_experiment.py) ---
model_names = ["ResNet18", "VGG19"]
pruning_methods = ["Traditional Pruning", "Iterative Magnitude Pruning"]

results_dict = {}

for model_name in model_names:
    results_dict[model_name] = {}
    for prune_method_name in pruning_methods:
        results_dict[model_name][prune_method_name] = []

        print(f"\n--- Starting Experiment: {model_name} - {prune_method_name} ---")

        for weight_remain_percent in weight_remain_percentages:
            print(f"\n--- Pruning to {weight_remain_percent:.0f}% Weight Remaining ---")
            prune_percent = 1.0 - (weight_remain_percent / 100.0)

            if model_name == "ResNet18":
                model = models.resnet18(pretrained=True)
                model.fc = nn.Linear(model.fc.in_features, 7) # 7 classes for HAM10000
            elif model_name == "VGG19":
                model = models.vgg19(pretrained=True)
                model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 7) # 7 classes for HAM10000
            model.to(device)
            initial_model = copy.deepcopy(model)

            if prune_method_name == "Traditional Pruning":
                model = copy.deepcopy(initial_model)
                traditional_pruning(model, prune_percent)
                results = evaluate_model(model, test_loader, model_name=f"{model_name} - Traditional Pruning - {weight_remain_percent:.0f}%")
                results_dict[model_name][prune_method_name].append({'weight_remain': weight_remain_percent, 'f1_score': results['F1 Score']})

            elif prune_method_name == "Iterative Magnitude Pruning":
                model = copy.deepcopy(initial_model)
                f1_scores_iterative = iterative_magnitude_pruning(
                    model, prune_percent_per_iteration_iterative, num_iterative_prune_iterations, train_loader, test_loader
                )

                final_f1_score_iterative = f1_scores_iterative[-1] if f1_scores_iterative else np.nan

                results_dict[model_name][prune_method_name].append({'weight_remain': weight_remain_percent, 'f1_score': final_f1_score_iterative})


# --- Print Results (same as odir5k_pruning_experiment.py, but adjusted for HAM10000) ---
print("\n--- Pruning Experiment Results (HAM10000) ---")
for model_name in model_names:
    for prune_method_name in pruning_methods:
        print(f"\n--- {model_name} - {prune_method_name} ---")
        for result in results_dict[model_name][prune_method_name]:
            print(f"Weight Remaining: {result['weight_remain']:.0f}%, F1 Score: {result['f1_score']:.4f}")


