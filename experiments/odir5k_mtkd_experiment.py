# Implementing Multi-Teacher Knowledge Distillation (MTKD) for ODIR-5K

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np

# --- Dataset and DataLoader 
odir_image_dir = "/kaggle/input/ocular-disease-recognition-odir5k/preprocessed_images/"
odir_label_path = "/kaggle/input/ocular-disease-recognition-odir5k/full_df.csv"

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class ODIRDataset(Dataset): 
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

# --- Define Teacher Models ---
# Teacher 1: ResNet18 (Stronger Teacher)
teacher1 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
teacher1.fc = nn.Linear(teacher1.fc.in_features, 8) # 8 classes
teacher1.to(device)

# Teacher 2: Smaller CNN (Weaker Teacher)
class Teacher2CNN(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), # Reduced channels
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), # Further reduced channels
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 128 * 128, 128), # Reduced Linear layer size
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

teacher2 = Teacher2CNN().to(device)


# --- Define Student Model (Lean CNN - same as before) ---
class LeanStudent(nn.Module): 
    def __init__(self, num_classes=8): # Added num_classes argument for consistency
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 128 * 128, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes) # Use num_classes here
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

student = LeanStudent().to(device)


# --- Training Function for Individual Teachers and Student 
def train_model(model, train_loader, test_loader, epochs=10, lr=1e-4, model_name="Model", mtkd_teachers=None): # mtkd_teachers for MTKD training
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print(f"--- Training {model_name} ---")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            if mtkd_teachers: # MTKD Training
                teacher1_output = mtkd_teachers[0](images).detach() # Teacher 1 output
                teacher2_output = mtkd_teachers[1](images).detach() # Teacher 2 output
                student_output = model(images)

                # Simple MTKD loss - average of CE loss against each teacher
                loss_teacher1 = criterion(student_output, teacher1_output.softmax(dim=1))
                loss_teacher2 = criterion(student_output, teacher2_output.softmax(dim=1))
                loss = (loss_teacher1 + loss_teacher2) / 2.0 # Averaged loss

            else: # Standard Training
                outputs = model(images)
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}")

    return evaluate_model(model, test_loader, model_name)


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
    auc_roc = roc_auc_score(all_labels, all_predicted_probs, multi_class='ovo', average='weighted') if len(np.unique(all_labels)) > 1 else float('nan') # Handle single class case

    print(f"\n--- {model_name} Evaluation ---")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Test F1 Score: {f1:.4f}")
    print(f"Test AUC-ROC: {auc_roc:.4f}")
    return {"Accuracy": accuracy, "F1 Score": f1, "AUC-ROC": auc_roc}


# --- Train Teacher Models Independently ---
teacher1_results = train_model(teacher1, train_loader, test_loader, model_name="Teacher1 (ResNet18)")
teacher2_results = train_model(teacher2, train_loader, test_loader, model_name="Teacher2 (Small CNN)")


# --- MTKD Training of Student Model ---
# Freeze teacher models for distillation
for param in teacher1.parameters():
    param.requires_grad = False
for param in teacher2.parameters():
    param.requires_grad = False

mtkd_student_results = train_model(student, train_loader, test_loader, model_name="Student (MTKD)", mtkd_teachers=[teacher1, teacher2])


# --- Print Results Table ---
print("\n--- MTKD Classification Report (ODIR-5K) ---")
print(f"{'Model':<10} {'Accuracy':<10} {'F1 Score':<10} {'AUC-ROC':<10}")
print(f"{'Teacher1':<10} {teacher1_results['Accuracy'] / 100:.2f}{'':<2} {teacher1_results['F1 Score']:.2f}{'':<2} {teacher1_results['AUC-ROC']:.2f}")
print(f"{'Teacher2':<10} {teacher2_results['Accuracy'] / 100:.2f}{'':<2} {teacher2_results['F1 Score']:.2f}{'':<2} {teacher2_results['AUC-ROC']:.2f}")
print(f"{'Student':<10} {mtkd_student_results['Accuracy'] / 100:.2f}{'':<2} {mtkd_student_results['F1 Score']:.2f}{'':<2} {mtkd_student_results['AUC-ROC']:.2f}")