# Evaluation of Student and Teacher Model

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

# --- Dataset and DataLoader ---
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


# --- Teacher and Student Models ---
teacher = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
teacher.fc = nn.Linear(teacher.fc.in_features, 8)

class LeanStudent(nn.Module):
    def __init__(self):
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
            nn.Linear(256, 8)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher.to(device)
student = LeanStudent().to(device)

# --- Optimizer and Loss Function ---
optimizer = optim.Adam(student.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

def generate_noise(batch_size, img_size=(512, 512)):
    return torch.randn(batch_size, 3, *img_size).to(device) * 0.1 + 0.45


# --- Training Loop ---
for epoch in range(10):
    for batch_idx, (real_images, labels) in enumerate(train_loader):
        if real_images is None or labels is None:
            continue

        real_images, labels = real_images.to(device), labels.to(device)
        batch_size = real_images.size(0)

        # Positive Pass
        with torch.no_grad():
            teacher_logits = teacher(real_images)

        student_logits = student(real_images)
        positive_loss = criterion(student_logits, teacher_logits.softmax(dim=1))

        optimizer.zero_grad()
        positive_loss.backward()
        optimizer.step()

        # Negative Pass
        noise = generate_noise(batch_size)
        with torch.no_grad():
            teacher_noise_logits = teacher(noise)

        student_noise_logits = student(noise)
        negative_loss = -criterion(student_noise_logits, teacher_noise_logits.softmax(dim=1))

        optimizer.zero_grad()
        negative_loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f"Epoch {epoch+1} [{batch_idx}/{len(train_loader)}] | Positive Loss: {positive_loss.item():.4f} | Negative Loss: {negative_loss.item():.4f}")

    print(f"Epoch {epoch+1} | Positive Loss: {positive_loss.item():.4f} | Negative Loss: {negative_loss.item():.4f}")

# --- Evaluation for Teacher and Student ---
teacher.eval()
student.eval()

teacher_correct = 0
teacher_total = 0
teacher_all_predicted_probs = []
teacher_all_labels = []

student_correct = 0
student_total = 0
student_all_predicted_probs = []
student_all_labels = []


with torch.no_grad():
    for images, labels in test_loader:
        if images is None or labels is None:
            continue
        images, labels = images.to(device), labels.to(device)

        # Teacher Evaluation
        teacher_outputs = teacher(images)
        _, teacher_predicted = torch.max(teacher_outputs, 1)
        teacher_total += labels.size(0)
        teacher_correct += (teacher_predicted == labels).sum().item()
        teacher_all_predicted_probs.extend(torch.softmax(teacher_outputs, dim=1).cpu().numpy())
        teacher_all_labels.extend(labels.cpu().numpy())

        # Student Evaluation
        student_outputs = student(images)
        _, student_predicted = torch.max(student_outputs, 1)
        student_total += labels.size(0)
        student_correct += (student_predicted == labels).sum().item()
        student_all_predicted_probs.extend(torch.softmax(student_outputs, dim=1).cpu().numpy())
        student_all_labels.extend(labels.cpu().numpy())


teacher_f1 = f1_score(teacher_all_labels, teacher_predicted.cpu().numpy(), average='weighted')
student_f1 = f1_score(student_all_labels, student_predicted.cpu().numpy(), average='weighted')

# AUC-ROC calculation
try:
    teacher_auc_roc = roc_auc_score(teacher_all_labels, teacher_all_predicted_probs, multi_class='ovo', average='weighted')
    student_auc_roc = roc_auc_score(student_all_labels, student_all_predicted_probs, multi_class='ovo', average='weighted')
except ValueError as e:
    print(f"Warning: AUC-ROC calculation failed: {e}. Setting AUC-ROC to NaN.")
    teacher_auc_roc = float('nan')
    student_auc_roc = float('nan')


print("\n--- Model Evaluation Results (Figure 2b) ---")
print(f"Teacher (ResNet18) - Test Accuracy: {100 * teacher_correct / teacher_total:.2f}%")
print(f"Teacher (ResNet18) - Test F1 Score: {teacher_f1:.4f}")
print(f"Teacher (ResNet18) - Test AUC-ROC: {teacher_auc_roc:.4f}")

print(f"\nStudent (LeanStudent) - Test Accuracy: {100 * student_correct / student_total:.2f}%")
print(f"Student (LeanStudent) - Test F1 Score: {student_f1:.4f}")
print(f"Student (LeanStudent) - Test AUC-ROC: {student_auc_roc:.4f}")