# Comparative Analysis of Algorithms with FFA

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import time

# --- Dataset and DataLoader ---
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

# --- Student Model ---
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

# --- FFA Implementation ---
def generate_noise(batch_size, img_size=(512, 512), device='cpu'):
    """Generates noise data from Gaussian distribution."""
    return torch.randn(batch_size, 3, *img_size).to(device) * 0.1 + 0.45

def positive_pass(student, teacher, real_images, criterion, optimizer):
    """Performs the positive pass of FFA."""
    with torch.no_grad():
        teacher_logits = teacher(real_images)

    student_logits = student(real_images)
    positive_loss = criterion(student_logits, teacher_logits.softmax(dim=1))

    optimizer.zero_grad()
    positive_loss.backward()
    optimizer.step()
    return positive_loss.item()

def negative_pass(student, teacher, noise_data, criterion, optimizer):
    """Performs the negative pass of FFA."""
    with torch.no_grad():
        teacher_noise_logits = teacher(noise_data)

    student_noise_logits = student(noise_data)
    negative_loss = -criterion(student_noise_logits, teacher_noise_logits.softmax(dim=1)) # Negative CE Loss

    optimizer.zero_grad()
    negative_loss.backward()
    optimizer.step()
    return negative_loss.item()


# --- Loss Functions and Training/Evaluation ---
loss_functions = {
    "CrossEntropy": nn.CrossEntropyLoss(),
    "MSE": nn.MSELoss(),
    "MAE": nn.L1Loss(),
    "KLDiv": nn.KLDivLoss(reduction='batchmean'),
    "FFA": nn.CrossEntropyLoss() # Using CrossEntropyLoss as goodness for FFA
}

results = {}

for loss_name, criterion in loss_functions.items():
    print(f"Training with {loss_name} Loss...")
    student = LeanStudent().to(device)
    optimizer = optim.Adam(student.parameters(), lr=1e-4)
    teacher = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True).to(device).eval() # Load Teacher here for each loss to keep training consistent
    teacher.fc = nn.Linear(teacher.fc.in_features, 8)

    start_time = time.time()

    for epoch in range(10):
        student.train()
        epoch_loss = 0
        for batch_idx, (real_images, labels) in enumerate(train_loader):
            if real_images is None or labels is None:
                continue

            real_images, labels = real_images.to(device), labels.to(device)
            optimizer.zero_grad()

            if loss_name == "FFA": # Modular FFA Training Loop
                batch_size = real_images.size(0)
                noise = generate_noise(batch_size, device=device)

                positive_loss_val = positive_pass(student, teacher, real_images, criterion, optimizer)
                negative_loss_val = negative_pass(student, teacher, noise, criterion, optimizer)

                loss_val = positive_loss_val + negative_loss_val # Sum of losses for printing
                epoch_loss += loss_val


            else: # Standard Training Loop for other losses
                outputs = student(real_images)
                if loss_name == "KLDiv":
                    teacher_probs_for_kldiv = torch.softmax(teacher(real_images).detach(), dim=1) # Teacher forward pass for KLDiv
                    loss = criterion(torch.log_softmax(outputs, dim=1), teacher_probs_for_kldiv)
                elif loss_name == "MSE" or loss_name == "MAE":
                    target_one_hot = torch.nn.functional.one_hot(labels, num_classes=8).float()
                    loss = criterion(outputs, target_one_hot)
                else:
                    loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{10}, Loss: {epoch_loss/len(train_loader):.4f}")

    training_time = time.time() - start_time

    # Evaluation (same as before)
    student.eval()
    all_predicted = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            if images is None or labels is None:
                continue
            images, labels = images.to(device), labels.to(device)
            outputs = student(images)
            _, predicted = torch.max(outputs, 1)
            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    f1 = f1_score(all_labels, all_predicted, average='weighted')

    results[loss_name] = {
        "f1_score": f1,
        "runtime": training_time
    }
    print(f"{loss_name} Loss - Test F1 Score: {f1:.4f}, Training Time: {training_time:.2f} seconds\n")


print("--- Final Results for Figure 2a ---")
for loss_name, res in results.items():
    print(f"{loss_name}: F1 Score = {res['f1_score']:.4f}, Runtime = {res['runtime']:.2f} seconds")