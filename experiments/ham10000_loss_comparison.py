# Comparative Analysis of Algorithms for HAM10000

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

# --- Student Model (Lean CNN) ---
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
            nn.Linear(56*56*64, 256), # Adjusted input size for 224x224 images
            nn.ReLU(),
            nn.Linear(256, 7) # 7 classes for HAM10000
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- FFA Implementation ---
def generate_noise(batch_size, device='cpu'): # Adjusted noise function for 224x224
    img_size=(224, 224)
    return torch.randn(batch_size, 3, *img_size).to(device) * 0.1 + 0.45

def positive_pass(student, teacher, real_images, criterion, optimizer):
    with torch.no_grad():
        teacher_logits = teacher(real_images)

    student_logits = student(real_images)
    positive_loss = criterion(student_logits, teacher_logits.softmax(dim=1))

    optimizer.zero_grad()
    positive_loss.backward()
    optimizer.step()
    return positive_loss.item()

def negative_pass(student, teacher, noise_data, criterion, optimizer):
    with torch.no_grad():
        teacher_noise_logits = teacher(noise_data)

    student_noise_logits = student(noise_data)
    negative_loss = -criterion(student_noise_logits, teacher_noise_logits.softmax(dim=1))

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
    "FFA": nn.CrossEntropyLoss()
}

results = {}

for loss_name, criterion in loss_functions.items():
    print(f"Training with {loss_name} Loss...")
    student = LeanStudent().to(device)
    optimizer = optim.Adam(student.parameters(), lr=1e-4)
    teacher = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True).to(device).eval()
    teacher.fc = nn.Linear(teacher.fc.in_features, 7) # 7 classes for HAM10000

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

                loss_val = positive_loss_val + negative_loss_val
                epoch_loss += loss_val

            else: # Standard Training Loop
                outputs = student(real_images)
                if loss_name == "KLDiv":
                    teacher_probs_for_kldiv = torch.softmax(teacher(real_images).detach(), dim=1)
                    loss = criterion(torch.log_softmax(outputs, dim=1), teacher_probs_for_kldiv)
                elif loss_name == "MSE" or loss_name == "MAE":
                    target_one_hot = torch.nn.functional.one_hot(labels, num_classes=7).float() # 7 classes for HAM10000
                    loss = criterion(outputs, target_one_hot)
                else:
                    loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{10}, Loss: {epoch_loss/len(train_loader):.4f}")

    training_time = time.time() - start_time

    # Evaluation
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

# --- Print Results for Figure 2a ---
print("--- Final Results for Figure 2a (HAM10000) ---")
for loss_name, res in results.items():
    print(f"{loss_name}: F1 Score = {res['f1_score']:.4f}, Runtime = {res['runtime']:.2f} seconds")