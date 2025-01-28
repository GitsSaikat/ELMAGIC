# Import libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

# Define paths
ham_image_dir = "/kaggle/input/skin-cancer-mnist-ham10000/HAM10000_images_part_1/"
ham_label_path = "/kaggle/input/skin-cancer-mnist-ham10000/HAM10000_metadata.csv"

# Define transforms (input images are 450x600, resize to 224x224)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define HAMDataset class
class HAMDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Get image paths and labels
        for _, row in self.df.iterrows():
            img_name = row["image_id"] + ".jpg"
            img_path = os.path.join(image_dir, img_name)
            if os.path.exists(img_path):  # Check file existence upfront
                self.image_paths.append(img_path)
                self.labels.append(row["dx"])
        
        # Encode labels (7 classes: akiec, bcc, bkl, df, mel, nv, vasc)
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

# Load dataset
ham_dataset = HAMDataset(ham_label_path, ham_image_dir, transform=transform)

# Split dataset
train_size = int(0.8 * len(ham_dataset))
test_size = len(ham_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(ham_dataset, [train_size, test_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define Teacher Model (ResNet-18)
teacher = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
teacher.fc = nn.Linear(512, 7)  # 7 classes for HAM10000
for param in teacher.parameters():
    param.requires_grad = False
teacher.fc.requires_grad = True

# Define Student Model (Optimized for 224x224 input)
class LeanStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  # Input: 224x224x3
            nn.ReLU(),
            nn.MaxPool2d(2),                # 112x112x32
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                # 56x56x64
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(56*56*64, 256),
            nn.ReLU(),
            nn.Linear(256, 7)               # 7 output classes
        )
    
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

student = LeanStudent()

# Move models to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher.to(device)
student.to(device)

# Define optimizer and loss function
optimizer = optim.Adam(student.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Noise generation function (224x224 resolution)
def generate_noise(batch_size):
    return torch.randn(batch_size, 3, 224, 224).to(device) * 0.1 + 0.45

# Training loop
for epoch in range(10):
    for real_images, labels in train_loader:
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
    
    print(f"Epoch {epoch+1} | Positive Loss: {positive_loss.item():.4f} | Negative Loss: {negative_loss.item():.4f}")

# Evaluation
student.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        if images is None or labels is None:
            continue
        
        images, labels = images.to(device), labels.to(device)
        outputs = student(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")