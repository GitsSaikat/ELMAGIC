# Image Generation and FID Calculation 
# pip install pytorch-fid
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
import numpy as np
from pytorch_fid import fid_score # Import pytorch_fid

# --- Dataset and DataLoader ---
odir_image_dir = "/kaggle/input/ocular-disease-recognition-odir5k/preprocessed_images/"
odir_label_path = "/kaggle/input/ocular-disease-recognition-odir5k/full_df.csv"

transform = transforms.Compose([
    transforms.Resize((64, 64)), # Reduced size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

class ODIRDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None, sample_size=1000):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        count = 0
        for _, row in self.df.iterrows():
            if count >= sample_size:
                break
            if pd.notna(row["Left-Fundus"]):
                img_path = os.path.join(image_dir, row["Left-Fundus"])
                if os.path.exists(img_path):
                    self.image_paths.append(img_path)
                    self.labels.append(row["Left-Diagnostic Keywords"])
                    count += 1
            if count >= sample_size:
                break
            if pd.notna(row["Right-Fundus"]):
                img_path = os.path.join(image_dir, row["Right-Fundus"])
                if os.path.exists(img_path):
                    self.image_paths.append(img_path)
                    self.labels.append(row["Right-Diagnostic Keywords"])
                    count += 1


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


odir_dataset = ODIRDataset(odir_label_path, odir_image_dir, transform=transform, sample_size=1000) # Reduced dataset
valid_dataset = torch.utils.data.Subset(odir_dataset, range(len(odir_dataset))) # Use all valid indices
train_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)


# --- Basic Autoencoder Model ---
class BasicAutoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1), # 64x64 -> 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 16x16 -> 8x8
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8*8*128, latent_dim)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8*8*128),
            nn.ReLU(),
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # 8x8 -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # 16x16 -> 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1), # 32x32 -> 64x64
            nn.Tanh() # Output range [-1, 1]
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
autoencoder = BasicAutoencoder().to(device)
optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# --- Training Loop ---
num_epochs = 10
generated_images_dir = "generated_images_odir_fid" # Directory for generated images for FID
real_images_dir = "real_images_odir_fid" # Directory for real images for FID
os.makedirs(generated_images_dir, exist_ok=True)
os.makedirs(real_images_dir, exist_ok=True)


for epoch in range(num_epochs):
    autoencoder.train()
    epoch_loss = 0
    for batch_idx, (real_images, _) in enumerate(train_loader):
        if real_images is None:
            continue
        real_images = real_images.to(device)

        optimizer.zero_grad()
        outputs = autoencoder(real_images)
        loss = criterion(outputs, real_images) # Reconstruction loss
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.4f}")

# Generate and save images for FID calculation after training completes
print("Generating images for FID calculation...")
autoencoder.eval()
num_fid_images = 500 # Number of images for FID calculation
real_images_saved = 0
generated_images_saved = 0

with torch.no_grad():
    for batch_idx, (real_images, _) in enumerate(train_loader):
        if real_images is None:
            continue
        real_images = real_images.to(device)
        generated_images = autoencoder(real_images)

        for i in range(real_images.size(0)):
            if real_images_saved < num_fid_images:
                save_image(real_images[i], os.path.join(real_images_dir, f"real_{real_images_saved}.png"), normalize=True, range=(-1, 1))
                real_images_saved += 1
            if generated_images_saved < num_fid_images:
                save_image(generated_images[i], os.path.join(generated_images_dir, f"generated_{generated_images_saved}.png"), normalize=True, range=(-1, 1))
                generated_images_saved += 1

            if real_images_saved >= num_fid_images and generated_images_saved >= num_fid_images:
                break 
        if real_images_saved >= num_fid_images and generated_images_saved >= num_fid_images:
            break

print(f"Saved {real_images_saved} real images to '{real_images_dir}'")
print(f"Saved {generated_images_saved} generated images to '{generated_images_dir}'")


# --- FID Calculation using pytorch-fid ---
print("\n--- Calculating FID Score (Figure 2c) ---")
if real_images_saved > 0 and generated_images_saved > 0:
    try:
        fid_value = fid_score.calculate_fid_given_paths([real_images_dir, generated_images_dir], batch_size=32, device=device, dims=2048)
        print(f"FID Score: {fid_value:.4f}")
    except Exception as e:
        print(f"Error calculating FID: {e}")
        print("Please ensure you have pytorch-fid installed correctly and have a sufficient number of images generated.")
else:
    print("Not enough images generated to calculate FID. Check image saving process.")