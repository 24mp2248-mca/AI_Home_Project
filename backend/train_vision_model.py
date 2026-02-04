import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os
import glob
from dl_model import UNet

# Configuration
BATCH_SIZE = 4
EPOCHS = 5
LR = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = "backend/data/train"
MODEL_PATH = "backend/wall_segmentation_model.pth"

class FloorPlanDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        # Synthetic Data
        self.images = sorted(glob.glob(os.path.join(root_dir, "images", "*.png")))
        self.masks = sorted(glob.glob(os.path.join(root_dir, "masks", "*.png")))
        
        # Real Data (Option A Upgrade)
        real_dir = "backend/data/real"
        if os.path.exists(real_dir):
            real_images = sorted(glob.glob(os.path.join(real_dir, "images", "*.*")))
            real_masks = sorted(glob.glob(os.path.join(real_dir, "masks", "*.*")))
            
            if len(real_images) > 0 and len(real_images) == len(real_masks):
                print(f"Found {len(real_images)} REAL training samples! Mixing them in.")
                self.images.extend(real_images)
                self.masks.extend(real_masks)
            else:
                 pass # No real data or mismatch, ignore
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load Image
        img_path = self.images[idx]
        mask_path = self.masks[idx]
        
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0
        
        # Add channel dim: (H, W) -> (C, H, W)
        img = np.expand_dims(img, axis=0)
        mask = np.expand_dims(mask, axis=0)
        
        return torch.from_numpy(img), torch.from_numpy(mask)

def train():
    print(f"Using device: {DEVICE}")
    
    # 1. Prepare Data
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory {DATA_DIR} not found. Run generate_synthetic_data.py first.")
        return

    dataset = FloorPlanDataset(DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Loaded {len(dataset)} training samples.")

    # 2. Init Model
    model = UNet(n_channels=1, n_classes=1).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss() # Binary Cross Entropy for mask

    # 3. Training Loop
    model.train()
    print("Starting Training...")
    
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for images, masks in dataloader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {epoch_loss/len(dataloader):.4f}")

    # 4. Save
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
    print("Vision AI Activation Complete!")

if __name__ == "__main__":
    train()
