"""
AI Face Detection Model Training Script
Downloads dataset and trains CNN to detect AI-generated vs real faces
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os
import requests
import zipfile
import io
import base64
import hashlib
from collections import defaultdict


class AIDetectorCNN(nn.Module):
    def __init__(self):
        super(AIDetectorCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.fc_layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


class FaceDataset(Dataset):
    def __init__(self, image_dir=None, transform=True):
        self.data = []
        self.transform = transform

        if image_dir and os.path.exists(image_dir):
            real_dir = os.path.join(image_dir, 'real')
            fake_dir = os.path.join(image_dir, 'fake')

            if os.path.exists(real_dir):
                for img_file in os.listdir(real_dir):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.data.append((os.path.join(real_dir, img_file), 0))

            if os.path.exists(fake_dir):
                for img_file in os.listdir(fake_dir):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.data.append((os.path.join(fake_dir, img_file), 1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = img.resize((64, 64))
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = np.transpose(img_array, (2, 0, 1))
            return torch.tensor(img_array, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return torch.zeros(3, 64, 64), torch.tensor(0, dtype=torch.long)


class SyntheticFaceDataset(Dataset):
    """Generate synthetic face-like patterns for initial training"""
    def __init__(self, num_samples=1000, real_ratio=0.5):
        self.num_samples = num_samples
        self.real_ratio = real_ratio

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        is_real = np.random.random() < self.real_ratio
        label = 0 if is_real else 1

        if is_real:
            img = self.generate_real_face()
        else:
            img = self.generate_ai_face()

        return img, torch.tensor(label, dtype=torch.long)

    def generate_real_face(self):
        """Generate patterns similar to real photographs"""
        img = np.random.randn(64, 64, 3) * 0.3 + 0.5

        center_y, center_x = 32, 32
        y, x = np.ogrid[:64, :64]
        mask = (x - center_x)**2 + (y - center_y)**2 <= 400

        skin_tone = np.random.uniform([0.7, 0.5, 0.4], [0.95, 0.75, 0.65])
        for c in range(3):
            img[:, :, c][mask] = np.clip(img[:, :, c][mask] * skin_tone[c], 0, 1)

        noise = np.random.randn(64, 64, 3) * 0.05
        img = np.clip(img + noise, 0, 1)

        img = np.transpose(img, (2, 0, 1))
        return torch.tensor(img, dtype=torch.float32)

    def generate_ai_face(self):
        """Generate patterns typical of AI-generated images"""
        img = np.random.uniform(0, 1, (64, 64, 3))

        center_y, center_x = 32, 32
        y, x = np.ogrid[:64, :64]
        mask = (x - center_x)**2 + (y - center_y)**2 <= 400

        ai_tone = np.random.uniform([0.6, 0.45, 0.35], [0.9, 0.7, 0.6])
        for c in range(3):
            img[:, :, c][mask] = img[:, :, c][mask] * ai_tone[c]

        for _ in range(3):
            y0, x0 = np.random.randint(20, 44), np.random.randint(20, 44)
            radius = np.random.randint(3, 8)
            circle_mask = (x - x0)**2 + (y - y0)**2 <= radius**2
            img[:, :, :][circle_mask] = np.random.uniform(0.3, 0.7)

        pattern = np.sin(np.linspace(0, 4*np.pi, 64))
        img += np.random.uniform(0.02, 0.08) * np.stack([pattern]*3, axis=1)

        img = np.transpose(img, (2, 0, 1))
        return torch.tensor(img, dtype=torch.float32)


def train_model(data_dir=None, epochs=10, batch_size=32, lr=0.001, save_path='ai_detector_model.pt'):
    print("=" * 50)
    print("AI Face Detection Model Training")
    print("=" * 50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = AIDetectorCNN().to(device)

    if data_dir and os.path.exists(data_dir):
        print(f"Loading data from {data_dir}")
        dataset = FaceDataset(data_dir)
        print(f"Found {len(dataset)} images")
    else:
        print("Using synthetic data for initial training...")
        dataset = SyntheticFaceDataset(num_samples=2000, real_ratio=0.5)
        print(f"Generated {len(dataset)} synthetic samples")

    if len(dataset) == 0:
        print("No data found! Using synthetic data...")
        dataset = SyntheticFaceDataset(num_samples=2000, real_ratio=0.5)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"\nTraining for {epochs} epochs...")
    print("-" * 50)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if (batch_idx + 1) % 20 == 0:
                print(f"  Batch {batch_idx+1}/{len(dataloader)} - Loss: {loss.item():.4f}")

        accuracy = 100. * correct / total
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.2f}%")

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\n[OK] Model saved to {save_path}")

    return model


def download_sample_data():
    """Download a small sample dataset for testing"""
    print("\nNote: For full training, download the dataset from:")
    print("  - https://www.kaggle.com/datasets/shreyanshpatel1/130k-real-vs-fake-face")
    print("  - https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces")
    print("\nPlace 'real' and 'fake' folders in a 'dataset' folder")
    return None


def test_model(model_path='ai_detector_model.pt'):
    """Test the trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AIDetectorCNN().to(device)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"Loaded model from {model_path}")
    else:
        print("No trained model found! Training a new one...")
        return None

    test_images = [
        ('test_real.jpg', 0),
        ('test_fake.jpg', 1)
    ]

    print("\nTesting model...")
    model.eval()
    with torch.no_grad():
        for img_name, expected_label in test_images:
            if os.path.exists(img_name):
                img = Image.open(img_name).convert('RGB').resize((64, 64))
                img_array = np.array(img, dtype=np.float32) / 255.0
                img_array = np.transpose(img_array, (2, 0, 1))
                img_tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).to(device)

                output = model(img_tensor)
                probs = torch.softmax(output, dim=1)
                pred = probs[0][1].item() > 0.5

                print(f"  {img_name}: {'AI' if pred else 'Real'} (confidence: {probs[0][1].item():.2f})")
            else:
                print(f"  {img_name}: Test image not found")

    return model


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == 'test':
            test_model()
        elif sys.argv[1] == 'download':
            download_sample_data()
        else:
            data_dir = sys.argv[1] if len(sys.argv) > 2 else None
            epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 10
            train_model(data_dir, epochs=epochs)
    else:
        download_sample_data()
        print("\nStarting training with synthetic data...")
        train_model(epochs=10)

        print("\n" + "=" * 50)
        print("Training Complete!")
        print("=" * 50)
        print("\nTo improve accuracy with real data:")
        print("1. Download dataset from Kaggle:")
        print("   - https://www.kaggle.com/datasets/shreyanshpatel1/130k-real-vs-fake-face")
        print("   - https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces")
        print("2. Create folder 'dataset' with subfolders 'real' and 'fake'")
        print("3. Place images in dataset/real/ and dataset/fake/")
        print("4. Run: python train_model.py dataset 20")