import os
from PIL import Image
import numpy as np
import base64
import io
import pickle

# Conditional imports for PyTorch (allows loading model/training locally but running serverless without PyTorch)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
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
        def __init__(self, real_dir, ai_dir):
            self.data = []

            if real_dir:
                for img_file in os.listdir(real_dir):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.data.append((os.path.join(real_dir, img_file), 0))

            if ai_dir:
                for img_file in os.listdir(ai_dir):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.data.append((os.path.join(ai_dir, img_file), 1))

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            img_path, label = self.data[idx]
            img = Image.open(img_path).convert('RGB').resize((64, 64))
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = np.transpose(img_array, (2, 0, 1))
            return torch.tensor(img_array, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def train_model(train_dir_real, train_dir_ai, epochs=20, save_path='ai_detector_model.pt'):
        dataset = FaceDataset(train_dir_real, train_dir_ai)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

        model = AIDetectorCNN().to('cuda' if torch.cuda.is_available() else 'cpu')
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            correct = 0
            total = 0

            for images, labels in dataloader:
                images, labels = images.to('cuda' if torch.cuda.is_available() else 'cpu'), labels.to('cuda' if torch.cuda.is_available() else 'cpu')

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}, Accuracy: {100*correct/total:.2f}%')

        torch.save(model.state_dict(), save_path)
        print(f'Model saved to {save_path}')
        return model


class AIFaceDetector:
    def __init__(self, model_path='ai_detector_model.onnx'):
        self.model_path = model_path
        self.use_onnx = False

        # Try to use ONNX runtime first
        if model_path.endswith('.onnx') or not HAS_TORCH:
            if not model_path.endswith('.onnx'):
                onnx_fallback = model_path.rsplit('.', 1)[0] + '.onnx'
                if os.path.exists(onnx_fallback):
                    self.model_path = onnx_fallback
            
            try:
                import onnxruntime as ort
                if os.path.exists(self.model_path):
                    self.session = ort.InferenceSession(self.model_path)
                    print(f"Loaded ONNX model from {self.model_path}")
                    self.use_onnx = True
                else:
                    self.session = None
                    print(f"No trained ONNX model found at {self.model_path}! Using default fallback.")
            except ImportError:
                self.session = None
                print("onnxruntime is not installed. Trying PyTorch fallback.")

        # Fallback to PyTorch if ONNX is not initialized and PyTorch is available
        if not self.use_onnx:
            if HAS_TORCH:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.model = AIDetectorCNN().to(self.device)
                if os.path.exists(self.model_path):
                    self.load_model(self.model_path)
                    print(f"Loaded PyTorch model from {self.model_path}")
                else:
                    print("No trained PyTorch model found! Using default fallback.")
            else:
                print("Warning: Neither onnxruntime nor PyTorch is available!")

    def load_model(self, model_path):
        if HAS_TORCH:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            self.model.eval()

    def save_model(self, model_path):
        if HAS_TORCH:
            torch.save(self.model.state_dict(), model_path)

    def preprocess_image(self, image_data):
        if isinstance(image_data, str):
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            img_bytes = base64.b64decode(image_data)
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        elif isinstance(image_data, bytes):
            img = Image.open(io.BytesIO(image_data)).convert('RGB')
        else:
            img = image_data.convert('RGB')

        img = img.resize((64, 64))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.transpose(img_array, (2, 0, 1))
        
        if self.use_onnx:
            return np.expand_dims(img_array, axis=0)
        else:
            if HAS_TORCH:
                return torch.tensor(img_array, dtype=torch.float32).unsqueeze(0)
            return img_array

    def detect(self, image_data):
        if self.use_onnx:
            if not self.session:
                return {
                    'is_ai_generated': False,
                    'confidence': 0.5,
                    'real_prob': 0.5,
                    'ai_prob': 0.5
                }
            img_array = self.preprocess_image(image_data)
            outputs = self.session.run(None, {'input': img_array})[0]
            
            # Compute softmax on numpy array
            exp_logits = np.exp(outputs[0] - np.max(outputs[0]))
            probs = exp_logits / np.sum(exp_logits)
            
            is_ai = probs[1] > 0.5
            confidence = probs[1] if is_ai else probs[0]
            
            return {
                'is_ai_generated': bool(is_ai),
                'confidence': float(confidence),
                'real_prob': float(probs[0]),
                'ai_prob': float(probs[1])
            }
        else:
            if not HAS_TORCH or not hasattr(self, 'model'):
                return {
                    'is_ai_generated': False,
                    'confidence': 0.5,
                    'real_prob': 0.5,
                    'ai_prob': 0.5
                }
            self.model.eval()
            with torch.no_grad():
                img_tensor = self.preprocess_image(image_data).to(self.device)
                output = self.model(img_tensor)
                probs = torch.softmax(output, dim=1)
                is_ai = probs[0][1].item() > 0.5
                confidence = probs[0][1].item() if is_ai else probs[0][0].item()
                return {
                    'is_ai_generated': is_ai,
                    'confidence': confidence,
                    'real_prob': probs[0][0].item(),
                    'ai_prob': probs[0][1].item()
                }


# Initialize the default detector to load ONNX
detector = AIFaceDetector('ai_detector_model.onnx')

def detect_ai_face(image_data):
    try:
        return detector.detect(image_data)
    except Exception as e:
        print(f"AI detection error: {e}")
        return {
            'is_ai_generated': False,
            'confidence': 0.5,
            'real_prob': 0.5,
            'ai_prob': 0.5
        }


if __name__ == '__main__':
    # Test block
    detector = AIFaceDetector('ai_detector_model.onnx')
    test_result = detector.detect('path_to_test_image.jpg')
    print(test_result)