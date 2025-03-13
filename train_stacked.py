import os
import numpy as np
from PIL import Image
import cv2
import torch
import matplotlib.cm as cm
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries, slic

# Fix 1: Image Transformation Consistency
class SummedImageDataset(Dataset):
    def __init__(self, scores, transform=None):
        self.scores = scores
        self.transform = transform if transform else transforms.ToTensor()

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        tt_path = os.path.join('polar_tt_predict_C_raw', f'{idx}_predict.png')
        vp_path = os.path.join('polar_vp_predict_C_raw', f'{idx}_predict.png')
        
        if not os.path.exists(tt_path) or not os.path.exists(vp_path):
            raise FileNotFoundError(f"Missing image: {tt_path} or {vp_path}")
        
        tt_image = cv2.imread(tt_path, cv2.IMREAD_GRAYSCALE)
        vp_image = cv2.imread(vp_path, cv2.IMREAD_GRAYSCALE)
        
        if tt_image is None or vp_image is None:
            raise ValueError(f"Error loading images: {tt_path}, {vp_path}")
        
        summed_array = (tt_image.astype(np.float32) + vp_image.astype(np.float32)) / (255 * 2)
        summed_image = Image.fromarray((summed_array * 255).astype(np.uint8))
        summed_image = self.transform(summed_image)
        
        if summed_image.shape[0] == 1:
            summed_image = summed_image.repeat(3, 1, 1)  # Convert to 3-channel
        
        label = torch.tensor(self.scores[idx], dtype=torch.float32)
        return summed_image, label

# Load scores
scores = np.load("sf.npy")
fused, tt, vp = scores[0], scores[1], scores[2]
scores = fused - np.maximum(tt, vp)

# Fix 2: Ensure Dataset Iterates Through All Indices
dataset = SummedImageDataset(scores, transform=transforms.ToTensor())
assert len(dataset) == 956, f"Dataset length mismatch: expected 956, got {len(dataset)}"
train_size = len(dataset) - 200
test_size = 200
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define Model
class ResNet(nn.Module):
    def __init__(self, num_classes=1):
        super(ResNet, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model():
    model.train()
    for epoch in range(2):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images).squeeze(), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

def test_model():
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            predictions.extend(model(images).squeeze().cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    return predictions, true_labels

def explain_global_feature_importance():
    model.eval()
    explainer = lime_image.LimeImageExplainer()
    
    # Initialize accumulators for three heatmaps
    heatmap_zero = None
    heatmap_positive = None
    heatmap_negative = None
    
    count_zero = 0
    count_positive = 0
    count_negative = 0

    num_images = len(test_dataset)
    
    # Get predictions
    predictions, true_labels = test_model()

    for i in range(num_images):
        img = test_dataset[i][0].numpy()
        img = np.transpose(img, (1, 2, 0))  # Convert (C, H, W) -> (H, W, C)
        
        pred_score = predictions[i]
        actual_score = true_labels[i]

        # Categorize images based on score conditions
        if abs(pred_score - actual_score) <= 0.05:
            category = "zero"
        elif pred_score > 0 and actual_score > 0 and abs(pred_score - actual_score) <= 0.05:
            category = "positive"
        elif pred_score < 0 and actual_score < 0 and abs(pred_score - actual_score) <= 0.05:
            category = "negative"
        else:
            continue  # Skip if it doesn't fall into one of the categories
        
        def model_predict(images):
            images = np.transpose(images, (0, 3, 1, 2))  # Convert (B, H, W, C) -> (B, C, H, W)
            images = torch.tensor(images, dtype=torch.float32).to(device)
            with torch.no_grad():
                return model(images).cpu().numpy()

        # Compute LIME explanation
        explanation = explainer.explain_instance(
            img, model_predict, top_labels=1, num_samples=1000, segmentation_fn=lambda x: slic(x, n_segments=100, compactness=5)
        )
        _, mask = explanation.get_image_and_mask(explanation.top_labels[0], num_features=5, hide_rest=False)

        # Accumulate importance maps for each category
        if category == "zero":
            if heatmap_zero is None:
                heatmap_zero = np.zeros_like(mask, dtype=np.float32)
            heatmap_zero += mask
            count_zero += 1
        elif category == "positive":
            if heatmap_positive is None:
                heatmap_positive = np.zeros_like(mask, dtype=np.float32)
            heatmap_positive += mask
            count_positive += 1
        elif category == "negative":
            if heatmap_negative is None:
                heatmap_negative = np.zeros_like(mask, dtype=np.float32)
            heatmap_negative += mask
            count_negative += 1

    def normalize_and_plot(heatmap, count, title):
        if heatmap is not None and count > 0:
            # Normalize heatmap
            heatmap_avg = heatmap / count
            heatmap_avg = (heatmap_avg - np.min(heatmap_avg)) / (np.max(heatmap_avg) - np.min(heatmap_avg))

            # Apply colormap
            colormap = cm.jet
            heatmap_colored = colormap(heatmap_avg)

            # Plot
            plt.figure(figsize=(8, 6))
            plt.imshow(heatmap_colored[:, :, :3])  # Show only RGB channels
            plt.axis("off")
            plt.title(title)
            plt.show()

    # Plot all three heatmaps
    normalize_and_plot(heatmap_zero, count_zero, "Global Feature Importance: Neutral (≈0)")
    normalize_and_plot(heatmap_positive, count_positive, "Global Feature Importance: Positive")
    normalize_and_plot(heatmap_negative, count_negative, "Global Feature Importance: Negative")

train_model()
predictions, true_labels = test_model()
explain_global_feature_importance()
