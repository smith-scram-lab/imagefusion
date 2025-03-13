# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, random_split
# from sklearn.model_selection import train_test_split
# from torchvision import models
# import numpy as np
# import matplotlib.pyplot as plt
# from lime import lime_image
# from skimage.segmentation import mark_boundaries
# import skimage.segmentation
# from dataset import SummedImageDataset, transform

# scores = np.load("sf.npy")
# fused, tt, vp = scores[0], scores[1], scores[2]
# scores = fused - np.maximum(tt, vp)

# dataset = SummedImageDataset(scores, transform)
# train_size = len(dataset) - 200
# test_size = 200
# train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# class ResNet(nn.Module):
#     def __init__(self, num_classes=1):
#         super(ResNet, self).__init__()
#         self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
#         self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

#     def forward(self, x):
#         return self.resnet(x)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = ResNet().to(device)
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Train the model
# def train_model():
#     num_epochs = 2
#     print("Starting training...")
#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
#         print(f"Epoch {epoch+1}/{num_epochs} - Training...")
#         for images, labels in train_loader:
#             images, labels = images.to(device), labels.to(device)

#             optimizer.zero_grad()
#             outputs = model(images).squeeze()

#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()

#         print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

# # Test the model and pick the most accurate predictions
# def test_model():
#     model.eval()
#     predictions = []
#     true_labels = []
#     test_loss = 0.0
#     with torch.no_grad():
#         for images, labels in test_loader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images).squeeze()
#             loss = criterion(outputs, labels)
#             test_loss += loss.item()
#             predictions.extend(outputs.cpu().numpy())
#             true_labels.extend(labels.cpu().numpy())
#     avg_test_loss = test_loss / len(test_loader)
#     print(f"Testing completed. Average Test Loss: {avg_test_loss:.6f}")
#     return predictions, true_labels

# def explain_with_lime():
#     explainer = lime_image.LimeImageExplainer()
#     accurate_predictions = []
    
#     for idx in range(len(test_dataset)):
#         image, true_label = test_dataset[idx]
#         image = image.unsqueeze(0).to(device)  # Add batch dimension
#         true_label = true_label.item()

#         with torch.no_grad():
#             model_output = model(image).squeeze().cpu().numpy()
        
#         deviation = abs(model_output - true_label)
        
#         if deviation <= 0.05:
#             accurate_predictions.append((idx, model_output, true_label, deviation))

#     accurate_predictions = sorted(accurate_predictions, key=lambda x: x[3])
#     pos_group, zero_group, neg_group = [], [], []

#     for idx, pred, true_label, deviation in accurate_predictions:
#         if abs(pred) <= 0.05 and abs(true_label) <= 0.05: 
#             zero_group.append((idx, pred, true_label))
#         elif pred > 0 and true_label > 0:  
#             pos_group.append((idx, pred, true_label))
#         elif pred < 0 and true_label < 0:  
#             neg_group.append((idx, pred, true_label))

#     selected_groups = [pos_group[:1], zero_group[:1], neg_group[:1]]  # Pick 1 from each for visualization

#     fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # One row, three columns

#     for i, group in enumerate(selected_groups):
#         if len(group) == 0:
#             continue  # Skip if no image available

#         idx, pred, true_label = group[0]
#         image = test_dataset[idx][0]  # No need to unsqueeze here

#         image_1channel = image.numpy()

#         # If the image is grayscale, it will have shape (1, H, W). In this case, squeeze the first dimension.
#         if len(image_1channel.shape) == 3 and image_1channel.shape[0] == 1:
#             image_1channel = image_1channel.squeeze(0)  # Remove the first dimension (grayscale)

#         # Now image_1channel should have shape (H, W)
#         image_3channel = np.repeat(image_1channel, 3, axis=0)  # Convert (1, H, W) → (3, H, W)

#         # Debug: Check the shape of image_3channel
#         print(f"Image shape before transpose: {image_3channel.shape}")

#         image_3channel = np.transpose(image_3channel, (1, 2, 0))  # Convert (3, H, W) → (H, W, 3)

#         def model_predict(images):
#             images = np.transpose(images, (0, 3, 1, 2))  # Convert (B, H, W, C) → (B, C, H, W)
#             images = torch.tensor(images, dtype=torch.float32).to(device)
#             with torch.no_grad():
#                 preds = model(images).cpu().numpy()
#             return preds

#         # Run LIME explanation with a grayscale-friendly segmentation function
#         explanation = explainer.explain_instance(
#             image_1channel,  # Now it's (H, W)
#             model_predict,
#             top_labels=1,
#             hide_color=0,
#             num_samples=1000,
#             segmentation_fn=lambda img: skimage.segmentation.slic(img, n_segments=100, compactness=5, start_label=1)
#         )

#         temp, mask = explanation.get_image_and_mask(
#             explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False
#         )

#         axes[i].imshow(mark_boundaries(temp, mask))
#         axes[i].set_title(f"Group: {'Positive' if i == 0 else 'Zero' if i == 1 else 'Negative'}\nPred: {pred:.2f}, True: {true_label:.2f}")
#         axes[i].axis("off")

#     plt.show()



# train_model()
# predictions, true_labels = test_model()
# explain_with_lime()
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

# def explain_with_lime():
#     model.eval()
#     explainer = lime_image.LimeImageExplainer()
#     test_samples = [test_dataset[i][0].numpy() for i in range(3)]  # Pick 3 images
    
#     fig, axes = plt.subplots(1, 3, figsize=(15, 5))
#     for i, img in enumerate(test_samples):
#         img = np.transpose(img, (1, 2, 0))  
#         def model_predict(images):
#             images = np.transpose(images, (0, 3, 1, 2))  
#             images = torch.tensor(images, dtype=torch.float32).to(device)
#             with torch.no_grad():
#                 return model(images).cpu().numpy()
        
#         explanation = explainer.explain_instance(img, model_predict, top_labels=1, num_samples=1000, segmentation_fn=lambda x: slic(x, n_segments=100, compactness=5))
#         temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], num_features=5, hide_rest=False)
#         axes[i].imshow(mark_boundaries(temp, mask))
#         axes[i].axis("off")
#     plt.show()
def explain_with_lime():
    model.eval()
    explainer = lime_image.LimeImageExplainer()
    
    # Compute prediction improvement scores for all test images
    improvement_scores = []
    for i in range(len(test_dataset)):
        img = test_dataset[i][0].numpy()
        img_tensor = torch.tensor(img).unsqueeze(0).to(device, dtype=torch.float32)
        actual_score = test_dataset[i][1].numpy().max()  # Assuming actual label confidence
        
        with torch.no_grad():
            pred_score = model(img_tensor).cpu().numpy().max()
        
        improvement_score = pred_score - actual_score
        
        # Only consider images where the deviation is within 0.03
        if abs(improvement_score) <= 0.03:
            improvement_scores.append((i, improvement_score))
    
    # Sort images by improvement score
    improvement_scores.sort(key=lambda x: x[1])
    
    # Select three images based on improvement score criteria
    best_img_idx = improvement_scores[-1][0]  # Greatest positive improvement
    neutral_img_idx = min(improvement_scores, key=lambda x: abs(x[1]))[0]  # Closest to zero
    worst_img_idx = improvement_scores[0][0]  # Most negative improvement
    
    selected_indices = [best_img_idx, neutral_img_idx, worst_img_idx]
    test_samples = [test_dataset[i][0].numpy() for i in selected_indices]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, (img, img_idx) in enumerate(zip(test_samples, selected_indices)):
        img = np.transpose(img, (1, 2, 0))
        
        def model_predict(images):
            images = np.transpose(images, (0, 3, 1, 2))
            images = torch.tensor(images, dtype=torch.float32).to(device)
            with torch.no_grad():
                return model(images).cpu().numpy()
        
        explanation = explainer.explain_instance(
            img, model_predict, top_labels=1, num_samples=1000, 
            segmentation_fn=lambda x: slic(x, n_segments=100, compactness=5)
        )
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], num_features=5, hide_rest=False)
        axes[i].imshow(mark_boundaries(temp, mask))
        axes[i].axis("off")
    plt.show()



train_model()
predictions, true_labels = test_model()
explain_with_lime()
