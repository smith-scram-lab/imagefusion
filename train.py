import os
import numpy as np
from PIL import Image
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from lime import lime_image
from skimage.segmentation import mark_boundaries, slic





class SummedImageDataset(Dataset):
    def __init__(self, scores, transform=None):
        self.scores = scores
        self.transform = transform if transform else transforms.ToTensor()
        self.indices = list(range(len(scores)))

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

        concatenated_image = np.concatenate((tt_image, vp_image), axis=1)
        concatenated_image = Image.fromarray(concatenated_image)
        concatenated_image = self.transform(concatenated_image)

        if concatenated_image.shape[0] == 1:
            concatenated_image = concatenated_image.repeat(3, 1, 1)

        label = torch.tensor(self.scores[idx], dtype=torch.float32)
        return concatenated_image, label, idx

scores = np.load("sf.npy")
fused, tt, vp = scores[0], scores[1], scores[2]
scores = fused - np.maximum(tt, vp)




dataset = SummedImageDataset(scores, transform=transforms.ToTensor())
train_size = len(dataset) - 200
test_size = 200

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Extract actual test indices
test_indices = test_dataset.indices if hasattr(test_dataset, 'indices') else test_dataset.dataset.indices[test_dataset.indices]

# Wrap test dataset to include original indices
test_dataset = [(dataset[idx][0], dataset[idx][1], idx) for idx in test_indices]


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Model
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
    for epoch in range(10):
        running_loss = 0.0
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images).squeeze(), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

def test_model():
    model.eval()
    predictions, true_labels, indices = [], [], []
    with torch.no_grad():
        for images, labels, idx in test_loader:
            images, labels = images.to(device), labels.to(device)
            predictions.extend(model(images).squeeze().cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            indices.extend(idx.cpu().numpy())
    return predictions, true_labels, indices

def explain_with_lime():
    model.eval()
    explainer = lime_image.LimeImageExplainer()
    
    improvement_scores = []
    for i in range(len(test_dataset)):
        img, actual_score, idx = test_dataset[i]
        img_tensor = img.unsqueeze(0).to(device, dtype=torch.float32)
        
        with torch.no_grad():
            pred_score = model(img_tensor).cpu().numpy().max()
        
        deviation = pred_score - actual_score.numpy().max()
        if abs(deviation) <= 0.07:
            print(f"SELECTED: at index {idx}", actual_score.numpy())
            print(f"SELECTED: at index {idx}, fused score:{fused[idx]}, d = {deviation}")

            improvement_scores.append((idx, actual_score.numpy().max(), pred_score))
    
    if len(improvement_scores) < 3:
        print("Not enough accurately predicted samples for LIME visualization.")
        return
    else:
        print(f"{len(improvement_scores)} Image fits for further LIME analysis")
    
    improvement_scores.sort(key=lambda x: x[1])
    fused_scores_for_improvement = [fused[idx] for idx, _, _ in improvement_scores]

    best_idx = improvement_scores[fused_scores_for_improvement.index(max(fused_scores_for_improvement))][0]
    best_img_idx = improvement_scores[-1][0]  
    neutral_img_idx = min(improvement_scores, key=lambda x: abs(x[1]))[0]  
    worst_img_idx = improvement_scores[0][0] 
    
    selected_indices = [idx for idx in [best_idx, best_img_idx, neutral_img_idx, worst_img_idx] if 0 <= idx < 956]


    if len(selected_indices) < 4:
        print(f"Insufficient valid indices for LIME analysis: {selected_indices}")
        return
    
    for idx in selected_indices:
        actual_score = scores[idx]  
        print(f"PRINTED: at index {idx}, actual score: {actual_score}")
        print(f"PRINTED: at index {idx}, fused score: {fused[idx]}, TT score: {tt[idx]}, VP score:{vp[idx]}")

    test_samples = [dataset[idx][0].numpy() for idx in selected_indices]

    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
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
        axes[0, i].imshow(mark_boundaries(temp, mask))
        axes[0, i].axis("off")
        
        label_img = cv2.imread(os.path.join("label", f"{img_idx}.tif"),cv2.IMREAD_GRAYSCALE)
        sf_img = cv2.imread(os.path.join("sf", f"{img_idx}.png"),cv2.IMREAD_GRAYSCALE)
        axes[1, i].imshow(label_img, cmap='gray')
        axes[2, i].imshow(sf_img, cmap='gray')
        axes[1, i].axis("off")
    plt.savefig("feature_10.svg")
    plt.show()

train_model()
predictions, true_labels, test_indices = test_model()
explain_with_lime()



