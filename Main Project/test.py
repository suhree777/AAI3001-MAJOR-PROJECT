import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from utils.eval_metrics import iou, surface_distance_metric
from models.u_net import UNet
from utils.dataset import MoNuSegDataset
import os
import json
from train_val import test_annotation_paths

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Test Data
## Test Dataset
with open('test_images.json', 'r') as f:
    test_image_paths = json.load(f)

# Create Test Dataset
test_dataset = MoNuSegDataset(
    image_paths=test_image_paths,
    annotation_paths=test_annotation_paths,
    mask_dir='dataset/test/masked',
    transform=None
)

batch_size = 4

# Create DataLoader
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Initialize the model
model = UNet(n_channels=3, n_classes=1).to(device)

# Load the trained weights
model.load_state_dict(torch.load('trained_unet.pth'))
model.eval()

# Loss function
criterion = nn.BCEWithLogitsLoss()

# Test loop
total_test_loss, total_test_iou, total_surface_distance = 0.0, 0.0, 0.0

with torch.no_grad():
    for test_images, test_masks in test_loader:
        test_images, test_masks = test_images.to(device), test_masks.to(device)
        test_outputs = model(test_images)
        test_loss = criterion(test_outputs, test_masks)
        total_test_loss += test_loss.item()

        test_preds = torch.sigmoid(test_outputs) > 0.5
        test_preds = test_preds.float()

        masks_binary = test_masks > 0.5
        total_test_iou += iou(test_preds, masks_binary).item()

        # Calculate surface distance for each batch
        surface_distance = surface_distance_metric(test_preds.cpu().numpy(), masks_binary.cpu().numpy())
        total_surface_distance += surface_distance

# Calculate average test loss
average_test_loss = total_test_loss / len(test_loader)

# Calculate average IoU across all test batches
average_test_iou = total_test_iou / len(test_loader)

# Calculate average surface distance across all test batches
average_surface_distance = total_surface_distance / len(test_loader)

# Print or log the results
print(f'Test Loss: {average_test_loss:.4f}, Test IoU: {average_test_iou:.4f}, Test Surface Distance: {average_surface_distance:.4f}')

# Visualise predictions (same as before)
plt.figure(figsize=(12, 6))
for i in range(min(3, len(test_images))):  # Visualise up to 3 images
    plt.subplot(3, 3, i + 1)
    plt.imshow(test_images[i].cpu().permute(1, 2, 0))
    plt.title('Original Image')

    plt.subplot(3, 3, i + 4)
    plt.imshow(test_preds[i].cpu().squeeze(), cmap='gray')
    plt.title('Predicted Mask')

    plt.subplot(3, 3, i + 7)
    plt.imshow(test_masks[i].cpu().squeeze(), cmap='gray')
    plt.title('True Mask')

plt.tight_layout()
plt.show()
