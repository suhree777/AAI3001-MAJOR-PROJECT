from sklearn.model_selection import train_test_split
import torch
from utils.dataset import MoNuSegDataset
import json
import os
from torch.utils.data import DataLoader
from models.u_net import UNet
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from utils.eval_metrics import iou, evaluate_model


# Set device for training
device = torch.device("mps")

# Load Data
## For Train and Val 
image_dir = 'dataset/training_data/tissue_images'
annotation_dir = 'data/training_data/annotation'
## For Test
test_data_dir = 'dataset/test_data/'


image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.tif')]
annotation_paths = [os.path.join(annotation_dir, f.replace('.tif', '.xml')) for f in os.listdir(image_dir) if f.endswith('.tif')]

## Split train dataset to train and val sets
train_image_paths, val_image_paths, train_annotation_paths, val_annotation_paths = train_test_split(
    image_paths, annotation_paths, test_size=0.2, random_state=42)

## Test Dataset
test_image_paths = [os.path.join(test_data_dir, f) for f in os.listdir(test_data_dir) if f.endswith('.tif')]
test_annotation_paths = [os.path.join(test_data_dir, f.replace('.tif', '.xml')) for f in os.listdir(test_data_dir) if f.endswith('.tif')]

## Mask Training Dataset
train_dataset = MoNuSegDataset(
    image_paths=train_image_paths, annotation_paths=train_annotation_paths, mask_dir='dataset/train/masked'
)

## Mask Val Dataset
val_dataset = MoNuSegDataset(
    image_paths=val_image_paths, annotation_paths=val_annotation_paths, mask_dir='dataset/val/masked'
)

## Mask Test Dataset
test_dataset = MoNuSegDataset(
    image_paths=test_image_paths, annotation_paths=test_annotation_paths, mask_dir='dataset/test/masked'
)


## Save Dataset Files' names as JSON
with open('train_images.json', 'w') as f:
    json.dump(train_image_paths, f)
with open('val_images.json', 'w') as f:
    json.dump(val_image_paths, f)
with open('test_images.json', 'w') as f:
    json.dump(test_image_paths, f)

## CHeck if datasets are disjoint
def check_disjoint_sets(train_path, val_path, test_path):
    with open(train_path, 'r') as f:
        train_set = set(json.load(f))

    with open(val_path, 'r') as f:
        val_set = set(json.load(f))

    with open(test_path, 'r') as f:
        test_set = set(json.load(f))

    # Check if sets are disjoint
    if not train_set.isdisjoint(val_set) or not train_set.isdisjoint(test_set) or not val_set.isdisjoint(test_set):
        print("Warning: The train, val, and test sets are not disjoint.")
    else:
        print("The train, val, and test datasets are disjoint.")



def main():
    check_disjoint_sets('train_images.json', 'val_images.json', 'test_images.json')

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True)
    #test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)

    # Initialize UNet model
    model = UNet(in_channels=3, out_channels=1).to(device)

    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training and Validation Loops
    num_epochs = 5  # You can adjust this based on your needs
    train_losses, val_losses = [], []
    train_ious, val_ious = [], []

    for epoch in range(num_epochs):
        # Training Loop
        model.train()
        total_train_loss = 0.0
        total_train_iou = 0.0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_train_loss += loss.item()

            preds = torch.sigmoid(outputs) > 0.5
            masks_binary = masks > 0.5
            total_train_iou += iou(preds, masks_binary).item()

            loss.backward()
            optimizer.step()

        train_losses.append(total_train_loss / len(train_loader))
        train_ious.append(total_train_iou / len(train_loader))

        # Validation Loop
        model.eval()
        total_val_loss, total_val_iou = 0.0, 0.0

        with torch.no_grad():
            for val_images, val_masks in val_loader:
                val_images, val_masks = val_images.to(device), val_masks.to(device)
                val_outputs = model(val_images)
                val_loss = criterion(val_outputs, val_masks)
                total_val_loss += val_loss.item()

                val_preds = torch.sigmoid(val_outputs) > 0.5
                total_val_iou += iou(val_preds, val_masks).item()

        val_losses.append(total_val_loss / len(val_loader))
        val_ious.append(total_val_iou / len(val_loader))

        print(f'Epoch {epoch + 1}/{num_epochs}, '
              f'Train Loss: {train_losses[-1]:.4f}, Train IoU: {train_ious[-1]:.4f}, '
              f'Val Loss: {val_losses[-1]:.4f}, Val IoU: {val_ious[-1]:.4f}')

    # Save the trained model
    torch.save(model.state_dict(), 'trained_unet.pth')

    # Plot the training and validation loss and IoU
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_ious, label='Train IoU')
    plt.plot(range(1, num_epochs + 1), val_ious, label='Validation IoU')
    plt.title('IoU over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()