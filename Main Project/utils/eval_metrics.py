import torch
from torch.utils.data import DataLoader
import torch.nn as nn

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0

    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()

            preds = torch.sigmoid(outputs) > 0.5
            masks_binary = masks > 0.5
            total_iou += iou(preds, masks_binary).item()

    avg_loss = total_loss / len(dataloader)
    avg_iou = total_iou / len(dataloader)

    return avg_loss, avg_iou

def iou(preds, masks):
    # Implementation of IoU (Intersection over Union)
    intersection = torch.logical_and(preds, masks).sum()
    union = torch.logical_or(preds, masks).sum()

    iou_value = intersection / union if union != 0 else 0.0
    return iou_value
