import torch
from scipy.spatial.distance import directed_hausdorff
import numpy as np

def iou(preds, masks):
    # Implementation of IoU (Intersection over Union)
    intersection = torch.logical_and(preds, masks).sum()
    union = torch.logical_or(preds, masks).sum()

    iou_value = intersection / union if union != 0 else 0.0
    return iou_value

def surface_distance_metric(segmentation, ground_truth):
    # Assuming 'segmentation' and 'ground_truth' are binary arrays
    seg_coords = np.array(np.where(segmentation)).T
    gt_coords = np.array(np.where(ground_truth)).T
    hausdorff_distance = directed_hausdorff(seg_coords, gt_coords)[0]
    return hausdorff_distance

def evaluate_model_with_surface_metrics(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_surface_distance = 0.0

    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()

            preds = torch.sigmoid(outputs) > 0.5
            masks_binary = masks > 0.5
            total_iou += iou(preds, masks_binary).item()

            surface_distance = surface_distance_metric(preds.cpu().numpy(), masks_binary.cpu().numpy())
            total_surface_distance += surface_distance

    avg_loss = total_loss / len(dataloader)
    avg_iou = total_iou / len(dataloader)
    avg_surface_distance = total_surface_distance / len(dataloader)

    return avg_loss, avg_iou, avg_surface_distance
