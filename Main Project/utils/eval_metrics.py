def iou(pred, target):
    intersection = (pred & target).sum((1, 2))
    union = (pred | target).sum((1, 2))
    iou = (intersection.float() / union.float()).mean()
    return iou.item()
