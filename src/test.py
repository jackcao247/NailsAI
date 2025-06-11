import torch
import numpy as np
from torchmetrics import Precision, Recall, F1Score
import segmentation_models_pytorch as smp
from model_setup import get_model
from data_loaders import get_loaders

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = get_model().to(device)
checkpoint = torch.load('models/nail_segmentation_model_final.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

_, _, test_loader = get_loaders()

loss_fn = smp.losses.DiceLoss(mode='binary')

precision_metric = Precision(task='binary', average='macro').to(device)
recall_metric = Recall(task='binary', average='macro').to(device)
f1_metric = F1Score(task='binary', average='macro').to(device)

test_loss = 0.0
ious = []
aps = []
iou_thresholds = np.arange(0.5, 1.0, 0.05)

def calc_iou(pred_mask, gt_mask, threshold=0.5):
    pred_mask = (pred_mask > threshold).float()
    intersection = (pred_mask * gt_mask).sum()
    union = pred_mask.sum() + gt_mask.sum() - intersection
    iou = intersection / (union + 1e-6)
    return iou.item()

def calc_avg_precision(pred_mask, gt_mask, conf_thresholds=np.linspace(0.1, 0.9, 9)):
    precisions = []
    recalls = []
    gt_positives = gt_mask.sum().item()
    
    for thresh in conf_thresholds:
        pred_binary = (pred_mask > thresh).float()
        tp = (pred_binary * gt_mask).sum().item()
        fp = pred_binary.sum().item() - tp
        fn = gt_positives - tp
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (gt_positives + 1e-6)
        precisions.append(precision)
        recalls.append(recall)
    
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    sorted_indices = np.argsort(recalls)
    precisions = precisions[sorted_indices]
    recalls = recalls[sorted_indices]

    interpolated_precisions = np.maximum.accumulate(precisions[::-1])[::-1]

    ap = np.trapz(interpolated_precisions, recalls)
    return ap

with torch.no_grad():
    for batch_idx, (images, masks) in enumerate(test_loader):
        images = images.to(device)
        masks = masks.to(device).unsqueeze(1)
        
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        test_loss += loss.item()
        
        probs = torch.sigmoid(outputs)
        
        precision_metric.update(probs, masks.int())
        recall_metric.update(probs, masks.int())
        f1_metric.update(probs, masks.int())
        
        for i in range(images.size(0)):
            pred_mask = probs[i, 0]
            gt_mask = masks[i, 0]
            
            iou_scores = [calc_iou(pred_mask, gt_mask, thresh) for thresh in iou_thresholds]
            ious.append(np.mean(iou_scores))
            
            ap = calc_avg_precision(pred_mask, gt_mask)
            aps.append(ap)

avg_test_loss = test_loss / len(test_loader)
avg_precision = precision_metric.compute().item()
avg_recall = recall_metric.compute().item()
avg_f1 = f1_metric.compute().item()
avg_iou = np.mean(ious)
mAP = np.mean(aps)

print(f"Dice score: {avg_test_loss:.4f}")
print(f"Precision score: {avg_precision:.4f}")
print(f"Recall: {avg_recall:.4f}")
print(f"F1 Score: {avg_f1:.4f}")
print(f"Avg IoU: {avg_iou:.4f}")