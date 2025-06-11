import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model_setup import get_model

transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_model().to(device)
checkpoint = torch.load('models/nail_segmentation_model_final.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    augmented = transform(image=image)
    image_tensor = augmented['image'].unsqueeze(0).to(device)
    return image, image_tensor

def predict_and_get_bboxes(image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        mask = torch.sigmoid(output).squeeze().cpu().numpy()
        mask = (mask > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
    return mask, bounding_boxes

def draw_bboxes(image, bounding_boxes, output_path):
    original_h, original_w = image.shape[:2]
    scale_x = original_w / 256
    scale_y = original_h / 256
    for (x, y, w, h) in bounding_boxes:
        x_scaled = int(x * scale_x)
        y_scaled = int(y * scale_y)
        w_scaled = int(w * scale_x)
        h_scaled = int(h * scale_y)
        cv2.rectangle(image, (x_scaled, y_scaled), (x_scaled + w_scaled, y_scaled + h_scaled), (0, 255, 0), 2)
    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

image_path = 'C:/Users/Bao Cao/Documents/Nails AI/testFiles/360_F_48067989_5Pr5M50N4AdKp3MwdJ59pR89Y98TGaQB.jpg'
output_path = 'C:/Users/Bao Cao/Documents/Nails AI/output/360_F_48067989_5Pr5M50N4AdKp3MwdJ59pR89Y98TGaQB_result.jpg'

original_image, image_tensor = preprocess_image(image_path)
mask, bounding_boxes = predict_and_get_bboxes(image_tensor)
draw_bboxes(original_image, bounding_boxes, output_path)
mask_output_path = output_path.replace('.jpg', '_mask.png')
cv2.imwrite(mask_output_path, mask * 255)
print(f"Saved output: {output_path}")
print(f"Saved mask: {mask_output_path}")