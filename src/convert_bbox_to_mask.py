import os
import numpy as np
from PIL import Image
from pycocotools.coco import COCO

def create_mask_from_segmentation(coco, img_id, output_dir):
    img_info = coco.loadImgs(img_id)[0]
    ann_ids = coco.getAnnIds(imgIds=img_id, catIds=coco.getCatIds(), iscrowd=0)
    anns = coco.loadAnns(ann_ids)
    mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
    
    print(f"Processing image ID: {img_id}, file: {img_info['file_name']}, annotations: {len(anns)}")
    
    for ann in anns:
        print(f"Annotation ID: {ann['id']}, category_id: {ann['category_id']}, segmentation: {ann.get('segmentation', 'Missing')}")
        if 'segmentation' in ann and ann['segmentation']:
            try:
                ann_mask = coco.annToMask(ann)
                mask = np.maximum(mask, ann_mask)
            except Exception as e:
                print(f"Error processing annotation ID {ann['id']} for image ID {img_id}: {e}")
                print(f"Full annotation: {ann}")
                raise
        else:
            print(f"Annotation ID {ann['id']} has missing or empty segmentation")
    
    mask_img = Image.fromarray(mask * 255)
    output_path = os.path.join(output_dir, img_info['file_name'].replace('.jpg', '_mask.png'))
    mask_img.save(output_path)

for split in ['train', 'valid', 'test']:
    json_path = f'data/{split}/_annotations.coco.json'
    if os.path.exists(json_path):
        coco = COCO(json_path)
        os.makedirs(f'data/{split}/masks', exist_ok=True)
        for img_id in coco.getImgIds():
            create_mask_from_segmentation(coco, img_id, f'data/{split}/masks')