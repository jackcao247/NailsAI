from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset import NailDataset

def get_transforms():
    return A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.ToTensorV2(),
    ], additional_targets={"mask": "mask"})

def get_loaders():
    print("Initializing datasets")
    train_dataset = NailDataset(
        img_dir='data/train/images',
        mask_dir='data/train/masks',
        transform=get_transforms()
    )
    valid_dataset = NailDataset(
        img_dir='data/valid/images',
        mask_dir='data/valid/masks',
        transform=get_transforms()
    )
    test_dataset = NailDataset(
        img_dir='data/test/images',
        mask_dir='data/test/masks',
        transform=get_transforms()
    )
    print(f"Train dataset: {len(train_dataset)} images")
    print(f"Valid dataset: {len(valid_dataset)} images")
    print(f"Test dataset: {len(test_dataset)} images")

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    return train_loader, valid_loader, test_loader