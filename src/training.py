import torch
import torch.optim as optim
import segmentation_models_pytorch as smp
from model_setup import get_model
from data_loaders import get_loaders
import logging
import time
from datetime import timedelta
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--start-epoch', type=int, default=0)
args = parser.parse_args()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log', mode='a'),
        logging.StreamHandler()
    ]
)

start_time = time.time()
logging.info("Start")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = get_model().to(device)
logging.info("Model loaded")

loss_fn = smp.losses.DiceLoss(mode='binary')
optimizer = optim.Adam(model.parameters(), lr=1e-4)

def load_checkpoint(model, optimizer, load_path, start_epoch_default):
    checkpoint = torch.load(load_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint.get('epoch', start_epoch_default)
    else:
        model.load_state_dict(checkpoint)
        epoch = start_epoch_default
    return model, optimizer, epoch

if args.resume and os.path.exists(args.resume):
    logging.info(f"Loading checkpoint {args.resume}")
    model, optimizer, start_epoch = load_checkpoint(model, optimizer, args.resume, args.start_epoch)
    logging.info(f"Resuming epoch {start_epoch + 1}")
else:
    start_epoch = args.start_epoch
    logging.info("Starting")

train_loader, valid_loader, _ = get_loaders()
logging.info(f"Training batch: {len(train_loader)}, Validation batch: {len(valid_loader)}")

num_epochs = 6
batch_times = []

os.makedirs('models', exist_ok=True)

for epoch in range(start_epoch, num_epochs):
    epoch_start = time.time()
    logging.info(f"\n*** Starting epoch {epoch + 1}/{num_epochs} ***")
    model.train()
    train_loss = 0
    for batch_idx, (images, masks) in enumerate(train_loader):
        try:
            batch_start = time.time()
            images = images.to(device)
            masks = masks.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            logging.info(f"*** Current Epoch: {epoch + 1}/{num_epochs} ***")
            logging.info(f"Training batch {batch_idx + 1}/{len(train_loader)}")
            logging.info(f"Batch loss: {loss.item():.4f}")
            logging.info(f"Batch {batch_idx + 1} runtime: {batch_time:.2f} seconds")

            if len(batch_times) >= 5:
                avg_batch_time = sum(batch_times) / len(batch_times)
                batches_left = (len(train_loader) * (num_epochs - epoch)) - (batch_idx + 1)
                time_left = avg_batch_time * batches_left
                logging.info(f"Uptime: {timedelta(seconds=int(time.time() - start_time))}")
                logging.info(f"Time left: {timedelta(seconds=int(time_left))}")

        except Exception as e:
            logging.error(f"Error in batch {batch_idx + 1}: {e}")
            raise


    model.eval()
    valid_loss = 0
    logging.info("Validating")
    try:
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(valid_loader):
                images = images.to(device)
                masks = masks.to(device).unsqueeze(1)
                outputs = model(images)
                valid_loss += loss_fn(outputs, masks).item()
                if (batch_idx + 1) % 10 == 0:
                    logging.info(f"Validation batch {batch_idx + 1}/{len(valid_loader)}, Current loss: {valid_loss / (batch_idx + 1):.4f}")
    except Exception as e:
        logging.error(f"Error in validation: {e}")
        raise

    logging.info(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss / len(train_loader):.4f}, Valid Loss: {valid_loss / len(valid_loader):.4f}")
    logging.info(f"Epoch runtime: {timedelta(seconds=int(time.time() - epoch_start))}")

    checkpoint_path = f'models/nail_segmentation_model_epoch_{epoch + 1}.pth'
    try:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_path)
        logging.info(f"Saved checkpoint: {checkpoint_path}")
    except Exception as e:
        logging.error(f"Failed to save: {e}")

logging.info("Saving model")
try:
    torch.save({
        'epoch': num_epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, 'models/nail_segmentation_model_final.pth')
except Exception as e:
    logging.error(f"Failed to save final model: {e}")
logging.info(f"Total runtime: {timedelta(seconds=int(time.time() - start_time))}")
logging.info("Complete")