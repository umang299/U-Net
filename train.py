import config
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import GoodReadsDataset
from engine import train_fn
from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    check_accuracy,
    save_predictions_as_imgs,
)

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_EPOCHS = 3
IMAGE_HEIGHT = 1280  # 1280 originally
IMAGE_WIDTH = 1918  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = config.train_image
TRAIN_MASK_DIR = config.train_mask
VAL_IMG_DIR = config.val_image
VAL_MASK_DIR = config.val_mask


# Train Data Augmentation
train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

# Validation Data Augmentation
val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )


# defining train dataset
train_ds = GoodReadsDataset(image_dir=TRAIN_IMG_DIR, mask_dir=TRAIN_MASK_DIR)
val_ds = GoodReadsDataset(image_dir=VAL_IMG_DIR, mask_dir=VAL_MASK_DIR)

print(f"Found train directory {TRAIN_IMG_DIR} with {train_ds.__len__()} samples")
print(f"Found train directory {TRAIN_MASK_DIR} with {train_ds.__len__()} samples")

print(f"Found train directory {VAL_IMG_DIR} with {val_ds.__len__()} samples")
print(f"Found train directory {VAL_MASK_DIR} with {val_ds.__len__()} samples")

train_loader = DataLoader(dataset=train_ds,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=0)

val_loader = DataLoader(dataset=val_ds,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=0)

model = UNET(in_channels=1, out_channels=1).to(DEVICE)      # initilizing the model
loss_fn = nn.BCEWithLogitsLoss()        # initializing the loss function
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)        # initializing the optimizer


if LOAD_MODEL:      # if true will load the pretrained model 
    """ CAUTION: USE ONLY WHEN YOU HAVE A TRAINED MODEL AND PASS THE PATH OF THE MODEL BELOW"""
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)


check_accuracy(val_loader, model, device=DEVICE)
scaler = torch.cuda.amp.GradScaler()

PYTORCH_CUDA_ALLOC_CONF=10
torch.cuda.empty_cache()
for epoch in range(NUM_EPOCHS):
    print(torch.cuda.memory_snapshot())
    
    train_fn(train_loader, model, optimizer, loss_fn, scaler)

    # save model
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer":optimizer.state_dict(),
        }
    save_checkpoint(checkpoint)

    # check accuracy
    check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    # print some examples to a folder
    save_predictions_as_imgs(
    val_loader, model, folder="saved_images/", device=DEVICE
    )