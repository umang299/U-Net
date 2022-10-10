import os
import torch
import pickle
import torchvision
from torchvision.transforms import ToTensor
import cv2 as cv
from dataset import GoodReadsDataset
from torch.utils.data import DataLoader

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)
    
def show(image, waittime):
    cv.imshow("Image", image)
    cv.waitKey(waittime)
    cv.destroyAllWindows()

def image_to_tensor(image):
    converted = ToTensor()
    tensor = converted(image)
    return tensor

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    pin_memory=True,
):
    train_ds = GoodReadsDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=0,
        shuffle=True,
    )

    val_ds = GoodReadsDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=0,
        shuffle=False,
    )

    return train_loader, val_loader

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.float().to(device)
            y = y.to(device).unsqueeze(1)
            print(model(x))
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            print(preds.size(), y.size())
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()

def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()
    
def save_model(model, folder):
    pickle.dump(model, open(folder + "\\" + "model.pkl", 'wb'))

def tif2png(tif_path, png_path):
    src = tif_path
    dst = png_path
    os.rename(src, dst)
    

def check_for_regions(data, regions):
    count = 0
    for reg in regions:
        if data.find_all(str(reg)) != []:
            count = count + 1
    return count

