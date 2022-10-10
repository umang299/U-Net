import os
import cv2 as cv
from torch.utils.data import Dataset



# Creating a dataset class for image and masks
class GoodReadsDataset(Dataset):
    def __init__(self, image_dir, mask_dir,  transform=None):
        super().__init__()
        self.image_dir = image_dir      # image directory
        self.mask_dir = mask_dir        # mask directory
        self.transform = transform      # apply transformations on data


    # fuction to return length of dataset
    def __len__(self):
        return len(os.listdir(self.mask_dir))

    # function to print image and mask at a given index
    def __getitem__(self,index):
        img_path = os.path.join(self.image_dir, os.listdir(self.image_dir)[index])
        mask_path = os.path.join(self.mask_dir, os.listdir(self.mask_dir)[index])
        image, mask = cv.imread(img_path), cv.imread(mask_path)
        # converting RGB image to gray scale to save compute
        image, mask =cv.cvtColor(image, cv.COLOR_RGB2GRAY), cv.cvtColor(mask, cv.COLOR_RGB2GRAY)
        img_h, img_w  = image.shape
        mask_h, mask_w = mask.shape

        image = image.reshape(1, img_h, img_w)
        mask = image.reshape(1, mask_h, mask_w)
        assert image.shape == mask.shape,f"InvalidInput: Found image with shape {image.shape} and mask with shape {mask.shape}"

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]


        return image, mask


