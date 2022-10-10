import sys
import os
import config
from tqdm import tqdm
from PIL import Image

# mask directory
mask_dir = config.val_mask
dst_dir = config.mask_dst_dir


# function to convert a .gif to .jpg file
def processImage( infile ):
    try:
        im = Image.open( infile )
    except IOError:
        print("Cant load", infile)
        sys.exit(1)

    i = 0
    
    try:
        while 1:
            name = infile.replace(".gif", "").split("\\")[-1]
            background = Image.new("RGB", im.size, (255, 255, 255))
            background.paste(im)
            background.save(os.path.join(dst_dir, name) + ".jpg", 'JPEG', quality=80)

            i += 1
            im.seek( im.tell() + 1 )

    except EOFError:
        pass # end of sequenc

# iterating over mask directory
print(f"Saving {len(os.listdir(mask_dir))} .gif images to .jpg.....")
for mask in tqdm(os.listdir(mask_dir), total=len(os.listdir(mask_dir))):
    mask_path = os.path.join(mask_dir, mask)
    processImage(mask_path)

print(f"Converted {len(os.listdir(dst_dir))} .gif images to .jpg")


