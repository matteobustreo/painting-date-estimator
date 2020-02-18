import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import os

import torch
from torch.utils.data import Dataset

# Defining the pyTorch custom dataset
class PaintersDataset(Dataset):
    def __init__(self, dataframe, base_dir, transform=None, log_enabled=False):

        self.img_filenames = dataframe.filename
        self.dates = dataframe.date
        self.transform = transform
        self.base_dir = base_dir

        self.log_enabled = log_enabled

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename = self.base_dir + self.img_filenames.iloc[idx]

        if os.path.isfile(filename):
            img = Image.open(filename)
            if not img.mode == 'RGB':
                img = img.convert('RGB')
        else:
            img = None

        # If corrupted image, generating an empty one with fake label
        if img is None or len(img.getbands()) < 3:
            if self.log_enabled:
                if img is None:
                    print("Corrupted image! - {} - Path: {} - Img is None!".format(idx, filename))
                else:
                    print("Corrupted image! - {} - Path: {} - len(img.getbands())".format(
                        idx, self.img_filenames[idx], len(img.getbands())))

            img = Image.fromarray(np.zeros([500, 500, 3], dtype=np.uint8))
            label = 0 * self.dates.iloc[idx]
        else:
            label = self.dates.iloc[idx]

        if self.transform:
            img = self.transform(img)

        return (img, label, self.base_dir + self.img_filenames.iloc[idx])

    def __len__(self):
        return len(self.img_filenames)