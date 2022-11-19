import os
import logging
import pathlib
import cv2
from torch.utils.data import (
    Dataset,
    DataLoader
)
from configs.general_configs import (
    DATA_FORMAT,
    MASK_BINARY,
    MASK_ZERO_BOUNDARY
)
import numpy as np
from utils.logging_funcs import info_log
from configs.general_configs import INFO_BAR_HEIGHT

__author__ = 'mchlsdrv@gmail.com'

logging.getLogger('PIL').setLevel(logging.WARNING)


# - CLASSES
class NanoUnetDS(Dataset):
    def __init__(self, data_dir: pathlib.Path, image_files: list, augmentations=None, data_format: str = 'tiff', binary_masks: bool = False, zero_boundary: bool = False):
        """
        Feeds files from a directory. The expected dir structure should be root (data_dir), under which should be two subdirectories, viz. images and masks.
        Each mask name should be the same as its corresponding image but with a '_mask' suffix, e.g., for image 1.tif the mask should be named 1_mask.tif
        :param data_dir: The root data directory where there are two subdirectories, viz. images and masks
        :param image_files: The image files to use in this dataset
        :param augmentations: The augmentation function to apply to the images and masks
        :param data_format: The format the images and the masks are coded in (e.g., tif, jpg etc.)
        """
        self.image_dir = data_dir / 'images'
        self.mask_dir = data_dir / 'masks'
        self.augmentations = augmentations
        self.data_fmt = data_format
        self.image_fls = image_files
        self.binary_masks = binary_masks
        self.zero_boundary = zero_boundary

    def __len__(self):
        return len(self.image_fls)

    def __getitem__(self, index):
        # - Get the path to image and mask
        img_path = self.image_dir / self.image_fls[index]
        mask_path = self.mask_dir / self.image_fls[index].replace(f'.tif', f'.tiff')
        # mask_path = self.mask_dir / self.image_fls[index].replace(f'.{self.data_fmt}', f'_mask.{self.data_fmt}')

        # - Load image and mask
        # print('image path: ', img_path)
        img = cv2.imread(str(img_path), -1)[:-INFO_BAR_HEIGHT, :]
        # print('mask path', mask_path)
        mask = cv2.imread(str(mask_path), -1)[:-INFO_BAR_HEIGHT, :]
        # img = np.array(Image.open(str(img_path)).convert('L'), dtype=np.uint8)
        # mask = np.array(Image.open(str(mask_path)).convert('L'), dtype=np.float32)

        # - Preprocess the mask
        if self.binary_masks:
            mask[mask > 0] = 1.0
        elif self.zero_boundary:
            mask[mask > 1] = 0.0  # -> zeros the boundary
            mask[mask == 2] = 1.0  # -> converts the foreground to the first class

        # - Augment image and mask
        if self.augmentations is not None:
            augs = self.augmentations(image=img, mask=mask)
            img, mask = augs.get('image'), augs.get('mask')

        return img, mask


# - FUNCTIONS
def get_train_val_split(data: list or np.ndarray, val_prop: float = .2, logger: logging.Logger = None):
    # - Find the total number of samples
    n_items = len(data)

    all_idxs = np.arange(n_items)

    # - Choose the items for validation
    val_idxs = np.random.choice(all_idxs, int(val_prop * n_items), replace=False)

    # - Choose the items for training
    train_idxs = np.setdiff1d(all_idxs, val_idxs)

    # - Convert the data from list into numpy.ndarray object to use the indexing
    np_data = np.array(data, dtype=object)

    # - The items for training are the once which are not included in the validation set
    train_data = np_data[train_idxs]

    # - Pick the items for the validation set
    val_data = np_data[val_idxs]

    info_log(logger=logger, message=f'| Number of train data files : {len(train_data)} | Number of validation data files : {len(val_data)} |')

    return train_data, val_data


def convert_2_int(mask: np.ndarray):
    """
    Converts a float mask to int mask, i.e., mask with running index values
    :param mask: The mask to perform the transformation on
    :return: np.ndarray - mask with int values
    """
    for lbl_idx, lbl_id in np.unique(mask):
        idxs = np.argwhere(mask == lbl_id)
        x, y = idxs[:, 0], idxs[:, 1]
        mask[x, y] = lbl_idx
    return mask


def get_data_loaders(data_dir: pathlib.Path, batch_size: int, train_augs, val_augs, val_prop: float = 0.2, num_workers: int = 4, pin_memory: bool = True, logger: logging.Logger = None):

    # - Split the files into train and validation files
    train_fls, val_fls = get_train_val_split(data=os.listdir(data_dir / 'images'), val_prop=val_prop, logger=logger)

    # - Get the train dataset, and data loader
    train_ds = NanoUnetDS(data_dir=data_dir, image_files=train_fls, augmentations=train_augs, data_format=DATA_FORMAT, binary_masks=MASK_BINARY, zero_boundary=MASK_ZERO_BOUNDARY)
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True
    )

    # - Get the val dataset, and data loader
    val_ds = NanoUnetDS(data_dir=data_dir, image_files=val_fls, augmentations=val_augs, data_format=DATA_FORMAT, binary_masks=MASK_BINARY, zero_boundary=MASK_ZERO_BOUNDARY)
    val_dl = DataLoader(
        val_ds,
        batch_size=1,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )

    return train_dl, val_dl
