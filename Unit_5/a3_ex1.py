import numpy as np
import glob
import os
import a2_ex1
import a2_ex2
from torch.utils.data import Dataset
from typing import Optional
from PIL import Image


class RandomImagePixelationDataset(Dataset):
    def __init__(self, image_dir, width_range: tuple[int, int], height_range: tuple[int, int],
                 size_range: tuple[int, int], dtype: Optional[type] = None):
        # Check the input directory
        if not os.path.isdir(image_dir):
            raise ValueError("Invalid image directory")
        # Search for all .jpg files recursive
        self.image_paths = sorted(glob.glob(image_dir + "\\**\\*.jpg", recursive=True))

        # check for the possible range
        if width_range[0] < 2 or height_range[0] < 2 or size_range[0] < 2:
            raise ValueError("The minimum of the range values must be 2")
        if width_range[0] > width_range[1] or height_range[0] > height_range[1] or size_range[0] > size_range[1]:
            raise ValueError("A minimum range value is greater than his maximum range value")

        # Set all self variables for this object
        self.width_range = width_range
        self.height_range = height_range
        self.size_range = size_range
        self.dtype = dtype

    def __getitem__(self, index):  # overwrite the x[i] function
        with Image.open(self.image_paths[index]) as im:
            # change the dtype if necessary
            if self.dtype is not None:
                image = np.array(im, dtype=self.dtype)
            else:
                image = np.array(im)
            # get the pixel size from the image
            image_width = im.size[1]
            image_hight = im.size[0]

        # to grayscale with Assignment 2
        image_gray = a2_ex1.to_grayscale(image)

        # create the random values
        rng = np.random.default_rng(index)
        width = rng.integers(self.width_range[0], high=self.width_range[1], endpoint=True)
        height = rng.integers(self.height_range[0], high=self.height_range[1], endpoint=True)
        size = rng.integers(self.size_range[0], high=self.size_range[1], endpoint=True)

        # ceck if if the pixeled sice hast to be cropped
        if width > image_hight:
            width = image_hight
        if height > image_width:
            height = image_width

        x = rng.integers(image_hight - width, endpoint=True)
        y = rng.integers(image_width - height, endpoint=True)
        pixelated_image, known_array, target_array = a2_ex2.prepare_image(image_gray, x, y, width, height, size)

        return pixelated_image, known_array, target_array, os.path.abspath(self.image_paths[index])

    def __len__(self):  # overwrite the len Funktion
        return len(self.image_paths)
