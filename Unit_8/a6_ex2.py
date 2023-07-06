import torch
import glob
import os
from PIL import Image
from typing import Union, Sequence
from a6_ex1 import random_augmented_image


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir):
        image_dir = os.path.abspath(image_dir)
        self.image_path_list = sorted(glob.glob(image_dir + "\\**\\*.jpg", recursive=True))

    def __getitem__(self, index: int):
        im = Image.open(self.image_path_list[index])
        return im, index

    def __len__(self):
        return len(self.image_path_list)


class TransformedImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: ImageDataset, image_size: Union[int, Sequence[int]]):
        self.dataset = dataset
        self.image_size = image_size

    def __getitem__(self, index: int):
        augmented_image = random_augmented_image(self.dataset[index][0], self.image_size, index)
        return augmented_image, index

    def __len__(self):
        return len(self.dataset)


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from torchvision import transforms

    imgs = ImageDataset(image_dir=r'.\Test_errors')
    transformed_imgs = TransformedImageDataset(imgs, image_size=300)
    for (original_img, index), (transformed_img, _) in zip(imgs, transformed_imgs):
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(original_img)
        axes[0].set_title("Original image")
        axes[1].imshow(transforms.functional.to_pil_image(transformed_img))
        axes[1].set_title("Transformed image")
        fig.suptitle(f"Image {index}")
        fig.tight_layout()
        plt.show()
