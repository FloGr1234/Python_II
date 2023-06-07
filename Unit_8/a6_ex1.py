import numpy as np
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt

from typing import Union, Sequence
from torchvision import transforms

def random_augmented_image(
    image: Image,
    image_size: Union[int, Sequence[int]],
    seed: int
) -> torch.Tensor:

    torch.random.manual_seed(seed)

    # 2 random of this 4 things

    idx = []
    while len(idx) < 2:
        rand_int = int(torch.rand(1).item()*4)
        if len(idx) >= 1:
            if rand_int != idx[0]:
                idx.append(rand_int)
        else:
            idx.append(rand_int)

    rand_transforms = [transforms.RandomRotation(degrees=torch.rand(1).item() * 360),
                       torchvision.transforms.RandomVerticalFlip(p=0.1),
                       torchvision.transforms.RandomHorizontalFlip(p=0.1),
                       torchvision.transforms.ColorJitter(torch.rand(1).item(), torch.rand(1).item(),
                                                          torch.rand(1).item(), torch.rand(1).item() * 0.5),
                       # (brightness,contrast,saturation,hue)
                       ]

    transform_chain = transforms.Compose([
        transforms.Resize(image_size),
        rand_transforms[idx[0]],
        rand_transforms[idx[1]],
        transforms.ToTensor(),  # Transform a PIL or numpy array to a tensor
        torch.nn.Dropout(p=0.02, inplace=False)
    ])
    transformed_image = transform_chain(image)
    return transformed_image











if __name__ == "__main__":
    from matplotlib import pyplot as plt
    # with Image.open("test_image.jpg") as image:
    with Image.open("08_example_image.jpg") as image:
        transformed_image = random_augmented_image(image, image_size=300, seed=3)
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(image)
        axes[0].set_title("Original image")
        axes[1].imshow(transforms.functional.to_pil_image(transformed_image))
        axes[1].set_title("Transformed image")
        fig.tight_layout()
        plt.show()

