import numpy as np
from PIL import Image
import os

def to_grayscale(pil_image: np.ndarray) -> np.ndarray:
    print(pil_image.size)





if __name__ == "__main__":
    imag_path = "0001.jpg"
    with Image.open(imag_path) as im:
        to_grayscale(im)

