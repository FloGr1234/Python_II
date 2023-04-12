import numpy as np
from PIL import Image
from a2_ex1 import to_grayscale
from matplotlib import cm
import matplotlib.image as mplimg

def prepare_image(image: np.ndarray, x: int, y: int, width: int, height: int, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # check from the input values
    if len(image.shape) != 3:
        raise ValueError(f"the image is not an 3D array")
    g, h, w = image.shape
    if x + width > w or x < 0:
        raise ValueError(f"the pixelated area would exceed the input image width")
    if y + height > h or y < 0:
        raise ValueError(f"the pixelated area would exceed the input image height")
    if height < 2 or width < 2 or size < 2:
        raise ValueError("height, width or size are too small")


    # The original target array
    target_array = image[0, x:x+width, y:y+height]
    pixelated_image = image.copy()

    # pixeling
    for y_i in range(y, y + height, size):
        # y correction at the end of the array
        kor_y = 0
        if y_i + size > y + width:
            kor_y = y_i + size - (y + height)
        for x_i in range(x, x+width, size):
            # x correction at the end of the array
            kor_x = 0
            if x_i+size > x+width:
                kor_x = x_i+size - (x+width)

            # set the mean value
            mean = np.mean(image[0, x_i:x_i+size-kor_x, y_i:y_i+size-kor_y])
            pixelated_image[0, x_i:x_i+size-kor_x, y_i:y_i+size-kor_y] = mean


    # generate the known array
    known_array = np.where(image == pixelated_image, True, False)

    # Für test mit 'prepare_image(gray_im, 100, 100, 5, 5, 2)'
    #print("pixelated = \n", pixelated_image[0, 98:107, 98:107])
    #print("original = \n", image[0, 98:107, 98:107])
    #print("map = \n", known_array[0, 98:107, 98:107])
    #print("target_array =\n", target_array)
    mplimg.imsave('pixeled.jpg', np.uint8(pixelated_image[0]), cmap=cm.gray)
    return (pixelated_image, known_array, target_array)


if __name__ == "__main__":
    imag_path = "0001.jpg"
    with Image.open(imag_path) as im:
        print(im.size)
        gray_im = to_grayscale(np.array(im))
    print(prepare_image(gray_im, 100, 300, 100, 1, 2))
