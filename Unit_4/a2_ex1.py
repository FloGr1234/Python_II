import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def to_grayscale(pil_image: np.ndarray) -> np.ndarray:
    if len(pil_image.shape) == 2:
        return pil_image[None]
    elif len(pil_image.shape) < 2 or len(pil_image.shape) > 3:
        raise ValueError(f"{len(pil_image.shape)} is a wrong Shape.")

    elif pil_image.shape[2] != 3:  # len(pil_image.shape) == 3 and
        raise ValueError(f"{pil_image.shape[2]} is an invalid colour size")
    image_norm = pil_image/255
    r = image_norm[:, :, 0]
    g = image_norm[:, :, 1]
    b = image_norm[:, :, 2]

    r_lin = np.where(r <= 0.04045, r/12.92, ((r+0.055)/1.055)**2.4)
    g_lin = np.where(g <= 0.04045, g/12.92, ((g+0.055)/1.055)**2.4)
    b_lin = np.where(b <= 0.04045, b/12.92, ((b+0.055)/1.055)**2.4)

    y_lin = 0.2126*r_lin + 0.7152*g_lin + 0.0722*b_lin
    y = np.where(y_lin <= 0.0031308, y_lin*12.92, 1.055*y_lin**(1/2.4)-0.055)  # y = np.where(y_lin <= 0.0031308, y_lin*12.92, 1.055*y_lin**(1/2.4)-0.055)
    y = y*255

    if np.issubdtype(pil_image.dtype, np.integer):
        y = y.astype(int)
    elif np.issubdtype(pil_image, np.floating):
        y = y.astype(float)

    y = y[None]
    print("y_final = ", y)

    #plt.imshow(y[0])

    mpimg.imsave('decorators_with_at.jpg', y[0])
    return y






        # First divide nparray/255
        # C is every singl value in the array
        # calculate Y
        #  denomilisate  Y*255
        # Y = Y[None] adds an new collum
    #return pil_image.copy()






if __name__ == "__main__":
    imag_path = "0001.jpg"
    with Image.open(imag_path) as im:
        #print(im.size)
        to_grayscale(np.array(im))

