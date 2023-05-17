import torch
import numpy as np


def stack_with_padding(batch_as_list: list):
    # finde the max width and max height
    max_heigth = 0
    max_width = 0
    for im, _, _, _ in batch_as_list:
        if im.shape[1] > max_heigth:
            max_heigth = im.shape[1]
        if im.shape[2] > max_width:
            max_width = im.shape[2]

    # changes from the arrays
    pixelated_images = []
    known_arrays = []
    target_arrays = []
    image_files = []
    for pixelated_image, known_array, target_array, image_file in batch_as_list:
        # pixelated image
        h = pixelated_image.shape[1]
        w = pixelated_image.shape[2]
        padded1 = np.full((max_heigth, max_width), 0)
        padded1[0:h, 0:w] = pixelated_image[0]
        pixelated_images.append(torch.tensor(padded1[None]))
        #pixelated_images.append(padded1[None])

        # create the known array expendet to the max_height and max_width
        h = known_array.shape[1]
        w = known_array.shape[2]
        padded2 = np.full((max_heigth, max_width), True)
        padded2[0:h, 0:w] = known_array[0]
        known_arrays.append(torch.tensor(padded2[None]))
        #known_arrays.append(padded2[None])


        # target array to a torch tensor
        target_arrays.append(torch.tensor(target_array))

        # image file
        image_files.append(image_file)

    return pixelated_images, known_arrays, target_arrays, image_files
    #return np.asarray(pixelated_images), np.asarray(known_arrays), target_arrays, image_files


