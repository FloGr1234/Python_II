import numpy as np

import pickle
image = np.random.rand(6,6)
bool_arr = np.ones_like(image, dtype=bool)
bool_arr[2:4,2:4] = False
target = np.where(bool_arr,0,image)

pice = image[~bool_arr]

print(image)
print(bool_arr)
print(target)
print(pice)

with open('test_set.pkl', 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    data = pickle.load(f)

print(len(data["pixelated_images"]))


