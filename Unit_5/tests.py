import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from a3_ex1 import RandomImagePixelationDataset
from a3_ex2 import stack_with_padding



if False:  #Testes a3_ex1
    # image_dir = f"C:\\Users\\flogr\OneDrive - Johannes Kepler Universität Linz\\JKU_Master\\Python_II\\Unit_5"
    image_dir = r"C:\Users\flogr\OneDrive - Johannes Kepler Universität Linz\JKU_Master\Python_II\Unit_1\TestOut"
    ds = RandomImagePixelationDataset(
        image_dir,
        width_range=(4, 500),
        height_range=(4, 500),
        size_range=(30, 30)
    )
    for pixelated_image, known_array, target_array, image_file in ds:
        fig, axes = plt.subplots(ncols=3)
        # pixelated_image, known_array, target_array, image_file = ds[3]
        axes[0].imshow(pixelated_image[0], cmap="gray", vmin=0, vmax=255)
        axes[0].set_title("pixelated_image")
        axes[1].imshow(known_array[0], cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("known_array")
        axes[2].imshow(target_array[0], cmap="gray", vmin=0, vmax=255)
        axes[2].set_title("target_array")
        fig.suptitle(image_file)
        fig.tight_layout()
    plt.show()
    # """



if True: # Tests a3_ex2
    ds = RandomImagePixelationDataset(
            r"C:\Users\flogr\OneDrive - Johannes Kepler Universität Linz\JKU_Master\Python_II\Unit_5\Test_bilder",
            width_range=(50, 300),
            height_range=(50, 300),
            size_range=(10, 50)
        )
    dl = DataLoader(ds, batch_size=3, shuffle=False, collate_fn=stack_with_padding)
    for (stacked_pixelated_images, stacked_known_arrays, target_arrays, image_files) in dl:

        fig, axes = plt.subplots(nrows=dl.batch_size, ncols=3)
        for i in range(dl.batch_size):
            axes[i, 0].imshow(stacked_pixelated_images[i][0], cmap="gray", vmin=0, vmax=255)
            axes[i, 1].imshow(stacked_known_arrays[i][0], cmap="gray", vmin=0, vmax=1)
            axes[i, 2].imshow(target_arrays[i][0], cmap="gray", vmin=0, vmax=255)
        fig.tight_layout()
        plt.show()

