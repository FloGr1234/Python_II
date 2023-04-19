import os
import shutil
from PIL import Image, ImageStat
import PIL
import glob
import hashlib


def validate_images(input_dir: str, output_dir: str, log_file: str, formatter: str = "07d"):
    log_file = log_file+".txt"
    input_dir = os.path.abspath(input_dir)
    if not os.path.isdir(input_dir):
        raise ValueError(f"'{input_dir}' must be an existing directory")

    # create a new folder "output_dir" if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # to clear the old log_file
    f = open(log_file, 'w')
    f.close()

    # find all files in beginning at the given input
    found_files = glob.glob(input_dir + "\\**\\*.*", recursive=True)
    found_files.sort()
    copied_files = []

    num_copiedfiles = 0
    for imag_path in found_files:
        imag_size = os.path.getsize(imag_path)  # get image size
        head, tail = os.path.split(imag_path)

        file_name, ext = os.path.splitext(tail)
        file_basename = os.path.basename(imag_path)

        try:
            with Image.open(imag_path) as im:
                no_image = False
                check_4 = False

                # check the couler scale
                if im.mode != "RGB" and (im.size[0] < 100 or im.size[1] < 100):
                    check_4 = True

                # Varianz variante
                stat = ImageStat.Stat(im)
                variance = stat.var[0]

                # create Hashcode
                hasher = hashlib.sha256()
                data = im.tobytes()
                hasher.update(data)
                img_hash = hasher.hexdigest()

        except PIL.UnidentifiedImageError as ex:
            no_image = True

        # Create new filename and the new path
        formats = "{:" + formatter + "}"
        new_name = formats.format(num_copiedfiles) + ".jpg"  # .jpg for all copied files
        new_name_path = os.path.join(output_dir, new_name)

        if ext not in {".jpg", ".JPG", ".jpeg", ".JPEG"}:  # 1.
            print("Error 1: wrong file attribute")
            with open(log_file, 'a') as f:
                f.write(f"{file_basename},1\n")

        elif imag_size > 250000:  # 2.
            print("Error 2: file is to big")
            with open(log_file, 'a') as f:
                f.write(f"{file_basename},2\n")

        elif no_image:      # 3.
            print("Error 3: this file is not read as image")
            with open(log_file, 'a') as f:
                f.write(f"{file_basename},3\n")

        elif check_4:       # 4.
            print("Error 4: less Pixel size or wrong colour scale")
            with open(log_file, 'a') as f:
                f.write(f"{file_basename},4\n")

        elif variance <= 0:  # 5.
            print("ERROR 5: The Pixelvariance is 0")
            with open(log_file, 'a') as f:
                f.write(f"{file_basename},5\n")

        elif os.path.exists(new_name_path) or img_hash in copied_files:  # 6.
            print("ERROR 6: the file already exist")
            with open(log_file, 'a') as f:
                f.write(f"{file_basename},6\n")

        else:  # Copy files
            shutil.copy(imag_path, output_dir)
            os.rename(os.path.join(output_dir, tail), os.path.join(output_dir, new_name))
            copied_files.append(img_hash)
            num_copiedfiles += 1
    return num_copiedfiles



if __name__ == "__main__":
    path_in = os.path.join(os.getcwd(), "Test_errors")
    path_out = os.path.join(os.getcwd(), "TestOut")
    print(validate_images(path_in, path_out, "log_file", "03d"))
