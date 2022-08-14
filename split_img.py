import os
import re

from PIL import Image

out_folder = "./splits/"
im_path = "./kitty_test.png"


def split(image, n_rows, n_cols):
    im_w, im_h = image.size

    # Can be used later if I want non-uniform splits
    """ min_dimension = min(im_w, im_h)
    max_dimension = max(im_w, im_h) """

    tile_h = int(im_h / n_rows)
    tile_w = int(im_w / n_cols)

    orig_name, ext = os.path.splitext(im_path)
    out_path = out_folder + "/" + orig_name + "/"

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for i in range(n_cols):
        for j in range(n_rows):
            # crop_coords is the left, top, right and bottom position values for the tile that is to be cut out
            crop_coords = (i * tile_w, j * tile_h, i * tile_w + tile_w, j * tile_h + tile_h)
            tile = image.crop(crop_coords)
            file_path = out_path + orig_name + "_" + str(i) + "_" + str(j) + ext
            tile.save(file_path)

if __name__ == "__main__":
    im = Image.open(im_path)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    split(im, 3, 3)
    
