import os

import pyvips
import numpy as np
from assets.sldc_pyvips.adapter import PyVipsSlide


def encode(slide: PyVipsSlide, model, preproc_fn, tile_size=512, tile_overlap=0, save_to=None):
    topology = slide.tile_topology(builder, max_width=tile_size, max_height=tile_size, overlap=tile_overlap)
    en_height, en_width = topology.tile_vertical_count, topology.tile_horizontal_count
    encoded = np.zeros((en_height, en_width, model.n_features), dtype=np.float)
    for tile in topology:
        identifier = tile.identifier
        x = (identifier - 1) % en_width
        y = (identifier - 1) // en_width
        processed = preproc_fn(tile.np_image)
        out = model.forward(processed)
        encoded[y, x] = out.detach().cpu().squeeze()

    if save_to is not None:
        filename = os.path.basename(slide.filepath)
        filename = filename.rsplit(".", 1)[0] + ".npz"
        save_path = os.path.join(save_to, filename)
        np.save(save_path, encoded)
        print("save encoded into '{}'".format(save_path))

    return encoded


if __name__ == "__main__":
    pvs = PyVipsSlide("C:/data/tissuenet/C13_B156_S11.tif", zoom_level=9)
    builder = PyVipsTileBuilder(pvs)
    print(pvs.tile(builder, (0, 0), 512, 512).np_image)

