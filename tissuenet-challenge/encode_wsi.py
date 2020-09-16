import os

import pyvips
import numpy as np
from assets.sldc import Image, Tile, TileBuilder


class PyVipsSlide(Image):
    def __init__(self, path, zoom_level=0):
        self._filepath = path
        self._slide = pyvips.Image.new_from_file(path, page=zoom_level)

    @property
    def filepath(self):
        return self._filepath

    @property
    def pyvips_slide(self):
        return self._slide

    @property
    def height(self):
        return self._slide.height

    @property
    def width(self):
        return self._slide.width

    @property
    def channels(self):
        return 3

    @property
    def np_image(self):
        raise NotImplementedError("cannot read the full image.")


class PyVipsTile(Tile):
    def __init__(self, region, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._region = region

    def np_image(self):
        region = self._region.fetch(self.abs_offset_x, self.abs_offset_y, self.width, self.height)
        return np.ndarray(
            buffer=region.write_to_memory(),
            dtype=np.uint8,
            shape=(region.height, region.width, region.bands)
        )


class PyVipsTileBuilder(TileBuilder):
    def __init__(self, slide: PyVipsSlide):
        super().__init__()
        self._region = pyvips.Region.new(slide.pyvips_slide)

    def build(self, image, offset, width, height, polygon_mask=None):
        return PyVipsTile(self._region, image, offset, width, height, polygon_mask=polygon_mask)


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

