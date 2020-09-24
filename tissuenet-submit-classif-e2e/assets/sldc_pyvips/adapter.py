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
