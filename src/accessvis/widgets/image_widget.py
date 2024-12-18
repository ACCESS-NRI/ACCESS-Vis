import numpy as np
import imageio

from .widget_base import Widget


class ImageWidget(Widget):
    def __init__(self, lv, file_path: str, **kwargs):
        super().__init__(lv, **kwargs)
        self.file_path = file_path

    def _make_pixels(self, *args, **kwargs):
        img = imageio.imread(self.file_path)
        array = np.asarray(img)
        return array
