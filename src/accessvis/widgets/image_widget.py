import matplotlib.pyplot as plt

from .widget_base import Widget


class ImageWidget(Widget):
    def __init__(self, lv, file_path: str, **kwargs):
        """
        Display a static image.

        Parameters
        ----------
        lv: lavavu.Viewer
            The viewer object to plot with.
        file_path: str
            Path to the image to be displayed.
        scale: float
            The size of the widget, where 1.0 means it will take up the entire height of the final image.
        offset: tuple[float, float]
            The position of the widget, with (0,0) placing it in the top left, and (1,1) the bottom right.
        """
        super().__init__(lv, **kwargs)
        self.file_path = file_path

    def _make_pixels(self, *args, **kwargs):
        """
        Reads the image from file.
        """
        return plt.imread(self.file_path)
