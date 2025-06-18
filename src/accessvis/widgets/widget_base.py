import os
from abc import ABC, abstractmethod
from functools import cached_property

import matplotlib.pyplot as plt
import numpy as np

from ..earth import Settings


class Widget(ABC):
    def __init__(self, lv, scale=0.2, offset=(0, 0)):
        """
        This is the base class for creating a "Widget", an image overlaid on top of the visualisation.

        Parameters
        ----------
        lv: lavavu.Viewer
            The viewer object to plot with
        scale: float
            The size of the widget, where 1.0 means it will take up the entire height of the final image.
        offset: tuple[float, float]
            The position of the widget, with (0,0) placing it in the top left, and (1,1) the bottom right.
        """

        self.overlay = None
        self.scale = scale  # between 0 and 1
        self.offset = offset
        self.lv = lv

    def make_overlay(self):
        """
        Adds a blank overlay to the lavavu object.
        """
        self.remove()  # Only one overlay.

        pixels = self._make_pixels()
        pixels[::, ::, ::] = 0
        y, x, c = np.shape(pixels)

        vert_path = os.path.join(Settings.INSTALL_PATH, "data", "screen.vert")
        frag_path = os.path.join(Settings.INSTALL_PATH, "data", "screen.frag")

        self.overlay = self.lv.screen(
            shaders=[vert_path, frag_path],
            vertices=[[0, 0, 0]],
            texture="blank.png",
            fliptexture=True,
        )

        self.lv.set_uniforms(
            self.overlay["name"],
            scale=self.scale,
            offset=self.offset,
            widthToHeight=x / y,
        )
        self.overlay.texture(pixels)  # Clear texture with transparent image

    @abstractmethod
    def _make_pixels(self, **kwargs) -> np.ndarray:
        """
        Generate the image to be displayed.

        Parameters
        ----------
        **kwargs:
            Arbitrary arguments may be provided to dynamically generate the image (e.g. date).
            These same arguments must be passed when the update_widget() method is called.
            See the relevant subclass for required arguments.

        Returns
        -------
        np.ndarray: An array of RGB/RGBA values for the displayed image.
        """
        pass

    def update_widget(self, **kwargs):
        """
        Updates the existing overlay with new pixel values.

        Parameters
        ----------
        **kwargs:
            See the _make_pixels() or _update_mpl() method of the relevant subclass for required arguments.
        """

        if self.overlay is None:
            self.make_overlay()

        pixels = self._make_pixels(**kwargs)
        self.overlay.texture(pixels)

    def remove(self):
        """
        Removes the overlay from the lavavu object
        """
        if self.overlay is not None:
            self.lv.delete(self.overlay["name"])
            self.overlay = None
            self.lv = None


class WidgetMPL(Widget):
    """
    This is the base class for creating a Widget using a Matplotlib.
    _make_mpl() initialises up the mpl figure.
    _update_mpl() Updates mpl figure for animations.
    _reset_mpl() resets the mpl figure to the initial state.
    """

    @abstractmethod
    def _make_mpl(self) -> tuple[plt.Figure, plt.Axes]:
        """
        Generate and set up the basic matplotlib figure.
        Set unchanging parameters such as axes and titles. Do not add data which you are animating.

        Returns
        -------
        matplotlib.figure.Figure
        matplotlib.figure.Axes
        """

        raise NotImplementedError

    @abstractmethod
    def _update_mpl(self, fig, ax, **kwargs):
        """
        Update the mpl Figure and Axes objects to make an animation.

        Parameters
        ----------
        fig: matplotlib.figure.Figure
        ax: matplotlib.figure.Axes
        **kwargs:
            Arbitrary arguments may be provided to dynamically generate the image (e.g. date).
            These same arguments must be passed when the update_widget() method is called.
            E.g. you may wish to show data at a given time.

            See the relevant subclass for required arguments.
        """
        raise NotImplementedError

    @abstractmethod
    def _reset_mpl(self, fig, ax, **kwargs):
        """
        Reset the mpl Figure to the initial state.
        This is done so it may be updated for the next frame in the animation.
        E.g. if data is added for a given date, it would be removed here.
        This should leave fig, ax as the same as when first created with _make_mpl()

        Parameters
        ----------
        fig: matplotlib.figure.Figure
        ax: matplotlib.figure.Axes
        **kwargs:
            Arbitrary arguments may be provided to help reset the image.
            These same arguments must be passed when the update_widget() method is called.

            See the relevant subclass for required arguments.
        """

        raise NotImplementedError

    @cached_property
    def _cache_mpl(self) -> tuple[plt.Figure, plt.Axes]:
        """
        Cache the matplotlib figure so it only creates it once.
        """
        return self._make_mpl()

    @property
    def fig(self) -> plt.Figure:
        return self._cache_mpl[0]

    @property
    def ax(self) -> plt.Axes:
        return self._cache_mpl[1]

    def _make_pixels(self, **kwargs):
        """
        Generate the image to be displayed.

        Parameters
        ----------
        **kwargs:
            See the _update_mpl() method in the relevant subclass.

        Returns
        -------
        np.ndarray: An array of RGB/RGBA values for the displayed image.
        """

        self._update_mpl(fig=self.fig, ax=self.ax, **kwargs)

        canvas = self.fig.canvas
        canvas.draw()
        pixels = np.asarray(canvas.buffer_rgba(), copy=True)

        self._reset_mpl(fig=self.fig, ax=self.ax, **kwargs)

        return pixels

    def show_mpl(self, **kwargs):
        """
        Displays the figure in a Jupyter notebook.
        Useful for debugging image creation.
        """
        self._update_mpl(fig=self.fig, ax=self.ax, **kwargs)
        self._reset_mpl(fig=self.fig, ax=self.ax, **kwargs)


def list_widgets():
    """
    Lists the Widgets which are available to be used.
    Use to discover other widgets you may consider using.

    Returns
    -------
    List[String]: The name of the classes.
    """

    def get_subclasses(cls):
        # From https://stackoverflow.com/questions/3862310/how-to-find-all-the-subclasses-of-a-class-given-its-name
        for subclass in cls.__subclasses__():
            yield from get_subclasses(subclass)
            yield subclass.__name__

    return [Widget.__name__] + list(get_subclasses(Widget))
