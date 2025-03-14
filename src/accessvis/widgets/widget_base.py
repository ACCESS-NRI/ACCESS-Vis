import os
from abc import ABC, abstractmethod
from functools import cached_property

import matplotlib.pyplot as plt
import numpy as np

from ..earth import Settings


class Widget(ABC):
    def __init__(self, lv, scale=0.2, offset=(0, 0)):
        self.overlay = None
        self.scale = scale  # between 0 and 1
        self.offset = offset
        self.lv = lv

    def make_overlay(self):
        self.remove()  # Only one overlay.

        pixels = self._make_pixels()
        pixels[::, ::, ::] = 0
        y, x, c = np.shape(pixels)

        vert_path = os.path.join(Settings.INSTALL_PATH, "data", "screen.vert")
        frag_path = os.path.join(Settings.INSTALL_PATH, "data", "screen.frag")

        self.overlay = self.lv.screen(
            shaders=[vert_path, frag_path], vertices=[[0, 0, 0]], texture="blank.png"
        )

        self.lv.set_uniforms(
            self.overlay["name"],
            scale=self.scale,
            offset=self.offset,
            widthToHeight=x / y,
        )
        self.overlay.texture(pixels)  # Clear texture with transparent image

    @abstractmethod
    def _make_pixels(self, **kwargs):
        pass

    def update_widget(self, **kwargs):
        if self.overlay is None:
            self.make_overlay()

        pixels = self._make_pixels(**kwargs)
        self.overlay.texture(pixels)

    def remove(self):
        if self.overlay is not None:
            self.lv.delete(self.overlay["name"])
            self.overlay = None
            self.lv = None


class WidgetMPL(Widget):
    @abstractmethod
    def _make_mpl(self) -> tuple[plt.Figure, plt.Axes]:
        # Make the basic matplotlib image
        # returns Figure, Axis
        raise NotImplementedError

    @abstractmethod
    def _update_mpl(self, fig, ax, **kwargs):
        # Update for an animation - e.g. you may want to move an arrow across time
        raise NotImplementedError

    @abstractmethod
    def _reset_mpl(self, fig, ax, **kwargs):
        # This is called after _update_mpl to reset it to the previous state.
        # e.g. remove an arrow added.
        # This should leave fig, ax as the same as when first created with _make_mpl()
        raise NotImplementedError

    @cached_property
    def _cache_mpl(self):  # only create base MPL once.
        return self._make_mpl()

    @property
    def fig(self):
        return self._cache_mpl[0]

    @property
    def ax(self):
        return self._cache_mpl[1]

    def _make_pixels(self, **kwargs):
        self._update_mpl(fig=self.fig, ax=self.ax, **kwargs)

        canvas = self.fig.canvas
        canvas.draw()
        pixels = np.asarray(canvas.buffer_rgba())

        self._reset_mpl(fig=self.fig, ax=self.ax, **kwargs)

        return pixels

    def show_mpl(self, **kwargs):
        self._update_mpl(fig=self.fig, ax=self.ax, **kwargs)
        self._reset_mpl(fig=self.fig, ax=self.ax, **kwargs)


def list_widgets():
    return [cls.__name__ for cls in Widget.__subclasses__()]
