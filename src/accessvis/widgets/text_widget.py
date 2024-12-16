import matplotlib.pyplot as plt
from .widget_base import WidgetMPL


class TextWidget(WidgetMPL):
    def __init__(self, lv, width=300, height=50, text_colour='black', background=(0, 0, 0, 0), **kwargs):
        super().__init__(lv, **kwargs)
        self.width = width
        self.height = height
        self.text_colour = text_colour
        self.background = background

    def _make_mpl(self):
        fig, ax = plt.subplots(figsize=(self.width / 100, self.height / 100), dpi=100)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax.set_axis_off()
        fig.patch.set_facecolor(self.background)

        return fig, ax

    def _update_mpl(self, fig, ax, text='', **kwargs):
        ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=20, color=self.text_colour)

    def _reset_mpl(self, fig, ax, **kwargs):
        pass
