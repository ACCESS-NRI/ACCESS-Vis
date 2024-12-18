import datetime

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from .widget_base import WidgetMPL


class SeasonWidget(WidgetMPL):
    def __init__(self, lv, text_colour="black", hemisphere="south", **kwargs):
        super().__init__(lv=lv, **kwargs)
        self.text_colour = text_colour
        self.hemisphere = hemisphere
        self.arrow = None

    def _make_mpl(self):
        plt.rc("axes", linewidth=4)
        plt.rc("font", weight="bold")
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(5, 5))
        fig.patch.set_facecolor((0, 0, 0, 0))  # make background transparent
        ax.set_facecolor("white")  # adds a white ring around edge

        # Setting up grid
        ax.set_rticks([])
        ax.grid(False)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)

        # Label Angles
        if self.hemisphere == "south":
            MONTH = ["Sum", "Aut", "Win", "Spr"]
            cmap = LinearSegmentedColormap.from_list(
                "custom_gradient", ["orange", "black", "blue", "black", "orange"]
            )
        else:
            MONTH = ["Win", "Spr", "Sum", "Aut"]
            cmap = LinearSegmentedColormap.from_list(
                "custom_gradient", ["blue", "black", "orange", "black", "blue"]
            )

        ANGLES = np.linspace(np.pi / 4, 2 * np.pi + np.pi / 4, 4, endpoint=False)
        ax.tick_params(axis="x", which="major", pad=12, labelcolor=self.text_colour)
        ax.set_xticks(ANGLES)
        ax.set_xticklabels(MONTH, size=20)
        ax.spines["polar"].set_color(self.text_colour)

        # Colour background based on time of year:
        dec22_doy = datetime.date(2001, 12, 22).timetuple().tm_yday - 1
        dec22 = np.pi * 2.0 * dec22_doy / 365  # summer solstice
        r = np.linspace(0, 10, 5)
        theta = np.linspace(dec22, dec22 + 2 * np.pi, 500)
        R, T = np.meshgrid(r, theta)
        ax.pcolormesh(T, R, T, cmap=cmap, shading="gouraud")

        return fig, ax

    def _update_mpl(self, fig, ax, date: datetime.datetime = None, show_year=True):
        if show_year and date is not None:
            title = str(date.year)
        else:
            title = ""
        fig.suptitle(
            title, fontsize=20, fontweight="bold", y=0.08, color=self.text_colour
        )

        if date is None:
            return
        else:
            day_of_year = date.timetuple().tm_yday - 1
            position = day_of_year / 365.0 * np.pi * 2.0
            self.arrow = ax.arrow(
                position,
                0,
                0,
                8.5,
                facecolor="#fff",
                width=0.1,
                head_length=2,
                edgecolor="black",
            )  # , zorder=11, width=1)

    def _reset_mpl(self, fig, ax, **kwargs):
        fig.suptitle("")
        if self.arrow is not None:
            self.arrow.remove()
