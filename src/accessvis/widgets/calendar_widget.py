import calendar
import datetime

import matplotlib.pyplot as plt
import numpy as np

from .widget_base import WidgetMPL


class CalendarWidget(WidgetMPL):
    def __init__(self, lv, text_colour="black", **kwargs):
        """
        Create a round dial with an arrow to indicate the time of the year.

        Parameters
        ----------
        lv: lavavu.Viewer
            The viewer object to plot with.
        text_colour:
            Matplotlib compatible colour.
        scale: float
            The size of the widget, where 1.0 means it will take up the entire height of the final image.
        offset: tuple[float, float]
            The position of the widget, with (0,0) placing it in the top left, and (1,1) the bottom right.
        """
        super().__init__(lv=lv, **kwargs)
        self.text_colour = text_colour
        self.arrow = None

    def _make_mpl(self):
        """
        Creates the round dial with the months listed.
        """
        plt.rc("axes", linewidth=4)
        plt.rc("font", weight="bold")
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(5, 5))
        fig.patch.set_facecolor((0, 0, 0, 0))  # make background transparent
        ax.set_facecolor("white")  # adds a white ring around edge

        # Setting up grid
        ax.set_rticks([])
        ax.grid(False)
        ax.set_theta_zero_location("NW")
        ax.set_theta_direction(-1)

        # Label Angles
        MONTH = []
        for i in range(1, 13):
            MONTH.append(calendar.month_name[i][0])
        MONTH = np.roll(MONTH, 2)
        ANGLES = np.linspace(0.0, 2 * np.pi, 12, endpoint=False)
        ax.tick_params(axis="x", which="major", pad=12, labelcolor=self.text_colour)
        ax.set_xticks(ANGLES)
        ax.set_xticklabels(MONTH, size=20)
        ax.spines["polar"].set_color(self.text_colour)

        ax.set_ylim([0, 10])

        # Make Colours:
        ax.bar(x=0, height=10, width=np.pi * 2, color="black")
        for i in range(12):
            c = "darkorange" if i % 2 else "darkcyan"
            ax.bar(x=i * np.pi / 6, height=10, width=np.pi / 6, color=c)

        return fig, ax

    def _update_mpl(self, fig, ax, date: datetime.datetime = None, show_year=True):
        """
        Adds an arrow to point to the time of year.

        Parameters
        ----------
        date: date or datetime
            The time of year the dial should point to.
        show_year: bool
            If true, the current year is displayed below the dial.
        """
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
                7.5,
                facecolor="#fff",
                width=0.1,
                head_length=2,
                edgecolor="black",
            )  # , zorder=11, width=1)

    def _reset_mpl(self, fig, ax, **kwargs):
        """
        Removes the arrow from the figure.
        """
        fig.suptitle("")
        if self.arrow is not None:
            try:
                self.arrow.remove()
            except ValueError:
                pass  # already removed
