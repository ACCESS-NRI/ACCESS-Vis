import numpy as np
import matplotlib.pyplot as plt
import datetime

from .widget_base import WidgetMPL

class SeasonWidget(WidgetMPL):
    # TODO: Change colours so summer colour starts in december, not jan.
    # Consider adding in astronomical seasons (solstices).
    def __init__(self, lv, text_colour='black', hemisphere='south', **kwargs):
        super().__init__(lv=lv, **kwargs)
        self.text_colour = text_colour
        self.hemisphere = hemisphere
        self.arrow = None

    def _make_mpl(self):
        plt.rc('axes', linewidth=4)
        plt.rc('font', weight='bold')
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(5, 5))
        fig.patch.set_facecolor((0, 0, 0, 0))  # make background transparrent
        ax.set_facecolor('white')  # adds a white ring around edge

        # Setting up grid
        ax.set_rticks([])
        ax.grid(False)
        ax.set_theta_zero_location('NW')
        ax.set_theta_direction(-1)

        # Label Angles
        MONTH = ['Jan', 'Apr', 'Jul', 'Oct']
        ANGLES = np.linspace(0.0, 2 * np.pi, 4, endpoint=False)
        ax.tick_params(axis='x', which='major', pad=12, labelcolor=self.text_colour)
        ax.set_xticks(ANGLES)
        ax.set_xticklabels(MONTH, size=20)
        ax.spines['polar'].set_color(self.text_colour)

        # Make Colours:
        ax.bar(x=0, height=10, width=np.pi * 2, color='black')
        ax.bar(x=5 * np.pi / 4, height=10, width=np.pi / 2,
               color='darkcyan' if self.hemisphere == 'south' else 'darkorange')  # Southern Winter
        ax.bar(x=np.pi / 4, height=10, width=np.pi / 2,
               color='darkorange' if self.hemisphere == 'south' else 'darkcyan')  # Southern Summer

        return fig, ax

    def _update_mpl(self, fig, ax, date: datetime.datetime = None, show_year=True):
        if show_year and date is not None:
            title = str(date.year)
        else:
            title = ''
        fig.suptitle(title, fontsize=20, fontweight='bold', y=0.08, color=self.text_colour)

        if date is None:
            return
        else:
            day_of_year = date.timetuple().tm_yday - 1
            position = day_of_year / 365. * np.pi * 2.0
            self.arrow = ax.arrow(position, 0, 0, 8.5, facecolor='#fff', width=0.1, head_length=2,
                                  edgecolor="black")  # , zorder=11, width=1)

    def _reset_mpl(self, fig, ax, **kwargs):
        fig.suptitle('')
        if self.arrow is not None:
            self.arrow.remove()
