from .calendar_widget import CalendarWidget
from .clock_widget import ClockWidget
from .image_widget import ImageWidget
from .season_widget import SeasonWidget
from .text_widget import TextWidget
from .widget_base import Widget, WidgetMPL, list_widgets

_ = (Widget, WidgetMPL, list_widgets, SeasonWidget)  # to stop the linter complaining
_ = (CalendarWidget, ClockWidget, ImageWidget, TextWidget)
