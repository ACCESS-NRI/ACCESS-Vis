from .widget_base import Widget, WidgetMPL, list_widgets
from .season_widget import SeasonWidget
from .calendar_widget import CalendarWidget
from .clock_widget import ClockWidget
from .image_widget import ImageWidget
from .text_widget import TextWidget

_ = (Widget, WidgetMPL, list_widgets, SeasonWidget) # to stop the linter complaining
_ = (CalendarWidget, ClockWidget, ImageWidget, TextWidget)