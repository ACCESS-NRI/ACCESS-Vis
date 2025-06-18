"""
These tests check the widgets change when they should, reset correctly, etc.
It does not check that the output looks the same (you may wish to change the appearance of the widgets).
"""


def test_import_all_widgets():
    import accessvis

    _ = accessvis.Widget
    _ = accessvis.WidgetMPL
    _ = accessvis.list_widgets

    _ = accessvis.CalendarWidget
    _ = accessvis.ClockWidget
    _ = accessvis.ImageWidget
    _ = accessvis.SeasonWidget
    _ = accessvis.TextWidget


def test_list_widgets():
    import accessvis

    assert set(accessvis.list_widgets()).issuperset(
        {"CalendarWidget", "ClockWidget", "ImageWidget", "SeasonWidget", "TextWidget"}
    ), accessvis.list_widgets()


def test_CalendarWidget():
    """
    Testing that CalendarWidget changes with different kwargs, and resets correctly.
    """

    from datetime import datetime

    import accessvis
    import numpy as np

    widget = accessvis.CalendarWidget(None)
    pixels1 = widget._make_pixels()

    time2 = datetime(2025, 6, 13)
    pixels2 = widget._make_pixels(date=time2, show_year=True)

    pixels3 = widget._make_pixels(date=time2, show_year=False)

    time4 = datetime(2025, 1, 13)
    pixels4 = widget._make_pixels(date=time4, show_year=False)

    pixels5 = widget._make_pixels()

    pixels6 = accessvis.CalendarWidget(None, text_colour="blue")._make_pixels()

    assert np.array_equal(pixels1, pixels5)  # testing it resets correctly
    assert not np.array_equal(pixels1, pixels2)  # everything else should be different
    assert not np.array_equal(pixels2, pixels3)  # show year different
    assert not np.array_equal(pixels3, pixels4)  # different date
    assert not np.array_equal(pixels1, pixels6)  # text colour


def test_ClockWidget():
    """
    Testing that ClockWidget changes with different kwargs, and resets correctly.
    """

    from datetime import datetime

    import accessvis
    import numpy as np

    widget = accessvis.ClockWidget(None)
    pixels1 = widget._make_pixels()

    time2 = datetime(2025, 6, 13, 12, 30, 45)
    pixels2 = widget._make_pixels(time=time2)

    pixels3 = widget._make_pixels()

    pixels4 = accessvis.ClockWidget(None, text_colour="blue")._make_pixels(time=time2)
    pixels5 = accessvis.ClockWidget(None, background="blue")._make_pixels(time=time2)
    pixels6 = accessvis.ClockWidget(None, show_seconds=True)._make_pixels(time=time2)
    pixels7 = accessvis.ClockWidget(None, show_hours=False)._make_pixels(time=time2)
    pixels8 = accessvis.ClockWidget(None, show_minutes=False)._make_pixels(time=time2)

    time9 = datetime(2025, 6, 13, 15, 30)
    pixels9 = accessvis.ClockWidget(None)._make_pixels(time=time9)

    time10 = datetime(2025, 6, 13, 12, 45)
    pixels10 = accessvis.ClockWidget(None)._make_pixels(time=time10)

    assert np.array_equal(pixels1, pixels3), "Did not reset correctly"
    assert not np.array_equal(pixels1, pixels2), "Not showing hands"
    assert not np.array_equal(pixels2, pixels4), "Text colour not changing"
    assert not np.array_equal(pixels2, pixels5), "Background colour not changing"
    assert not np.array_equal(pixels2, pixels6), "Second hand not showing"
    assert not np.array_equal(pixels2, pixels7), "Hour hand not hidden"
    assert not np.array_equal(pixels2, pixels8), "Minute hand not hidden"
    assert not np.array_equal(pixels2, pixels9), "Hour not different"
    assert not np.array_equal(pixels2, pixels10), "Minute not different"


def test_SeasonWidget():
    """
    Testing that SeasonWidget changes with different kwargs, and resets correctly.
    """

    from datetime import datetime

    import accessvis
    import numpy as np

    widget = accessvis.SeasonWidget(None)
    pixels1 = widget._make_pixels()

    time2 = datetime(2025, 6, 13)
    pixels2 = widget._make_pixels(date=time2, show_year=True)

    pixels3 = widget._make_pixels(date=time2, show_year=False)

    time4 = datetime(2025, 1, 13)
    pixels4 = widget._make_pixels(date=time4, show_year=False)

    pixels5 = widget._make_pixels()

    pixels6 = accessvis.SeasonWidget(None, text_colour="blue")._make_pixels()
    pixels7 = accessvis.SeasonWidget(None, hemisphere="north")._make_pixels()

    assert np.array_equal(pixels1, pixels5), "Not resetting correctly"
    assert not np.array_equal(pixels1, pixels2), "Not showing arrow"
    assert not np.array_equal(pixels2, pixels3), "Not updating year"
    assert not np.array_equal(pixels3, pixels4), "Not updating date"
    assert not np.array_equal(pixels1, pixels6), "Not changing text colour"
    assert not np.array_equal(pixels1, pixels7), "Not changing hemisphere colours"


def test_TextWidget():
    """
    Testing that TextWidget changes with different kwargs, and resets correctly.
    """

    import accessvis
    import numpy as np

    widget = accessvis.TextWidget(None)
    pixels1 = widget._make_pixels()

    pixels2 = widget._make_pixels(text="Hello")

    pixels3 = widget._make_pixels()

    pixels4 = accessvis.TextWidget(None, text_colour="blue")._make_pixels(text="Hello")
    pixels5 = accessvis.TextWidget(None, background="blue")._make_pixels()

    assert np.array_equal(pixels1, pixels3), "fails to reset"
    assert not np.array_equal(pixels1, pixels2), "not showing text"
    assert not np.array_equal(pixels2, pixels4), "not changing text colour"
    assert not np.array_equal(pixels1, pixels5), "not changing background colour"


# def test_ImageWidget(): # TODO
