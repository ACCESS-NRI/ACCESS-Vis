# Ensure all textures are loaded and generated
import datetime

import accessvis


def generate(tex, resrange=range(1, 5), month=None):
    for resolution in resrange:
        accessvis.set_resolution(resolution)
        if month is None:
            lv = accessvis.plot_earth(texture=tex, blendtex=False)
            lv.render()
            lv.image(f"test_image_{tex}_res={resolution}.png")
        else:
            when = datetime.datetime(day=1, month=month, year=2025, hour=12, minute=0)
            lv = accessvis.plot_earth(texture=tex, when=when, blendtex=False)
            lv.render()
            lv.image(f"test_image_{tex}_res={resolution}_month={month}.png")


# Relief
generate("relief")

# Plot each month for bluemarble
for m in range(1, 13):
    generate("bluemarble", month=m)
