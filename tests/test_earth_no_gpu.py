"""
These tests cover functions in earth.py which do not require a GPU/lavavu.
"""


def test_array_to_rgba():
    import accessvis
    import numpy as np

    data1 = np.array([[i + j for i in range(20)] for j in range(10)])

    # Grays colourmap is monotonic, data1 is monotonic, out1 should be monotonic.
    out1 = accessvis.array_to_rgba(values=data1, colourmap="Greys")
    assert out1.shape == (10, 20, 4)
    assert np.all(np.diff(out1[:, :, 0], axis=0) > 0)
    assert np.all(np.diff(out1[:, :, 1], axis=0) > 0)
    assert np.all(np.diff(out1[:, :, 2], axis=0) > 0)
    assert np.all(np.diff(out1[:, :, 0], axis=1) > 0)
    assert np.all(np.diff(out1[:, :, 1], axis=1) > 0)
    assert np.all(np.diff(out1[:, :, 2], axis=1) > 0)

    assert np.all(out1[:, :, 3] == 255), "Should not be transparent"

    # opacity
    out2 = accessvis.array_to_rgba(values=data1, colourmap="Greys", opacity=0.5)
    assert np.all(out2[:, :, 3] == 127), "Should be transparent"

    # opacity map
    out3 = accessvis.array_to_rgba(values=data1, colourmap="Greys", opacitymap=True)
    assert np.all(np.diff(out3[:, :, 3], axis=0) > 0)
    assert np.all(np.diff(out3[:, :, 3], axis=1) > 0)

    om1 = np.zeros_like(data1)
    om1[:5] = 1
    out4 = accessvis.array_to_rgba(values=data1, colourmap="Greys", opacitymap=om1)
    assert np.all(out4[:5, :, 3] == 255)
    assert np.all(out4[5:, :, 3] == 0)

    # min/max
    data2 = np.ones((10, 10))
    data2[:5] = 10
    out5 = accessvis.array_to_rgba(
        values=data2, colourmap="Greys", minimum=2, maximum=8
    )
    assert np.all(out5[:5, :, :3] == 0)  # all values should be equal
    assert np.all(out5[5:, :, :3] == 255)  # all values should be equal

    # Test colourmap changes output
    out6 = accessvis.array_to_rgba(values=data1, colourmap="viridis")
    assert np.any(out6 != out1), "Does not change with different colourmaps"


def test_normalise_array():
    import accessvis
    import numpy as np

    arr1 = accessvis.normalise_array(np.array([1, 2, 3]))
    exp1 = [0, 0.5, 1]
    assert np.allclose(arr1, exp1)

    arr2 = accessvis.normalise_array(np.array([2, 3, 4]), minimum=1, maximum=5)
    exp2 = [0.25, 0.5, 0.75]
    assert np.allclose(arr2, exp2)

    arr3 = accessvis.normalise_array(np.array([1, 3, 5]), minimum=2, maximum=4)
    exp3 = [0, 0.5, 1]
    assert np.allclose(arr3, exp3)


def test_lonlat_vector_to_north():
    """
    At the equator, all point directly upwards.
    Should have length 1.
    """
    import accessvis
    import numpy as np

    vec1 = accessvis.lonlat_vector_to_north(lat=0.0, lon=0.0)
    vec2 = accessvis.lonlat_vector_to_north(lat=0, lon=90)
    vec3 = [0, 1, 0]

    assert np.allclose(vec1, vec2)
    assert np.allclose(vec1, vec3)

    vec4 = accessvis.lonlat_vector_to_north(lat=47.3, lon=88.1)
    assert np.allclose(np.linalg.norm(vec4), 1), "doesn't have length 1."


def test_lonlat_vector_to_east():
    import accessvis
    import numpy as np

    vec1 = accessvis.lonlat_vector_to_east(lat=0.0, lon=0.0)
    vec2 = [1, 0, 0]
    assert np.allclose(vec1, vec2)

    vec3 = accessvis.lonlat_vector_to_east(lat=0, lon=-90)
    vec4 = [0, 0, 1]
    assert np.allclose(vec3, vec4)

    norm_vec = accessvis.lonlat_vector_to_east(lat=87.3, lon=-25.1)
    assert np.allclose(np.linalg.norm(norm_vec), 1), "doesn't have length 1."


def test_lonlat_normal_vector():
    import accessvis
    import numpy as np

    vec1 = accessvis.lonlat_normal_vector(lat=0.0, lon=0.0)
    vec2 = [0, 0, 1]
    assert np.allclose(vec1, vec2)

    vec3 = accessvis.lonlat_normal_vector(lat=0, lon=90)
    vec4 = [1, 0, 0]
    assert np.allclose(vec3, vec4)

    vec5 = accessvis.lonlat_normal_vector(lat=90, lon=47.2)
    vec6 = [0, 1, 0]
    assert np.allclose(vec5, vec6)

    vec7 = accessvis.lonlat_normal_vector(lat=-90, lon=47.2)
    vec8 = [0, -1, 0]
    assert np.allclose(vec7, vec8)

    norm_vec = accessvis.lonlat_normal_vector(lat=-65.1, lon=-48.1)
    assert np.allclose(np.linalg.norm(norm_vec), 1), "doesn't have length 1."


def test_normalise():
    import accessvis
    import numpy as np

    vec1 = accessvis.normalise([1, -2, 3])
    mag = 14**0.5
    vec2 = [1 / mag, -2 / mag, 3 / mag]
    assert np.allclose(vec1, vec2)


def test_magnitude():
    import accessvis
    import numpy as np

    mag = accessvis.magnitude([1, -2, 3])
    assert np.allclose(mag, 14**0.5)


def test_vec_rotate():
    import accessvis
    import numpy as np

    vec1 = np.array([1, 0, 0])
    vec2 = accessvis.vec_rotate(vec1, np.pi / 2, np.array([0, 0, 1]))
    assert np.allclose(vec2, [0, 1, 0])

    vec3 = np.array([0, 1, 0])
    vec4 = accessvis.vec_rotate(vec3, np.pi / 2, np.array([1, 0, 0]))
    assert np.allclose(vec4, [0, 0, 1])

    vec5 = np.array([0, 0, 1])
    vec6 = accessvis.vec_rotate(vec5, np.pi / 2, np.array([1, 0, 0]))
    assert np.allclose(vec6, [0, -1, 0])

    vec7 = np.array([0, 0, 1])
    vec8 = accessvis.vec_rotate(vec7, 7982.4, np.array([35, 0.878, 10.7]))
    assert np.allclose(np.linalg.norm(vec8), 1), "Magnutude changes"


def test_crop_img_uv_numpy():
    """
    I make an array with half 1, half 2.
    Various crops will have different 1s and 2s.
    """
    import accessvis
    import numpy as np

    arr = np.zeros((10, 20))
    arr[:, :10] = 1
    arr[:, 10:] = 2

    cropbox1 = (0, 0), (0.5, 1)  # left side
    out1 = accessvis.crop_img_uv(img=arr, cropbox=cropbox1)
    assert out1.shape == (10, 10), f"Left shape {out1.shape}"
    assert np.all(out1 == 1), "Left sum"

    cropbox2 = (0.25, 0.25), (0.75, 0.75)  # Middle
    out2 = accessvis.crop_img_uv(img=arr, cropbox=cropbox2)
    assert out2.shape == (5, 10), f"Middle shape {out2.shape}"
    assert np.all(out2[:, :5] == 1), "middle 1"
    assert np.all(out2[:, 5:] == 2), "middle 2"

    cropbox3 = (-0.25, 0), (0.25, 1)  # overflow left
    out3 = accessvis.crop_img_uv(img=arr, cropbox=cropbox3)
    assert out3.shape == (10, 10), "overflow left shape"
    assert np.all(out3[:, :5] == 2), "overflow left 1"
    assert np.all(out3[:, 5:] == 1), "overflow left 2"

    cropbox4 = (0.75, 0), (1.25, 1)  # overflow right
    out4 = accessvis.crop_img_uv(img=arr, cropbox=cropbox4)
    assert out4.shape == (10, 10), "overflow right shape"
    assert np.all(out4[:, :5] == 2), "overflow right 1"
    assert np.all(out4[:, 5:] == 1), "overflow right 2"


def test_crop_img_uv_pil():
    """
    I make a PIL image with half black, half white.
    Various crops will have different black/white.
    """
    import accessvis
    import numpy as np
    from PIL import Image

    arr = [
        [[0, 0, 0] if j < 10 else [255, 255, 255] for j in range(20)] for i in range(10)
    ]
    arr = np.uint8(arr)
    image = Image.fromarray(arr)

    cropbox1 = (0, 0), (0.5, 1)  # left side
    out1 = np.array(accessvis.crop_img_uv(img=image, cropbox=cropbox1))
    assert out1.shape == (10, 10, 3), f"Left shape {out1.shape}"
    assert np.all(out1 == 0), "Left sum"

    cropbox2 = (0.25, 0.25), (0.75, 0.75)  # Middle
    out2 = np.array(accessvis.crop_img_uv(img=image, cropbox=cropbox2))
    assert out2.shape == (5, 10, 3), f"Middle shape {out2.shape}"
    assert np.all(out2[:, :5] == 0), "middle 1"
    assert np.all(out2[:, 5:] == 255), "middle 2"

    cropbox3 = (-0.25, 0), (0.25, 1)  # overflow left
    out3 = np.array(accessvis.crop_img_uv(img=image, cropbox=cropbox3))
    assert out3.shape == (10, 10, 3), "overflow left shape"
    assert np.all(out3[:, :5] == 255), "overflow left 1"
    assert np.all(out3[:, 5:] == 0), "overflow left 2"

    cropbox4 = (0.75, 0), (1.25, 1)  # overflow right
    out4 = np.array(accessvis.crop_img_uv(img=image, cropbox=cropbox4))
    assert out4.shape == (10, 10, 3), "overflow right shape"
    assert np.all(out4[:, :5] == 255), "overflow right 1"
    assert np.all(out4[:, 5:] == 0), "overflow right 2"


def test_crop_img_lat_lon():
    """
    I make an array with half 1, half 2.
    Various crops will have different 1s and 2s.
    """
    import accessvis
    import numpy as np

    arr = np.zeros((10, 20))
    arr[:, :10] = 1
    arr[:, 10:] = 2

    tl1, br1 = (-180, 90), (0, -90)  # left side
    out1 = accessvis.crop_img_lon_lat(img=arr, top_left=tl1, bottom_right=br1)
    assert out1.shape == (10, 10), f"Left shape {out1.shape}"
    assert np.all(out1 == 1), "Left sum"

    tl2, br2 = (-90, 45), (90, -45)  # Middle
    out2 = accessvis.crop_img_lon_lat(img=arr, top_left=tl2, bottom_right=br2)
    assert out2.shape == (5, 10), f"Middle shape {out2.shape}"
    assert np.all(out2[:, :5] == 1), "middle 1"
    assert np.all(out2[:, 5:] == 2), "middle 2"

    tl3, br3 = (-270, 90), (-90, -90)  # overflow left
    out3 = accessvis.crop_img_lon_lat(img=arr, top_left=tl3, bottom_right=br3)
    assert out3.shape == (10, 10), "overflow left shape"
    assert np.all(out3[:, :5] == 2), "overflow left 1"
    assert np.all(out3[:, 5:] == 1), "overflow left 2"

    tl4, br4 = (90, 90), (270, -90)  # overflow right
    out4 = accessvis.crop_img_lon_lat(img=arr, top_left=tl4, bottom_right=br4)
    assert out4.shape == (10, 10), "overflow right shape"
    assert np.all(out4[:, :5] == 2), "overflow right 1"
    assert np.all(out4[:, 5:] == 1), "overflow right 2"


def test_latlon_to_uv():
    import accessvis

    assert accessvis.lonlat_to_uv(lat=0, lon=0) == (0.5, 0.5)
    assert accessvis.lonlat_to_uv(lat=90, lon=-180) == (0, 0)
    assert accessvis.lonlat_to_uv(lat=-90, lon=180) == (1, 1)


def test_uv_to_pixel():
    import accessvis

    assert accessvis.uv_to_pixel(u=0.1, v=0.8, width=1017, height=8256) == (101, 6604)


def test_latlon_to_pixel():
    import accessvis

    lon = 0.1 * 360 - 180
    lat = (1 - 0.8) * 180 - 90  # same as test_uv_to_pixel
    assert accessvis.lonlat_to_pixel(lon=lon, lat=lat, width=1017, height=8256) == (
        101,
        6604,
    )


def test_lonlat_to_3D_true():
    import accessvis
    import numpy as np

    # With Flattening
    R = 6.371
    assert np.allclose(
        accessvis.lonlat_to_3D_true(lon=0, lat=90, flattening=0), [0, R, 0]
    ), "North"
    assert np.allclose(
        accessvis.lonlat_to_3D_true(lon=0, lat=-90, flattening=0), [0, -R, 0]
    ), "South"
    assert np.allclose(
        accessvis.lonlat_to_3D_true(lon=0, lat=0, flattening=0), [0, 0, R]
    ), "Equator1"
    assert np.allclose(
        accessvis.lonlat_to_3D_true(lon=180, lat=0, flattening=0), [0, 0, -R]
    ), "Equator2"
    assert np.allclose(
        accessvis.lonlat_to_3D_true(lon=90, lat=0, flattening=0), [R, 0, 0]
    ), "Equator3"
    assert np.allclose(
        accessvis.lonlat_to_3D_true(lon=270, lat=0, flattening=0), [-R, 0, 0]
    ), "Equator4"

    # Altitude
    assert np.allclose(
        accessvis.lonlat_to_3D_true(lon=270, lat=0, alt=1, flattening=0), [-R - 1, 0, 0]
    ), "Equator4"
    assert np.allclose(
        accessvis.lonlat_to_3D_true(lon=0, lat=90, alt=1, flattening=0), [0, R + 1, 0]
    ), "North"
    assert np.allclose(
        accessvis.lonlat_to_3D_true(lon=0, lat=-90, alt=1, flattening=0), [0, -R - 1, 0]
    ), "South"
    assert np.allclose(
        accessvis.lonlat_to_3D_true(lon=0, lat=0, alt=1, flattening=0), [0, 0, R + 1]
    ), "Equator1"
    assert np.allclose(
        accessvis.lonlat_to_3D_true(lon=180, lat=0, alt=1, flattening=0), [0, 0, -R - 1]
    ), "Equator2"
    assert np.allclose(
        accessvis.lonlat_to_3D_true(lon=90, lat=0, alt=1, flattening=0), [R + 1, 0, 0]
    ), "Equator3"
    assert np.allclose(
        accessvis.lonlat_to_3D_true(lon=270, lat=0, alt=1, flattening=0), [-R - 1, 0, 0]
    ), "Equator4"

    v1 = accessvis.lonlat_to_3D_true(lon=85.1, lat=63.48)
    v2 = accessvis.lonlat_to_3D_true(lon=85.1, lat=63.48, alt=1)
    v3 = accessvis.lonlat_to_3D_true(lon=85.1, lat=63.48, alt=2.5)

    assert np.allclose(np.linalg.norm(v2 - v1), 1), "Altitude==1 is incorrect"
    assert np.allclose(np.linalg.norm(v3 - v1), 2.5), "Altitude==2.5 is incorrect"

    # Testing it works with array inputs
    # Note: there is some code which depends on this being shaped this way. E.g. plot_vectors_xr
    # alt 1 unit increments
    arr1 = accessvis.lonlat_to_3D_true(
        lon=23.7, lat=84.3, alt=np.array(range(10)), flattening=0
    )
    norm1 = np.linalg.norm(arr1[:, 1:] - arr1[:, :-1], axis=0)
    assert arr1.shape == (3, 10)
    assert norm1.shape == (9,)
    assert np.allclose(norm1, 1)

    # lon 1 deg increments
    arr2 = accessvis.lonlat_to_3D_true(
        lon=np.array(range(-10, 10)), lat=84.3, alt=0, flattening=0
    )
    norm2 = np.linalg.norm(arr2[:, 1:] - arr2[:, :-1], axis=0)
    assert arr2.shape == (3, 20)
    assert norm2.shape == (19,)
    assert np.allclose(norm2, norm2[0])

    # lat 1 deg increments
    arr3 = accessvis.lonlat_to_3D_true(
        lon=46.4, lat=0.67 + np.array(range(-10, 10)), alt=0, flattening=0
    )
    norm3 = np.linalg.norm(arr3[:, 1:] - arr3[:, :-1], axis=0)
    assert arr3.shape == (3, 20)
    assert norm3.shape == (19,)
    assert np.allclose(norm3, norm3[0])

    # All are arrays
    arr3 = accessvis.lonlat_to_3D_true(
        lon=np.array(range(-10, 10)),
        lat=0.67 + np.array(range(-10, 10)),
        alt=np.array(range(20)),
    )
    assert arr3.shape == (3, 20)


def test_latlon_to_3D():
    import accessvis
    import numpy as np

    R = 6.371
    assert np.allclose(accessvis.latlon_to_3D(lon=0, lat=90), [0, R, 0]), "North"
    assert np.allclose(accessvis.latlon_to_3D(lon=0, lat=-90), [0, -R, 0]), "South"
    assert np.allclose(accessvis.latlon_to_3D(lon=0, lat=0), [0, 0, R]), "Equator1"
    assert np.allclose(accessvis.latlon_to_3D(lon=180, lat=0), [0, 0, -R]), "Equator2"
    assert np.allclose(accessvis.latlon_to_3D(lon=90, lat=0), [R, 0, 0]), "Equator3"
    assert np.allclose(accessvis.latlon_to_3D(lon=270, lat=0), [-R, 0, 0]), "Equator4"


def test_lonlat_to_3D():
    import accessvis
    import numpy as np

    R = 6.371
    assert np.allclose(accessvis.lonlat_to_3D(lon=0, lat=90), [0, R, 0]), "North"
    assert np.allclose(accessvis.lonlat_to_3D(lon=0, lat=-90), [0, -R, 0]), "South"
    assert np.allclose(accessvis.lonlat_to_3D(lon=0, lat=0), [0, 0, R]), "Equator1"
    assert np.allclose(accessvis.lonlat_to_3D(lon=180, lat=0), [0, 0, -R]), "Equator2"
    assert np.allclose(accessvis.lonlat_to_3D(lon=90, lat=0), [R, 0, 0]), "Equator3"
    assert np.allclose(accessvis.lonlat_to_3D(lon=270, lat=0), [-R, 0, 0]), "Equator4"


def test_earth_vertices_to_3D():
    import accessvis
    import numpy as np

    R = 6.371
    vertices = np.array(
        [[0, 90, 0], [0, -90, 0], [0, 0, 0], [180, 0, 0], [90, 0, 0], [270, 0, 0]]
    )

    out = accessvis.earth_vertices_to_3D(vertices=vertices)
    expected = np.array(
        [[0, R, 0], [0, -R, 0], [0, 0, R], [0, 0, -R], [R, 0, 0], [-R, 0, 0]]
    )
    assert np.allclose(out, expected)


def test_lonlat_grid_3D():
    import accessvis
    import numpy as np

    out1 = accessvis.lonlat_grid_3D(
        latitudes=(-30, 30), longitudes=(-75, 130), resolution=(20, 10)
    )
    assert out1.shape == (20, 10, 3)
    assert np.all(np.diff(out1[:, :, 1], axis=1) > 0), "Z direction does not increase"

    out2 = accessvis.lonlat_grid_3D(
        latitudes=np.linspace(-30, 30, 10), longitudes=np.linspace(-75, 130, 20)
    )
    assert np.allclose(out1, out2)

    out3 = accessvis.lonlat_grid_3D(
        latitudes=np.array([-90, 0, 90]),
        longitudes=np.array([0, 90, 180, 270]),
        altitude=0,
    )
    R = 6.371
    exp3 = np.array(
        [
            [[0, -R, 0], [0, 0, R], [0, R, 0]],
            [[0, -R, 0], [R, 0, 0], [0, R, 0]],
            [[0, -R, 0], [0, 0, -R], [0, R, 0]],
            [[0, -R, 0], [-R, 0, 0], [0, R, 0]],
        ]
    )
    assert np.allclose(out3, exp3)


# def test_read_image():# TODO
#     import accessvis
#     accessvis.read_image()

# def test_paste_image():# TODO
#     import accessvis
#     accessvis.paste_image()
