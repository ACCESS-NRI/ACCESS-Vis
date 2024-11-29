# https://github.com/sunset1995/py360convert
# #!pip install py360convert
import datetime
import glob
import gzip
import math
import os
from contextlib import closing
from io import BytesIO
from pathlib import Path

import lavavu
import matplotlib
import numpy as np
import py360convert
import quaternion as quat
import xarray as xr
from PIL import Image

from .utils import download, is_notebook, pushd

Image.MAX_IMAGE_PIXELS = None

MtoLL = 1.0 / 111133  # Rough conversion from metres to lat/lon units


class Settings:
    # Texture and geometry resolutions
    RES = 0
    # Equirectangular res in y (x=2*y)
    FULL_RES_Y = 10800
    # Cubemap res
    TEXRES = 2048
    GRIDRES = 1024
    MAXGRIDRES = 4096

    # INSTALL_PATH is used for files such as sea-water-normals.png
    INSTALL_PATH = Path(__file__).parents[0]

    # Default to non-headless mode
    HEADLESS = False
    hostname = os.getenv("HOSTNAME", "")
    gadi = "gadi.nci.org.au" in hostname
    if gadi:
        # Enable headless via moderngl when running on gadi
        HEADLESS = True

    # Where data is stored, should use public cache dir on gadi
    # Check if the data directory is specified in environment variables
    envdir = os.getenv("ACCESSVIS_DATA_DIR")
    if envdir:
        DATA_PATH = Path(envdir)
    else:
        # Check if running on "gadi.nci.org.au"
        if gadi:
            # Use public shared data cache on gadi
            DATA_PATH = Path("/g/data/xp65/public/apps/access-vis-data")
            if not os.access(DATA_PATH, os.R_OK):
                # Use /scratch
                project = os.getenv("PROJECT")
                user = os.getenv("USER")
                DATA_PATH = Path(f"/scratch/{project}/{user}/.accessvis")
        else:
            DATA_PATH = Path.home() / ".accessvis"

    os.makedirs(DATA_PATH, exist_ok=True)

    GEBCO_PATH = DATA_PATH / "gebco" / "GEBCO_2020.nc"

    def __repr__(self):
        return f"resolution {self.RES}, {self.FULL_RES_Y}, texture {self.TEXRES}, grid {self.GRIDRES}, maxgrid {self.MAXGRIDRES} basedir {self.DATA_PATH}"


settings = Settings()


def get_viewer(*args, **kwargs):
    """

    Parameters
    ----------
    arguments for lavavu.Viewer().
    See https://lavavu.github.io/Documentation/lavavu.html#lavavu.Viewer for more documentation.

    Returns
    -------
    return: lavavu.Viewer
    """
    if settings.HEADLESS:
        from importlib import metadata, util

        try:
            # If this fails, lavavu-osmesa was installed
            # headless setting not required as implicitly in headless mode
            metadata.metadata("lavavu")
            # Also requires moderngl for EGL headless context
            settings.HEADLESS = util.find_spec("moderngl") is not None

        except (metadata.PackageNotFoundError):
            settings.HEADLESS = False

    if settings.HEADLESS:
        return lavavu.Viewer(*args, context="moderngl", **kwargs)
    else:
        return lavavu.Viewer(*args, **kwargs)


def set_resolution(val):
    """
    Sets the resolution of the following:
         settings.RES, settings.TEXRES, settings.FULL_RES_Y, settings.GRIDRES

    Parameters
    ----------
    val: Texture and geometry resolution
    """
    settings.RES = val
    settings.TEXRES = pow(2, 10 + val)
    settings.FULL_RES_Y = pow(2, max(val - 2, 0)) * 10800
    settings.GRIDRES = min(pow(2, 9 + val), settings.MAXGRIDRES)


def resolution_selection(default=1):
    """

    Parameters
    ----------
    default: resolution 1=low ... 4=high

    Returns
    -------
    widget: ipython.widgets.Dropdown
        Allows a user to select their desired resolution.
    """
    # Output texture resolution setting
    desc = """Low-res 2K - fast for testing
Mid-res 4K - good enough for full earth views
High res 8K - better if showing close up at country scale
Ultra-high 16K - max detail but requires a fast GPU with high memory"""
    if settings.RES == 0:
        set_resolution(default)
    if not is_notebook():
        return None
    print(desc)
    # from IPython.display import display
    import ipywidgets as widgets

    w = widgets.Dropdown(
        options=[
            ("Low-res 2K", 1),
            ("Mid-res 4K", 2),
            ("High-res 8K", 3),
            ("Ultra-high-res 16K", 4),
        ],
        value=settings.RES,
        description="Detail:",
    )

    def on_change(change):
        if change and change["type"] == "change" and change["name"] == "value":
            set_resolution(w.value)

    w.observe(on_change)
    set_resolution(default)
    return w


def read_image(fn):
    """
    Reads an image and returns as a numpy array,
    also supporting gzipped images (.gz extension)

    Parameters
    ----------
    fn: str|Path
        The file path to an image

    Returns
    -------
    image: numpy.ndarray
    """
    # supports .gz extraction on the fly
    p = Path(fn)
    # print(p.suffixes, p.suffixes[-2].lower())
    if p.suffix == ".gz" and p.suffixes[-2].lower() in [
        ".tif",
        ".tiff",
        ".png",
        ".jpg",
        ".jpeg",
    ]:
        with gzip.open(fn, "rb") as f:
            file_content = f.read()
            buffer = BytesIO(file_content)
            image = Image.open(buffer)
            return np.array(image)
    else:
        image = Image.open(fn)
        return np.array(image)


def paste_image(fn, xpos, ypos, out):
    """
    #Read an image from filename then paste a tile into a larger output image
    #Assumes output is a multiple of source tile image size and matching data type

    Parameters
    ----------
    fn: str|Path
        file name
    xpos: int
    ypos: int
    out: np.ndarray
        image to update
    """
    col = read_image(fn)

    # print(fn, col.shape)
    xoff = xpos * col.shape[0]
    yoff = ypos * col.shape[1]
    # print(f"{yoff}:{yoff+col.shape[1]}, {xoff}:{xoff+col.shape[0]}")
    out[yoff : yoff + col.shape[1], xoff : xoff + col.shape[0]] = col


def latlon_to_3D(lat, lon, alt=0):
    """
    Convert lat/lon coord to 3D coordinate for visualisation
    Uses simple spherical earth rather than true ellipse
    see http://www.mathworks.de/help/toolbox/aeroblks/llatoecefposition.html
    https://stackoverflow.com/a/20360045
    """
    return latlon_to_3D_true(lat, lon, alt, flattening=0.0)


def latlon_to_3D_true(lat, lon, alt=0, flattening=1.0 / 298.257223563):
    """
    Convert lat/lon coord to 3D coordinate for visualisation
    Uses flattening factor for elliptical earth
    see http://www.mathworks.de/help/toolbox/aeroblks/llatoecefposition.html
    https://stackoverflow.com/a/20360045
    """
    rad = np.float64(6.371)  # Radius of the Earth (in 1000's of kilometers)

    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    cos_lat = np.cos(lat_r)
    sin_lat = np.sin(lat_r)

    # Flattening factor WGS84 Model
    FF = (1.0 - np.float64(flattening)) ** 2
    C = 1 / np.sqrt(cos_lat**2 + FF * sin_lat**2)
    S = C * FF

    x = (rad * C + alt) * cos_lat * np.cos(lon_r)
    y = (rad * C + alt) * cos_lat * np.sin(lon_r)
    z = (rad * S + alt) * sin_lat

    # Coord order swapped to match our coord system
    return np.array([y, z, x])
    # return (x, y, z)


def split_tex(data, res, flip=[]):
    """
    Convert a texture image from equirectangular to a set of 6 cubemap faces
    (requires py360convert)
    """
    if len(data.shape) == 2:
        data = data.reshape(data.shape[0], data.shape[1], 1)
    channels = data.shape[2]
    # Convert equirectangular to cubemap
    out = py360convert.e2c(data, face_w=res, mode="bilinear", cube_format="dict")
    tiles = {}
    for o in out:
        # print(o, out[o].shape)
        tiles[o] = out[o].reshape(res, res, channels)
        if True:  # o in flip:
            tiles[o] = np.flipud(tiles[o])
            # tiles[o] = np.fliplr(tiles[o])
    return tiles


def draw_latlon_grid(base_img, out_fn, lat=30, lon=30, linewidth=5, colour=0):
    """
    Create lat/lon grid image over provided base image
    """
    from PIL import Image

    # Open base image
    image = Image.open(base_img)

    # Set the gridding interval
    x_div = 360.0 / lat  # degrees grid in X [0,360]
    y_div = 180.0 / lon  # degree grid in Y [-90,90]
    interval_x = round(image.size[0] / x_div)
    interval_y = round(image.size[1] / y_div)

    # Vertical lines
    lw = round(linewidth / 2)
    for i in range(0, image.size[0], interval_x):
        for j in range(image.size[1]):
            for k in range(-lw, lw):
                if i + k < image.size[0]:
                    image.putpixel((i + k, j), colour)
    # Horizontal lines
    for i in range(image.size[0]):
        for j in range(0, image.size[1], interval_y):
            # image.putpixel((i, j), colour)
            for k in range(-lw, lw):
                if j + k < image.size[1]:
                    image.putpixel((i, j + k), colour)

    # display(image)
    image.save(out_fn)


def latlon_to_uv(lat, lon):
    """
    Convert a decimal longitude, latitude coordinate
    to a tex coord in an equirectangular image
    """
    # X/u E-W Longitude - [-180,180]
    u = 0.5 + lon / 360.0

    # Y/v N-S Latitude  - [-90,90]
    v = 0.5 - lat / 180.0

    return u, v


def uv_to_pixel(u, v, width, height):
    """
    Convert tex coord [0,1]
    to a raster image pixel coordinate for given width/height
    """
    return int(u * width), int(v * height)


def latlon_to_pixel(lat, lon, width, height):
    """
    Convert a decimal latitude/longitude coordinate
    to a raster image pixel coordinate for given width/height
    """
    u, v = latlon_to_uv(lat, lon)
    return uv_to_pixel(u, v, width, height)


def crop_img_uv(img, cropbox):
    """
    Crop an image (PIL or numpy array) based on corner coords
    Provide coords as texture coords in [0,1]
    """
    top_left, bottom_right = cropbox
    u0 = top_left[0]
    u1 = bottom_right[0]
    v0 = top_left[1]
    v1 = bottom_right[1]
    # Swap coords if order incorrect
    if u0 > u1:
        u0, u1 = u1, u0
    if v0 > v1:
        v0, v1 = v1, v0
    # Supports numpy array or PIL image
    if isinstance(img, np.ndarray):
        # Assumes [lat][lon]
        lat = int(v0 * img.shape[0]), int(v1 * img.shape[0])
        lon = int(u0 * img.shape[1]), int(u1 * img.shape[1])
        print(lat, lon)
        return img[lat[0] : lat[1], lon[0] : lon[1]]
    elif hasattr(img, "crop"):
        area = (
            int(u0 * img.size[0]),
            int(v0 * img.size[1]),
            int(u1 * img.size[0]),
            int(v1 * img.size[1]),
        )
        return img.crop(area)
    else:
        print("Unknown type: ", type(img))


def crop_img_lat_lon(img, cropbox):
    """
    Crop an equirectangular image (PIL or numpy array) based on corner coords
    Provide coords as lat/lon coords in decimal degrees
    """
    a = latlon_to_uv(*cropbox[0])
    b = latlon_to_uv(*cropbox[1])
    return crop_img_uv(img, (a, b))


def sphere_mesh(radius=1.0, quality=256, cache=True):
    """
    Generate a simple spherical mesh, not suitable for plotting accurate texture/data at the
    poles as there will be visible pinching artifacts,
    see: cubemap_sphere_vertices() for mesh without artifacts

    Parameters
    ----------
    radius: float
        Radius of the sphere
    quality: int
        Sphere mesh quality (higher = more triangles)
    cache: bool
        If true will attempt to load cached data and if not found will generate
        and save the data for next time

    """
    # Generate cube face grid
    fn = f"Sphere_{quality}_{radius:.4f}"
    if cache and os.path.exists(fn + ".npz"):
        sdata = np.load(fn + ".npz")
    else:
        lv = get_viewer()
        tris0 = lv.spheres(
            "sphere",
            scaling=radius,
            segments=quality,
            colour="grey",
            vertices=[0, 0, 0],
            fliptexture=False,
        )
        tris0["rotate"] = [
            0,
            -90,
            0,
        ]  # This rotates the sphere coords to align with [0,360] longitude texture
        tris0[
            "texture"
        ] = "data/blank.png"  # Need an initial texture or texcoords will not be generated
        tris0["renderer"] = "sortedtriangles"
        lv.render()

        # Generate and extract sphere vertices, texcoords etc
        lv.bake()  # 1)
        sdata = {}
        element = tris0.data[0]
        keys = element.available.keys()
        for k in keys:
            sdata[k] = tris0.data[k + "_copy"][0]

    # Save compressed vertex data
    if cache:
        np.savez_compressed(fn, **sdata)
    return sdata


def cubemap_sphere_vertices(
    radius=1.0,
    resolution=None,
    heightmaps=None,
    vertical_exaggeration=1.0,
    cache=True,
    hemisphere=None,
):
    """
    Generate a spherical mesh from 6 cube faces, suitable for cubemap textures and
    without stretching/artifacts at the poles

    Parameters
    ----------
    radius: float
        Radius of the sphere
    resolution: int
        Each face of the cube will have this many vertices on each side
        Higher for more detailed surface features
    heightmaps: dictionary of numpy.ndarrays [face](resolution,resolution)
        If provided will add the heights for each face to the radius to provide
        surface features, eg: topography/bathymetry for an earth sphere
    vertical_exaggeration: float
        Multiplier to exaggerate the heightmap in the vertical axis,
        eg: to highlight details of topography and bathymetry
    cache: bool
        If true will attempt to load cached data and if not found will generate
        and save the data for next time
    hemisphere: str
        Crop the data to show a single hemisphere
        "N" = North polar
        "S" = South polar
        "EW" = Antimeridian at centre (Oceania/Pacific)
        "WE" = Prime meridian at centre (Africa/Europe)
        "E" = Eastern hemisphere - prime meridian to antimeridian (Indian ocean)
        "W" = Western hemisphere - antimeridian to prime meridian (Americas)
    """
    if resolution is None:
        resolution = settings.GRIDRES
    # Generate cube face grid
    sdata = {}
    cdata = {}
    fn = f"{settings.DATA_PATH}/sphere/cube_sphere_{resolution}"
    if cache and os.path.exists(fn + ".npz"):
        cdata = np.load(fn + ".npz")
        cache = False  # Don't need to write again
    os.makedirs(settings.DATA_PATH / "sphere", exist_ok=True)

    # For each cube face...
    minmax = []
    for f in ["F", "R", "B", "L", "U", "D"]:
        if f in cdata:
            verts = cdata[f]
        else:
            ij = np.linspace(-1.0, 1.0, resolution, dtype="float32")
            ii, jj = np.meshgrid(ij, ij)  # 2d grid
            zz = np.zeros(shape=ii.shape, dtype="float32")  # 3rd dim
            if f == "F":  ##
                vertices = np.dstack((ii, jj, zz + 1.0))
            elif f == "B":
                vertices = np.dstack((ii, jj, zz - 1.0))
            elif f == "R":
                vertices = np.dstack((zz + 1.0, jj, ii))
            elif f == "L":  ##
                vertices = np.dstack((zz - 1.0, jj, ii))
            elif f == "U":
                vertices = np.dstack((ii, zz + 1.0, jj))
            elif f == "D":  ##
                vertices = np.dstack((ii, zz - 1.0, jj))
            # Normalise the vectors to form spherical patch  (normalised cube)
            V = vertices.ravel().reshape((-1, 3))
            norms = np.sqrt(np.einsum("...i,...i", V, V))
            norms = norms.reshape(resolution, resolution, 1)
            verts = vertices / norms
            cdata[f] = verts.copy()

        # Scale and apply surface detail?
        if heightmaps:
            # Apply radius and heightmap
            verts *= heightmaps[f] * vertical_exaggeration + radius
            minmax += [heightmaps[f].min(), heightmaps[f].max()]
        else:
            # Apply radius only
            verts *= radius
        sdata[f] = verts

    # Save height range
    minmax = np.array(minmax)
    sdata["range"] = (minmax.min(), minmax.max())

    # Hemisphere crop?
    half = resolution // 2
    if hemisphere == "N":  # U
        del sdata["D"]  # Delete south
        for f in ["F", "R", "B", "L"]:
            sdata[f] = sdata[f][half::, ::, ::]  # Crop bottom section
    elif hemisphere == "S":  # D
        del sdata["U"]  # Delete north
        for f in ["F", "R", "B", "L"]:
            sdata[f] = sdata[f][0:half, ::, ::]  # Crop top section
    elif hemisphere == "E":  # R
        del sdata["L"]  # Delete W
        for f in ["F", "B", "U", "D"]:
            sdata[f] = sdata[f][::, half::, ::]  # Crop left section
    elif hemisphere == "W":  # L
        del sdata["R"]  # Delete E
        for f in ["F", "B", "U", "D"]:
            sdata[f] = sdata[f][::, 0:half, ::]  # Crop right section
    elif hemisphere == "EW":  # B
        del sdata["F"]  # Delete prime meridian
        for f in ["R", "L"]:
            sdata[f] = sdata[f][::, 0:half, ::]  # Crop right section
        for f in ["U", "D"]:
            sdata[f] = sdata[f][0:half, ::, ::]  # Crop top section
    elif hemisphere == "WE":  # F
        del sdata["B"]  # Delete antimeridian
        for f in ["R", "L"]:
            sdata[f] = sdata[f][::, half::, ::]  # Crop left section
        for f in ["U", "D"]:
            sdata[f] = sdata[f][half::, ::, ::]  # Crop bottom section

    # Save compressed un-scaled vertex data
    if cache:
        np.savez_compressed(fn, **cdata)
    return sdata


def load_topography_cubemap(
    resolution=None,
    radius=6.371,
    vertical_exaggeration=1,
    bathymetry=True,
    hemisphere=None,
):
    """
    Load topography from pre-saved data
    TODO: Support land/sea mask, document args

    Parameters
    ----------
    resolution: int
        Each face of the cube will have this many vertices on each side
        Higher for more detailed surface features
    vertical_exaggeration: number
        Multiplier to topography/bathymetry height
    radius: float
        Radius of the sphere, defaults to 6.371 Earth's approx radius in Mm
    hemisphere: str
        Crop the data to show a single hemisphere
        "N" = North polar
        "S" = South polar
        "EW" = Antimeridian at centre (Oceania/Pacific)
        "WE" = Prime meridian at centre (Africa/Europe)
        "E" = Eastern hemisphere - prime meridian to antimeridian (Indian ocean)
        "W" = Western hemisphere - antimeridian to prime meridian (Americas)
    """
    # Load detailed topo data
    if resolution is None:
        resolution = settings.GRIDRES
    process_gebco()  # Ensure data exists
    fn = f"{settings.DATA_PATH}/gebco/gebco_cubemap_{resolution}.npz"
    if not os.path.exists(fn):
        raise (Exception("GEBCO data not found"))
    heights = np.load(fn)
    # Apply to cubemap sphere
    return cubemap_sphere_vertices(
        radius, resolution, heights, vertical_exaggeration, hemisphere=hemisphere
    )


def load_topography(resolution=None, subsample=1, cropbox=None, bathymetry=True):
    """
    Load topography from pre-saved equirectangular data, can be cropped for regional plots

    TODO: document args
    """
    if resolution is None:
        resolution = settings.FULL_RES_Y
    heights = None
    # Load medium-detail topo data
    if resolution > 21600:
        # Attempt to load full GEBCO
        if not settings.GEBCO_PATH or not os.path.exists(settings.GEBCO_PATH):
            resolution = 21600
            print("Please pass path to GEBCO_2020.nc in settings.GEBCO_PATH")
            print("https://www.bodc.ac.uk/data/open_download/gebco/gebco_2020/zip/")
            print(f"Dropping resolution to {resolution} in order to continue...")
        else:
            ds = xr.open_dataset(settings.GEBCO_PATH)
            heights = ds["elevation"][::subsample, ::subsample].to_numpy()
            heights = np.flipud(heights)

    if heights is None:
        process_gebco()  # Ensure data exists
        basefn = f"gebco_equirectangular_{resolution * 2}_x_{resolution}.npz"
        fn = f"{settings.DATA_PATH}/gebco/{basefn}"
        if not os.path.exists(fn):
            raise (Exception("GEBCO data not found"))
        else:
            heights = np.load(fn)
            heights = heights["elevation"]

    if subsample > 1:
        heights = heights[::subsample, ::subsample]
    if cropbox:
        heights = crop_img_lat_lon(heights, cropbox)

    # Bathymetry?
    if not bathymetry or not isinstance(bathymetry, bool):
        # Ensure resolution matches topo grid res
        # res_y = resolution//4096 * 10800
        # res_y = max(resolution,2048) // 2048 * 10800
        mask = load_mask(
            res_y=resolution, subsample=subsample, cropbox=cropbox, masktype="oceanmask"
        )
        # print(type(mask), mask.dtype, mask.min(), mask.max())
        if bathymetry == "mask":
            # Return a masked array
            return np.ma.array(heights, mask=(mask < 255), fill_value=0)
        elif not isinstance(bathymetry, bool):
            # Can pass a fill value, needs to return as floats instead of int though
            ma = np.ma.array(heights.astype(float), mask=(mask < 255))
            return ma.filled(bathymetry)
        else:
            # Zero out to sea level
            # Use the mask to zero the bathymetry
            heights[mask < 255] = 0

    return heights  # * vertical_exaggeration


def plot_region(
    lv=None,
    cropbox=None,
    vertical_exaggeration=10,
    texture="bluemarble",
    lighting=True,
    when=None,
    waves=False,
    blendtex=True,
    bathymetry=False,
    name="surface",
    uniforms={},
    shaders=None,
    background="black",
    *args,
    **kwargs,
):
    """
    Plots a flat region of the earth with topography cropped to specified region (lat/lon bounding box coords)
    uses bluemarble textures by default and sets up seasonal texture blending based on given or current date and time

    Uses lat/lon as coordinate system, so no use for polar regions, scales heights to equivalent
    TODO: support using km as unit or other custom units instead with conversions from lat/lon

    TODO: FINISH DOCUMENTING PARAMS

    Parameters
    ----------
    texture: str
        Path to textures, face label and texres will be applied with .format(), eg:
        texture='path/{face}_mytexture_{texres}.png'
        with: texture.format(face='F', texres=settings.TEXRES)
        to:'path/F_mytexture_1024.png'
    name: str
        Append this label to each face object created
    vertical_exaggeration: number
        Multiplier to topography/bathymetry height
    """
    if lv is None:
        lv = get_viewer(
            border=False, axis=False, resolution=[1280, 720], background=background
        )

    # Custom uniforms / additional textures
    uniforms = {}

    """
    #TODO: wave shader etc for regional sections
    if waves:
        uniforms["wavetex"] = f"{settings.INSTALL_PATH}/data/sea-water-1024x1024_gs.png"
        uniforms["wavenormal"] = f"{settings.INSTALL_PATH}/data/sea-water_normals.png"
        uniforms["waves"] = True

    if shaders is None:
        shaders = [f"{settings.INSTALL_PATH}/data/earth_shader.vert", f"{settings.INSTALL_PATH}/data/earth_shader.frag"]
    """

    # Split kwargs into global props, object props and uniform values
    objargs = {}
    for k in kwargs:
        if k in lv.properties:
            if "object" in lv.properties[k]["target"]:
                objargs[k] = kwargs[k]
            else:
                lv[k] = kwargs[k]
        else:
            uniforms[k] = kwargs[k]

    # Load topo and crop via lat/lon boundaries of data
    topo = load_topography(cropbox=cropbox, bathymetry=bathymetry)
    height = np.array(topo)

    # Scale and apply vertical exaggeration
    height = height * MtoLL * vertical_exaggeration

    D = [height.shape[1], height.shape[0]]
    sverts = np.zeros(shape=(height.shape[0], height.shape[1], 3))
    lat0, lon0 = cropbox[0]
    lat1, lon1 = cropbox[1]
    xy = lv.grid2d(corners=((lon0, lat1), (lon1, lat0)), dims=D)
    sverts[::, ::, 0:2] = xy
    sverts[::, ::, 2] = height[::, ::]

    if texture == "bluemarble":
        # TODO: support cropping tiled high res blue marble textures
        # Also download relief textures if not found or call process_bluemarble
        # TODO2: write a process_relief function for splitting/downloading relief from Earth_Model.ipynb
        # colour_tex = f"{settings.DATA_PATH}/relief/4_no_ice_clouds_mts_16k.jpg"
        colour_tex = f"{settings.DATA_PATH}/bluemarble/source_full/world.200412.3x21600x10800.jpg"
        # colour_tex = f"{settings.DATA_PATH}/landmask/world.oceanmask.21600x10800.png"
        uniforms["bluemarble"] = True
    elif texture == "relief":
        colour_tex = f"{settings.DATA_PATH}/relief/4_no_ice_clouds_mts_16k.jpg"
    else:
        colour_tex = texture

    surf = lv.triangles(
        name, vertices=sverts, uniforms=uniforms, cullface=True, opaque=True
    )  # , fliptexture=False)

    img = Image.open(colour_tex)
    cropped_img = crop_img_lat_lon(img, cropbox)
    arr = np.array(cropped_img)
    surf.texture(arr, flip=False)
    return lv


def plot_earth(
    lv=None,
    radius=6.371,
    vertical_exaggeration=10,
    texture="bluemarble",
    lighting=True,
    when=None,
    hour=None,
    minute=None,
    waves=None,
    sunlight=False,
    blendtex=True,
    name="",
    uniforms={},
    shaders=None,
    background="black",
    hemisphere=None,
    *args,
    **kwargs,
):
    """
    Plots a spherical earth using a 6 face cubemap mesh with bluemarble textures
    and sets up seasonal texture blending and optionally, sun position,
    based on given or current date and time

    TODO: FINISH DOCUMENTING PARAMS

    Parameters
    ----------
    texture: str
        Path to textures, face label and texres will be applied with .format(), eg:
        texture="path/{face}_mytexture_{texres}.png"
        with: texture.format(face="F", texres=settings.TEXRES)
        to:"path/F_mytexture_1024.png"
    radius: float
        Radius of the sphere, defaults to 6.371 Earth's approx radius in Mm
    vertical_exaggeration: number
        Multiplier to topography/bathymetry height
    texture: str
        Texture set to use, "bluemarble" for the 2004 NASA satellite data, "relief" for a basic relief map
        or provide a custom set of textures using a filename template with the following variables, only face is required
        {face} (F/R/B/L/U/D) {month} (name of month, capitialised) {texres} (2048/4096/8192/16384)
    lighting: bool
        Enable lighting, default=True, disable for flat rendering without light and shadow, or to set own lighting params later
    when: datetime
        Provide a datetime object to set the month for texture sets that vary over the year and time for
        position of sun and rotation of earth when calculating sun light position
    hour: int
        If not providing "when" datetime, provide just the hour and minute
    minute: int
        If not providing "when" datetime, provide just the hour and minute
    waves: bool
        When plotting ocean as surface, set this to true to render waves
    sunlight: bool
        Enable sun light based on passed time of day args above, defaults to disabled and sun will follow the viewer,
        always appearing behind the camera position to provide consistant illumination over accurate positioning
    blendtex: bool
        When the texture set has varying seasonal images, enabling this will blend between the current month and next
        months texture to smoothly transition between them as the date changes, defaults to True
    name: str
        Append this label to each face object created
    uniforms: dict
        Provide a set of uniform variables, these can be used to pass data to a custom shader
    shaders: list
        Provide a list of two custom shader file paths eg: ["vertex_shader.glsl", "fragment_shader.glsl"]
    background: str
        Provide a background colour string, X11 colour name or hex RGB
    hemisphere: str
        Crop the data to show a single hemisphere
        "N" = North polar
        "S" = South polar
        "EW" = Antimeridian at centre (Oceania/Pacific)
        "WE" = Prime meridian at centre (Africa/Europe)
        "E" = Eastern hemisphere - prime meridian to antimeridian (Indian ocean)
        "W" = Western hemisphere - antimeridian to prime meridian (Americas)
    """
    if lv is None:
        lv = get_viewer(
            border=False, axis=False, resolution=[1280, 720], background=background
        )

    topo = load_topography_cubemap(
        settings.GRIDRES, radius, vertical_exaggeration, hemisphere=hemisphere
    )
    if when is None:
        when = datetime.datetime.now()
    month = when.strftime("%B")

    # Custom uniforms / additional textures
    uniforms["radius"] = radius

    if texture == "bluemarble":
        texture = "{basedir}/bluemarble/cubemap_{texres}/{face}_blue_marble_{month}_{texres}.png"
        uniforms["bluemarble"] = True
        if waves is None:
            waves = True
    elif texture == "relief":
        process_relief()  # Ensure images available
        texture = "{basedir}/relief/cubemap_{texres}/{face}_relief_{texres}.png"

    # Waves - load textures as shared
    lv.texture("wavetex", f"{settings.INSTALL_PATH}/data/sea-water-1024x1024_gs.png")
    lv.texture("wavenormal", f"{settings.INSTALL_PATH}/data/sea-water_normals.png")
    # Need to set the property too or will not know to load the texture
    if waves is None:
        waves = False
    uniforms["wavetex"] = ""
    uniforms["wavenormal"] = ""
    uniforms["waves"] = waves

    # Pass in height range of topography as this is dependent on vertical exaggeration
    # Convert metres to Mm and multiply by vertical exag
    # hrange = np.array([-10952, 8627]) * 1e-6 * vertical_exaggeration
    hrange = np.array(topo["range"]) * vertical_exaggeration
    uniforms["heightmin"] = hrange[0]
    uniforms["heightmax"] = hrange[1]

    if shaders is None:
        shaders = [
            f"{settings.INSTALL_PATH}/data/earth_shader.vert",
            f"{settings.INSTALL_PATH}/data/earth_shader.frag",
        ]

    # Split kwargs into global props, object props and uniform values
    objargs = {}
    for k in kwargs:
        if k in lv.properties:
            if "object" in lv.properties[k]["target"]:
                objargs[k] = kwargs[k]
            else:
                lv[k] = kwargs[k]
        else:
            uniforms[k] = kwargs[k]

    for f in ["F", "R", "B", "L", "U", "D"]:
        if f not in topo:
            continue  # For hemisphere crops
        verts = topo[f]

        texfn = texture.format(
            basedir=settings.DATA_PATH, face=f, texres=settings.TEXRES, month=month
        )

        lv.triangles(
            name=f + name,
            vertices=verts,
            texture=texfn,
            fliptexture=False,
            flip=f in ["F", "L", "D"],  # Reverse facing
            renderer="simpletriangles",
            opaque=True,
            cullface=True,
            shaders=shaders,
            uniforms=uniforms,
            **objargs,
        )

    # Setup seasonal texture for blue marble
    if "bluemarble" in texture:
        update_earth_datetime(lv, when, name, texture, sunlight, blendtex)

    # Default light props
    if lighting:
        lp = sun_light(time=when if sunlight else None, hour=hour, minute=minute)
        # lv.set_properties(diffuse=0.5, ambient=0.5, specular=0.05, shininess=0.06, light=[1,1,0.98,1])
        # lv.set_properties(diffuse=0.6, ambient=0.1, specular=0.0, shininess=0.01, light=[1,1,0.98,1])
        lv.set_properties(
            diffuse=0.6,
            ambient=0.6,
            specular=0.3,
            shininess=0.04,
            light=[1, 1, 0.98, 1],
            lightpos=lp,
        )

    # Hemisphere crop? Alter texcoords to fix half cubemap sections
    def replace_texcoords(f, idx, lr):
        obj = lv.objects[f + name]
        el = np.copy(obj.data["texcoords"][0])
        obj.cleardata("texcoords")
        col = el[::, ::, idx]
        if lr == "r":
            el[::, ::, idx] = col * 0.5 + 0.5
        elif lr == "l":
            el[::, ::, idx] = col * 0.5
        obj.texcoords(el)

    if hemisphere is not None:
        lv.render()  # Display/update

    if hemisphere == "N":
        for f in ["F", "R", "B", "L"]:
            replace_texcoords(f, 1, "r")
        lv.rotation(90.0, 0.0, 0.0)
    elif hemisphere == "S":
        for f in ["F", "R", "B", "L"]:
            replace_texcoords(f, 1, "l")
        lv.rotation(-90.0, 0.0, 0.0)
    elif hemisphere == "E":
        for f in ["F", "B", "U", "D"]:
            replace_texcoords(f, 0, "r")
        lv.rotation(0.0, -90.0, 0.0)
    elif hemisphere == "W":
        for f in ["F", "B", "U", "D"]:
            replace_texcoords(f, 0, "l")
        lv.rotation(0.0, 90.0, 0.0)
    elif hemisphere == "EW":
        for f in ["R", "L"]:
            replace_texcoords(f, 0, "l")
        for f in ["U", "D"]:
            replace_texcoords(f, 1, "l")
        lv.rotation(0.0, 180.0, 0.0)
    elif hemisphere == "WE":
        for f in ["R", "L"]:
            replace_texcoords(f, 0, "r")
        for f in ["U", "D"]:
            replace_texcoords(f, 1, "r")
        lv.rotation(0.0, 0.0, 0.0)
    lv.reload()
    return lv


def update_earth_datetime(
    lv, when, name="", texture=None, sunlight=False, blendtex=True
):
    # Update date/time for texture blending and lighting
    d = when.day - 1
    m = when.month
    month = when.strftime("%B")
    m2 = m + 1 if m < 12 else 1
    when2 = when.replace(day=1, month=m2)
    month2 = when2.strftime("%B")
    # days = (datetime.date(when.year, m, 1) - datetime.date(when.year, m, 1)).days
    # days = (date(2004, m2, 1) - date(2004, m, 1)).days
    days = (
        when.replace(month=when.month % 12 + 1, day=1) - datetime.timedelta(days=1)
    ).day
    factor = d / days

    if texture is None:
        texture = "{basedir}/bluemarble/cubemap_{texres}/{face}_blue_marble_{month}_{texres}.png"

    if "bluemarble" in texture:
        # Check texture exists, if not download and process
        process_bluemarble(when, blendtex=blendtex)

    for f in ["F", "R", "B", "L", "U", "D"]:
        texfn = texture.format(
            basedir=settings.DATA_PATH, face=f, texres=settings.TEXRES, month=month
        )
        texfn2 = texture.format(
            basedir=settings.DATA_PATH, face=f, texres=settings.TEXRES, month=month2
        )
        assert os.path.exists(texfn)
        assert os.path.exists(texfn2)
        o = f + name
        if o in lv.objects:
            obj = lv.objects[o]
            uniforms = obj["uniforms"]

            # if not "blendTex" in uniforms or uniforms["blendTex"] != texfn2:
            if obj["texture"] != texfn:
                obj["texture"] = texfn  # Not needed, but set so can be checked above
                obj.texture(texfn, flip=False)

            if blendtex and (
                "blendTex" not in uniforms or uniforms["blendTex"] != texfn2
            ):
                uniforms["blendTex"] = texfn2
                obj.texture(texfn2, flip=False, label="blendTex")

            if not blendtex:
                factor = -1.0  # Disable blending multiple textures
            uniforms["blendFactor"] = factor

            obj["uniforms"] = uniforms

    lv.render()  # Required to render a frame which fixes texture glitch
    if sunlight:
        lv.set_properties(lightpos=sun_light(time=when))


def update_earth_texture(
    lv, label, texture, flip=False, shared=True, name="", *args, **kwargs
):
    # Update texture values for a specific texture on earth model
    if shared:
        # No need to update each object
        lv.texture(label, texture, flip)
    for f in ["F", "R", "B", "L", "U", "D"]:
        o = f + name
        if o in lv.objects:
            obj = lv.objects[o]
            uniforms = obj["uniforms"]
            if not shared:
                obj.texture(texture, flip=flip, label=label)
            else:
                uniforms[label] = ""
            uniforms.update(kwargs)
            obj["uniforms"] = uniforms

    lv.render()  # Required to render a frame which fixes texture glitch


def update_earth_values(lv, name="", flip=False, *args, **kwargs):
    # Update uniform values on earth objects via passed kwargs

    # Replace texture data load shared texture afterwards
    for k in kwargs:
        if isinstance(kwargs[k], (np.ndarray, np.generic)) or k == "data":
            lv.texture(k, kwargs[k], flip=flip)
            kwargs[k] = ""  # Requires a string value to trigger texture load

    for f in ["F", "R", "B", "L", "U", "D"]:
        o = f + name
        if o in lv.objects:
            obj = lv.objects[o]
            uniforms = obj["uniforms"]
            uniforms.update(kwargs)
            obj["uniforms"] = uniforms

    lv.render()  # Required to render a frame which fixes texture glitch


def vec_rotate(v, theta, axis):
    """
    Rotate a 3D vector about an axis by given angle

    Parameters
    ----------
    v : list/numpy.ndarray
        The 3 component vector
    theta : float
        Angle in degrees
    axis : list/numpy.ndarray
        The 3 component axis of rotation

    Returns
    -------
    numpy.ndarray: rotated 3d vector
    """
    rot_axis = np.array([0.0] + axis)
    axis_angle = (theta * 0.5) * rot_axis / np.linalg.norm(rot_axis)

    vec = quat.quaternion(*v)
    qlog = quat.quaternion(*axis_angle)
    q = np.exp(qlog)

    v_prime = q * vec * np.conjugate(q)

    # print(v_prime) # quaternion(0.0, 2.7491163, 4.7718093, 1.9162971)
    return v_prime.imag


def magnitude(vec):
    return np.linalg.norm(vec)


def normalise(vec):
    norm = np.linalg.norm(vec)
    if norm == 0:
        vn = vec
    else:
        vn = vec / norm
    return vn


def vector_align(v1, v2, lvformat=True):
    """
    Get a rotation quaterion to align vectors v1 with v2

    Parameters
    ----------
    v1 : list/numpy.ndarray
         First 3 component vector
    v2 : list/numpy.ndarray
         Second 3 component vector to align the first to

    Returns
    -------
    list: quaternion to rotate v1 to v2 (in lavavu format)
    """

    # Check for parallel or opposite
    v1 = normalise(np.array(v1))
    v2 = normalise(np.array(v2))
    epsilon = np.finfo(np.float32).eps
    one_minus_eps = 1.0 - epsilon
    if np.dot(v1, v2) > one_minus_eps:  #  1.0
        # No rotation
        return [0, 0, 0, 1]
    elif np.dot(v1, v2) < -one_minus_eps:  # -1.0
        # 180 rotation about Y
        return [0, 1, 0, 1]
    xyz = np.cross(v1, v2)
    l1 = np.linalg.norm(v1)
    l2 = np.linalg.norm(v2)
    w = math.sqrt((l1 * l1) * (l2 * l2)) + np.dot(v1, v2)
    qr = quat.quaternion(w, xyz[0], xyz[1], xyz[2])
    qr = qr.normalized()
    # Return in LavaVu quaternion format
    if lvformat:
        return [qr.x, qr.y, qr.z, qr.w]
    else:
        return qr


def lookat(lv, pos, lookat=None, up=None):
    """
    Set the camera with a position coord and lookat coord

    Parameters
    ----------
    lv : lavavu.Viewer
        The viewer object
    pos : list/numpy.ndarray
        Camera position in world coords
    lookat : list/numpy.ndarray
        Look at position in world coords, defaults to model origin
    up : list/numpy.ndarray
        Up vector, defaults to Y axis [0,1,0]
    """

    # Use the origin from viewer if no target provided
    if lookat is None:
        lookat = lv["focus"]
    else:
        lv["focus"] = lookat

    # Default to Y-axis up vector
    if up is None:
        up = np.array([0, 1, 0])

    # Calculate the rotation matrix
    heading = np.array(pos) - np.array(lookat)
    zd = normalise(heading)
    xd = normalise(np.cross(up, zd))
    yd = normalise(np.cross(zd, xd))
    q = quat.from_rotation_matrix(np.array([xd, yd, zd]))
    q = q.normalized()

    # Apply the rotation
    lv.rotation(q.x, q.y, q.z, q.w)

    # Translate back by heading vector length in Z
    # (model origin in lavavu takes care of lookat offset)
    tr = [0, 0, -magnitude(np.array(pos) - np.array(lookat))]

    # Apply translation
    lv.translation(tr)


def sun_light(
    time=None,
    now=False,
    local=True,
    tz=None,
    hour=None,
    minute=None,
    xyz=[0.4, 0.4, 1.0],
):
    """
    Setup a sun light for earth model illumination
    Default with no parameters is a sun light roughly behind the camera that
    follows the camera position the model is always illuminated

    With time parameters passed, the sun will be placed in the correct position
    accounting for the time of year (earth position related to sun)
    and time of day (earth's rotation)

    NOTE: the first time using this may be slow and require an internet connection
    to allow astropy to download the IERS file https://docs.astropy.org/en/stable/utils/data.html

    Parameters
    ----------
    time : datetime.datetime
        Time as a datetime object
        defaults to utc, use local to set local timezone or tz to provide one
    now : bool
        Alternative to passing time, will use current local time in your timezone
    local : bool
        When true, will assume and set the local timezone,
        otherwise leaves the provided datetime as is
    tz : datetime.timezone
        Pass a timezone object to set for the provided time
    hour : float
        Pass an hour value, will override the hour of current or passed in time
    minute : float
        Pass a minute value, will override the minute of current or passed in time
    xyz : list/ndarray
        When not using a time of day, this reference vector is used for a light that
        follows (sits behind) the camera, controlling the x,y,z components
        of the final light position, will be normalised and then
        multiplied by the earth to sun distance to get the position

    Returns
    -------
    list: light position array to pass to lavavu, eg: lv["lightpos"] = sun_light(now=True)
    """

    # Distance Earth --> Sun. in our units Mm (Millions of metres)
    # 151.17 million km
    # 151170 million metres
    dist = 151174
    vdist = np.linalg.norm(xyz)
    # Calculate a sun position based on reference xyz vector
    LP = np.array(xyz) / vdist * dist

    if now or time is not None or hour is not None or minute is not None:
        # Calculate sun position and earth rotation given time
        # requires astropy
        try:
            import astropy.coordinates
            from astropy.time import Time

            # Get local timezone
            ltz = datetime.datetime.now(datetime.timezone.utc).astimezone().tzinfo
            if now or time is None:
                time = datetime.datetime.now(tz=ltz)

            # Replace timezone?
            if tz:
                time = time.replace(tzinfo=tz)
            elif local and time.tzinfo is None:
                time = time.replace(tzinfo=ltz)

            # Replace hour or minute?
            if hour is not None:
                time = time.replace(hour=hour)
                # If only hour provided, always zero the minute
                if minute is None:
                    minute = 0
            if minute is not None:
                time = time.replace(minute=minute)

            # Create astropy Time object
            at = Time(time, scale="utc")

            # Get sun position in ECI / GCRS coords (earth centre of mass = origin)
            sun_pos = astropy.coordinates.get_body("sun", at)
            # Add location so we can get rotation angle
            t = Time(time, location=("0d", "0d"))
            a = t.earth_rotation_angle()
            # print(a.deg, a.rad)

            S = (
                sun_pos.cartesian.x.to("Mm"),
                sun_pos.cartesian.y.to("Mm"),
                sun_pos.cartesian.z.to("Mm"),
            )

            # Swap x=y, y=z, z=x for our coord system
            S = (S[1].value, S[2].value, S[0].value)

            # Apply rotation about Y axis (negative)
            SR = vec_rotate(np.array(S), -a.rad, axis=[0, 1, 0])

            # Set 4th component to 1 to enable fixed light pos
            LP = [SR[0], SR[1], SR[2], 1]

        except (ImportError):
            print("Sun/time lighting requires astropy, pip install astropy")

    return LP


def normalise_array(values, minimum=None, maximum=None):
    """
    Normalize an array to the range [0,1]

    Parameters
    ----------
    values : numpy.ndarray
        Values to convert, numpy array
    minimum: number
        Use a fixed minimum bound, default is to use the data minimum
    maximum: number
        Use a fixed maximum bound, default is to use the data maximum
    """

    # Ignore nan when getting min/max
    if not minimum:
        minimum = np.nanmin(values)
    if not maximum:
        maximum = np.nanmax(values)

    # Normalise
    array = (values - minimum) / (maximum - minimum)
    # Clip out of [0,1] range - in case defined range is not the global minima/maxima
    array = np.clip(array, 0, 1)

    return array


def array_to_rgba(
    values,
    colourmap="coolwarm",
    minimum=None,
    maximum=None,
    flip=False,
    opacity=0.0,
    opacitymap=False,
):
    """
    Array to rgba texture using a matplotlib colourmap

    Parameters
    ----------
    values : numpy.ndarray
        Values to convert, numpy array
    colourmap: str or list or object
        If a string, provides the name of a matplotlib colourmap to use
        If a list is passed, should contain RGB[A] colours as lists or tuples in range [0,1] or [0,255],
        a matplotlib.colors.LinearSegmentedColormap will be created from the list
        Otherwise, will assume is a valid matplotlib colormap already
    minimum: number
        Use a fixed minimum for the colourmap range, default is to use the data minimum
    maximum: number
        Use a fixed maximum for the colourmap range, default is to use the data maximum
    flip: bool
        Flips the output vertically
    opacity: float
        Set a fixed opacity value in the output image
    opacitymap: bool or numpy.ndarray
        Set to true to use values as an opacity map, top of range will be opaque, bottom transparent
        Provide an array to use a different opacity map dataset
    """

    array = normalise_array(values, minimum, maximum)

    if flip:
        array = np.flipud(np.array(array))

    if isinstance(colourmap, str):
        mcm = matplotlib.pyplot.get_cmap(colourmap)
    elif isinstance(colourmap, list):
        colours = np.array(colourmap)
        if colours.max() > 1.0:
            # Assume range [0,255]
            colours = colours / 255.0
        matplotlib.colors.LinearSegmentedColormap.from_list(
            "custom", colours[::, 0:3]
        )  # , N=len(colours))
        mcm = matplotlib.pyplot.get_cmap(colourmap)
    else:
        mcm = colourmap
    # TODO: support LavaVu ColourMap objects

    # Apply colourmap
    colours = mcm(array)

    # Convert to uint8
    rgba = (colours * 255).round().astype(np.uint8)
    if opacity:
        if opacity <= 1.0:
            opacity = int(255 * opacity)
        rgba[::, ::, 3] = opacity
    elif opacitymap is True:  # ndarrays are incompatible with bool().
        oarray = (array * 255).round().astype(np.uint8)
        rgba[::, ::, 3] = oarray
    elif hasattr(opacitymap, "__array__"):  # numpy compatible object
        oarray = normalise_array(opacitymap)
        if flip:
            oarray = np.flipud(np.array(oarray))
        if oarray.max() <= 1.0:
            oarray = (oarray * 255).round().astype(np.uint8)
        rgba[::, ::, 3] = oarray
    elif opacitymap:
        raise TypeError("Unknown opacitymap type: Expected bool or ndarray")

    return rgba


"""
Functions for loading Blue Marble Next Generation 2004 dataset

- Download data tiles for each month
- Subsample and save (21600x10800)
- Split into cubemap and save (1K,2K,4K,8K,16K per tile)

Plan to upload the results of these to github releases for easier download

Also includes water mask download and tile

---

NASA Blue Marble Next Generation

https://visibleearth.nasa.gov/collection/1484/blue-marble

https://neo.gsfc.nasa.gov/view.php?datasetId=BlueMarbleNG

Getting full resolution imagery, converting to cubemap textures
Full res images come in 8 tiles of 21600x21600 total size (86400 x 43200)
There are 12 images from 2004, one for each month

This code grabs all the imagery at full resolution and converts to cubemap textures, then creates a sample animation blending the monthly images together to create a smooth transition through the year
---

STEPS:

1) If processed imagery found: return mask/textures based on passed resolution and cropping options
2) If no processed imagery found
   a) Attempt to download from github release (TODO)
   b) If not available for download, and source imagery found: process the source images then retry step 1
3) If no processed imagery or source imagery, download the sources then retry previous step

"""

bm_tiles = ["A1", "B1", "C1", "D1", "A2", "B2", "C2", "D2"]


def load_mask(res_y=None, masktype="watermask", subsample=1, cropbox=None):
    # TODO: Document args
    """
    Loads watermask/oceanmask

    Water mask / Ocean mask - using landmask_new for better quality

    https://neo.gsfc.nasa.gov/archive/bluemarble/bmng/landmask/

    Much cleaner oceanmasks without edge artifacts and errors,
    but only available in tif.gz format at full res

    https://neo.gsfc.nasa.gov/archive/bluemarble/bmng/landmask_new/
    https://neo.gsfc.nasa.gov/archive/bluemarble/bmng/landmask/world.watermask.21600x21600.A1.png

    masktype = "oceanmask" / "watermask"
    """
    if res_y is None:
        res_y = settings.FULL_RES_Y
    # Get the tiled high res images
    os.makedirs(settings.DATA_PATH / "landmask/source_tiled", exist_ok=True)
    filespec = f"{settings.DATA_PATH}/landmask/source_tiled/world.{masktype}.21600x21600.*.tif.gz"
    if len(glob.glob(filespec)) < 8:
        # Download tiles
        for t in bm_tiles:
            # https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73967/world.200402.3x21600x21600.A1.jpg
            url = f"https://neo.gsfc.nasa.gov/archive/bluemarble/bmng/landmask_new/world.{masktype}.21600x21600.{t}.tif.gz"
            # print(url)
            download(url, f"{settings.DATA_PATH}/landmask/source_tiled")

    # Calculate full image res to use for specified TEXRES
    ffn = f"{settings.DATA_PATH}/landmask/world.{masktype}.{2 * res_y}x{res_y}.png"
    if not os.path.exists(ffn):
        # Combine 4x2 image tiles into single image
        # [A1][B1][C1][D1]
        # [A2][B2][C2][D2]
        mask = np.zeros(shape=(43200, 86400), dtype=np.uint8)
        for t in bm_tiles:
            x = ord(t[0]) - ord("A")
            y = 1 if int(t[1]) == 2 else 0
            filespec = f"{settings.DATA_PATH}/landmask/source_tiled/world.{masktype}.21600x21600.{t}.tif.gz"
            paste_image(filespec, x, y, mask)

        # Save full mask in various resolutions
        for res in [(86400, 43200), (43200, 21600), (21600, 10800)]:
            r_fn = (
                f"{settings.DATA_PATH}/landmask/world.{masktype}.{res[0]}x{res[1]}.png"
            )
            if not os.path.exists(r_fn):
                # Create medium res mask image
                mimg = Image.fromarray(mask)
                if mimg.size != res:
                    mimg = mimg.resize(res, Image.Resampling.LANCZOS)
                mimg.save(r_fn)
            elif res_y == res[1]:
                image = Image.open(r_fn)
                mask = np.array(image)

            # Use this mask resolution?
            if res_y == res[1]:
                mask = np.array(mimg)

    else:
        # Use existing full mask image
        image = Image.open(ffn)
        mask = np.array(image)

    if subsample > 1:
        mask = mask[::subsample, ::subsample]
    if cropbox:
        return crop_img_lat_lon(mask, cropbox)
    return mask


def process_relief(overwrite=False, redownload=False):
    """
    Download and process relief map images

    overwrite: bool
        Always re-process from source images overwriting any existing
    redownload: bool
        Always download and overwrite source images, even if they exist
    """
    # Check for processed imagery
    # print(midx,month_name,settings.TEXRES)
    pdir = f"{settings.DATA_PATH}/relief/cubemap_{settings.TEXRES}"
    os.makedirs(pdir, exist_ok=True)
    cubemaps = len(glob.glob(f"{pdir}/*_relief_{settings.TEXRES}.png"))
    # print(cur_month, next_month)
    if not overwrite and cubemaps == 6:
        return  # Processed images present

    # Check for source images, download if not found
    colour_tex = "4_no_ice_clouds_mts_16k.jpg"
    sdir = f"{settings.DATA_PATH}/relief"
    src = f"{sdir}/{colour_tex}"
    if redownload or not os.path.exists(src):
        print("Downloading missing source images...")
        url = f"http://shadedrelief.com/natural3/ne3_data/16200/textures/{colour_tex}"
        download(url, sdir, overwrite=redownload)

    water_mask = "water_16k.png"
    if redownload or not os.path.exists(f"{sdir}/{water_mask}"):
        url = f"http://shadedrelief.com/natural3/ne3_data/16200/masks/{water_mask}"
        download(url, sdir, overwrite=redownload)

    # Land water mask for relief map
    water = np.array(Image.open(f"{sdir}/{water_mask}"))
    # Renders a jpeg downsampled view
    water = water.reshape(water.shape[0], water.shape[1], 1)

    # 50% alpha over water/ocean areas
    alphamask = 255 - water // 2

    # Open source image
    col = np.array(Image.open(f"{sdir}/{colour_tex}"))

    # Split the colour texture image into cube map tiles, including water mask
    # Export individial textures
    with closing(pushd(pdir)):
        full = np.dstack((col, alphamask))
        textures = split_tex(full, settings.TEXRES)
        # Write colour texture tiles
        for f in ["F", "R", "B", "L", "U", "D"]:
            tfn = f"{f}_relief_{settings.TEXRES}.png"
            print(tfn)
            if overwrite or not os.path.exists(tfn):
                # tex = lavavu.Image(data=textures[f])
                # tex.save(tfn)
                tex = Image.fromarray(textures[f])
                tex.save(tfn)


def process_bluemarble(when=None, overwrite=False, redownload=False, blendtex=True):
    """
    Download and process NASA Blue Marble next gen imagery

    month: int
        Month [1-12]
        If omitted will gather and process all 12 month images
    overwrite: bool
        Always re-process from source images overwriting any existing
    redownload: bool
        Always download and overwrite source images, even if they exist
    blendtex: bool
        When texture blending enabled we use images from current and next month,
        so need to check for both
    """
    midx = 0
    month_name = ""
    if when is not None:
        midx = when.month
        midx2 = midx + 1 if midx < 12 else 1
        month_name = when.strftime("%B")
        month_name2 = datetime.date(2004, midx2, 1).strftime("%B")
    # Check for processed imagery
    # print(midx,month_name,settings.TEXRES)
    pdir = f"{settings.DATA_PATH}/bluemarble/cubemap_{settings.TEXRES}"
    os.makedirs(pdir, exist_ok=True)
    cur_month = len(
        glob.glob(f"{pdir}/*_blue_marble_{month_name}_{settings.TEXRES}.png")
    )
    next_month = len(
        glob.glob(f"{pdir}/*_blue_marble_{month_name2}_{settings.TEXRES}.png")
    )
    # print(cur_month, next_month)
    if not overwrite and cur_month == 6 and (not blendtex or next_month == 6):
        return  # Full month processed images present
    if (
        not overwrite
        and len(glob.glob(f"{pdir}/*_blue_marble_*_{settings.TEXRES}.png")) == 6 * 12
    ):
        return  # Full year processed images present

    # Check for source images, download if not found
    sdir = f"{settings.DATA_PATH}/bluemarble/source_tiled"
    os.makedirs(sdir, exist_ok=True)
    all_tiles = len(glob.glob(f"{sdir}/world.2004*.3x21600x21600.*.jpg"))
    month_tiles = len(glob.glob(f"{sdir}/world.2004{midx}.3x21600x21600.*.jpg"))
    months = range(1, 13)
    if midx > 0:
        # Get current and next month to allow blending
        if blendtex:
            months = [midx, midx2]
        else:
            months = [midx]
    if redownload or month_tiles < 8 and all_tiles < 8 * 12:
        print("Downloading missing source images...")
        os.makedirs(f"{settings.DATA_PATH}/bluemarble/source_full", exist_ok=True)
        # Still checks for existing files, but compares size with server copy, which takes time
        for m in months:
            dt = datetime.date(2004, m, 1)
            month = dt.strftime("%B")
            ym = dt.strftime("%Y%m")
            print(month)
            # 21600x10800 1/4 resolution single images (2km grid)
            url = f"https://neo.gsfc.nasa.gov/archive/bluemarble/bmng/world_2km/world.{ym}.3x21600x10800.jpg"
            print(f" - {url}")
            filename = download(
                url,
                f"{settings.DATA_PATH}/bluemarble/source_full",
                overwrite=redownload,
            )

            # Tiles are as above with .[ABCD][12].jpg (500m grid)
            # Download monthly tiles
            for t in bm_tiles:
                # https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73967/world.200402.3x21600x21600.A1.jpg
                url = f"https://neo.gsfc.nasa.gov/archive/bluemarble/bmng/world_500m/world.{ym}.3x21600x21600.{t}.jpg"
                # https://neo.gsfc.nasa.gov/archive/bluemarble/bmng/world_500m/world.200401.3x21600x21600.A1.jpg
                print(url)
                filename = download(url, sdir, overwrite=redownload)
                print(filename)

    # Load full water mask image
    mask = load_mask()

    # 50% alpha over water/ocean areas
    alphamask = mask // 2 + 128

    # Split the colour texture image into cube map tiles
    full = np.zeros(shape=(43200, 86400, 3), dtype=np.uint8)
    for m in months:
        dt = datetime.date(2004, m, 1)
        month = dt.strftime("%B")
        print(f"Processing images for {month}...")
        ym = dt.strftime("%Y%m")
        if settings.TEXRES > 4096:
            # Combine 4x2 image tiles into single image
            # [A1][B1][C1][D1]
            # [A2][B2][C2][D2]
            for t in bm_tiles:
                x = ord(t[0]) - ord("A")
                y = 1 if int(t[1]) == 2 else 0
                paste_image(f"{sdir}/world.{ym}.3x21600x21600.{t}.jpg", x, y, full)
        else:
            # Medium resolution, full image is detailed enough for 4096^2 textures and below
            img = Image.open(
                f"{settings.DATA_PATH}/bluemarble/source_full/world.{ym}.3x21600x10800.jpg"
            )
            full = np.array(img)

        # Set ocean to semi-transparent
        full4 = np.dstack((full, alphamask))

        # Export individial textures
        if (
            overwrite
            or len(glob.glob(f"{pdir}/*_blue_marble_{month}_{settings.TEXRES}.png"))
            != 6
        ):
            with closing(pushd(pdir)):
                print(" - Splitting")
                textures = split_tex(full4, settings.TEXRES)
                # Write colour texture tiles
                for f in ["F", "R", "B", "L", "U", "D"]:
                    tfn = f"{f}_blue_marble_{month}_{settings.TEXRES}.png"
                    print(" - ", tfn)
                    if overwrite or not os.path.exists(tfn):
                        tex = lavavu.Image(data=textures[f])
                        tex.save(tfn)


def process_gebco(overwrite=False, redownload=False):
    """
    # Full res GEBCO .nc grid

    This function generates cubemap sections at the desired resolution from the full res GEBCO dataset

    https://www.bodc.ac.uk/data/hosted_data_systems/gebco_gridded_bathymetry_data/

    Combined topo / bath full dataset (86400 x 43200):
    - See https://download.gebco.net/
    - NC version: https://www.bodc.ac.uk/data/open_download/gebco/gebco_2020/zip/
    - Sub-ice topo version: https://www.bodc.ac.uk/data/open_download/gebco/gebco_2023_sub_ice_topo/zip/
    """
    subsampled = len(
        glob.glob(f"{settings.DATA_PATH}/gebco/gebco_equirectangular_*_x_*")
    )
    cubemap = len(
        glob.glob(f"{settings.DATA_PATH}/gebco/gebco_cubemap_{settings.GRIDRES}.npz")
    )
    if not overwrite and subsampled == 2 and cubemap == 1:
        return  # Processed data exists

    """
    #Download from github releases
    #TODO: create release and upload these files
    #TODO2: move subsampling and export functions from GEBCO.ipynb to this module
    url = f"https://github.com/ACCESS-NRI/visualisations/releases/download/v0.0.1/gebco_cubemap_{settings.GRIDRES}.npz"
    raise(Exception("TODO: upload gebco cubemap data to github releases!"))
    filename = utils.download(url, "./data/gebco")
    """

    # Attempt to load full GEBCO
    if not os.path.exists(settings.GEBCO_PATH):
        print(f"Please update the path to GEBCO_2020.nc for {settings.GEBCO_PATH=}")
        print("https://www.bodc.ac.uk/data/open_download/gebco/gebco_2020/zip/")
        raise (FileNotFoundError("Missing GEBCO path/file"))

    ds = xr.open_dataset(settings.GEBCO_PATH)

    # Subsampled full equirectangular datasets
    # (keep in lat/lon)
    # Export subsampled equirectangular data for regional clipping at lower res
    def export_subsampled(ss):
        os.makedirs(settings.DATA_PATH / "gebco", exist_ok=True)
        fn = f"{settings.DATA_PATH}/gebco/gebco_equirectangular_{86400 // ss}_x_{43200 // ss}"
        print(fn)
        if overwrite or not os.path.exists(fn + ".npz"):
            height_ss = ds["elevation"][::ss, ::ss].to_numpy()
            height_ss = np.flipud(height_ss)
            np.savez_compressed(fn, elevation=height_ss)

    export_subsampled(4)
    export_subsampled(2)

    # Cubemap (units = Mm)
    # Subsample to a reasonable resolution for our grid resolution
    SS = ds["elevation"].shape[0] // (settings.GRIDRES * 2) - 1
    height = ds["elevation"][::SS, ::SS].to_numpy()
    height = np.flipud(height)

    # Convert from M to Mm
    height = height * 1e-6

    # Split the equirectangular array into cube map tiles
    # (cache/load to save time)
    fn = f"{settings.DATA_PATH}/gebco/gebco_cubemap_{settings.GRIDRES}"
    if os.path.exists(fn + ".npz"):
        heights = np.load(fn + ".npz")
    else:
        heights = split_tex(height, settings.GRIDRES)
        np.savez_compressed(fn, **heights)
