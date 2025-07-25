# https://github.com/sunset1995/py360convert
# #!pip install py360convert
import datetime
import glob
import gzip
import math
import os
from collections import defaultdict
from contextlib import closing
from io import BytesIO
from pathlib import Path

import matplotlib
import numpy as np
import py360convert
import quaternion as quat
import xarray as xr
from PIL import Image

from .utils import download, is_notebook, pushd

# Before lavavu import...
# OpenGL context setup for gadi
hostname = os.getenv("HOSTNAME", "")
gadi = "gadi.nci.org.au" in hostname
if gadi:
    # Enable headless
    if os.getenv("PBS_NGPUS", "0") == "0":
        # via osmesa on CPU
        os.environ["LV_CONTEXT"] = "osmesa"
    else:
        # via moderngl/EGL with GPU
        os.environ["LV_CONTEXT"] = "moderngl"

import lavavu  # noqa: E402
from lavavu import tracers  # noqa: E402

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
    DATA_URL = "https://object-store.rc.nectar.org.au/v1/AUTH_685340a8089a4923a71222ce93d5d323/accessvis"

    # INSTALL_PATH is used for files such as sea-water-normals.png
    INSTALL_PATH = Path(__file__).parents[0]

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


def set_resolution(val):
    """
    Sets the resolution of the following:
         settings.RES, settings.TEXRES, settings.FULL_RES_Y, settings.GRIDRES

    Parameters
    ----------
    val: int
        Texture and geometry resolution
        1=low ... 4=high
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


def lonlat_to_3D(lon, lat, alt=0):
    """
    Convert lat/lon coord to 3D coordinate for visualisation
    Uses simple spherical earth rather than true ellipse
    see http://www.mathworks.de/help/toolbox/aeroblks/llatoecefposition.html
    https://stackoverflow.com/a/20360045
    """
    return lonlat_to_3D_true(lon, lat, alt, flattening=0.0)


def latlon_to_3D(lat, lon, alt=0, flattening=0.0):
    """
    Convert lon/lat coord to 3D coordinate for visualisation

    Provided for backwards compatibility as main function now reverses arg order of
    (lat, lon) to (lon, lat)
    """
    return lonlat_to_3D_true(lon, lat, alt, flattening)


def lonlat_to_3D_true(lon, lat, alt=0, flattening=1.0 / 298.257223563):
    """
    Convert lon/lat coord to 3D coordinate for visualisation
    Now using longitude, latitude, altitude order for more natural argument order
    longitude=x, latitude=y, altitude=z

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
    z = (rad * S + alt) * sin_lat * np.ones_like(lon_r)
    # np.ones_like: if lon is array and lat is not, z is a const (shape != x,y)

    # Coord order swapped to match our coord system
    # NOTE: will need to call np.dstack on result
    # if passing in arrays and want result as 3d vertices
    return np.array([y, z, x])


def earth_vertices_to_3D(vertices, true_earth=False):
    """
    Convert lon/lat/alt coords to 3D coordinates for visualisation

    Same as calling lonlat_to_3D but takes and returns an array of 3d coordinates
    instead of separate arrays for each coord

    Parameters
    ----------
    vertices: ndarray
        Coord vertices in order lon,lat,alt
    true_earth: boolean
        Pass true to apply flattening factor for WGS84 elliptical earth shape
        Default is to use a perfect spherical earth
    Returns
    -------
    np.ndarray
        Vertices as 3D cartesian coords
    """
    shape = vertices.shape
    vertices = vertices.reshape((-1, 3))
    if true_earth:
        arr = np.dstack(
            lonlat_to_3D_true(vertices[::, 0], vertices[::, 1], vertices[::, 2])
        )
    else:
        arr = np.dstack(lonlat_to_3D(vertices[::, 0], vertices[::, 1], vertices[::, 2]))
    return np.dstack(arr).reshape(shape)


def split_tex(data, res, flipud=False, fliplr=False):
    """
    Convert a texture image from equirectangular to a set of 6 cubemap faces
    (requires py360convert)

    Parameters
    ----------
    data: np.ndarray
        data to split into cube faces.
    res: int
        Resolution of each face
    flipud: bool|str
        if up/down should be flipped (or which faces should be flipped)
    fliplr: bool|str
        if left/right should be flipped (or which faces should be flipped)
    """
    if isinstance(flipud, bool):
        # Can provide a list/string of faces to flip, or just a single bool to flip all
        flipud = "FRBLUD" if flipud else ""
    if isinstance(fliplr, bool):
        fliplr = "FRBLUD" if fliplr else ""
    if len(data.shape) == 2:
        data = data.reshape(data.shape[0], data.shape[1], 1)
    channels = data.shape[2]
    # Convert equirectangular to cubemap
    out = py360convert.e2c(data, face_w=res, mode="bilinear", cube_format="dict")
    tiles = {}
    for i, o in enumerate(out):
        print(o, out[o].shape, o in flipud, o in fliplr)
        tiles[o] = out[o].reshape(res, res, channels)
        if o in flipud:
            tiles[o] = np.flipud(tiles[o])
        if o in fliplr:
            tiles[o] = np.fliplr(tiles[o])
    return tiles


def draw_lonlat_grid(base_img, out_fn, lon=30, lat=30, linewidth=5, colour=0):
    """
    Create lat/lon grid image over provided base image

    Parameters
    ----------
    base_img:
        Path to the input image
    out_fn:
        Path to the output image
    lon: int
        Distance between lines of longitude (degrees)
    lat: int
        Distance between lines of latitude (degrees)
    linewidth: int
        pixel width of the lines
    colour:
        Colour of the lines
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


def lonlat_to_uv(lon, lat):
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


def lonlat_to_pixel(lon, lat, width, height):
    """
    Convert a decimal latitude/longitude coordinate
    to a raster image pixel coordinate for given width/height
    """
    u, v = lonlat_to_uv(lon=lon, lat=lat)
    return uv_to_pixel(u, v, width, height)


def crop_img_uv(img, cropbox):
    """
    Crop an image (PIL or numpy array) based on corner coords

    Parameters
    ----------
    img: np.ndarray or PIL.ImageFile.ImageFile
        The image to be cropped (numpy or PIL).
    cropbox:
        Coordinates of the corners of the box to crop.
        ((u0,v0), (u1,v1))
        u0, u1, v0, v1 are typically in the range [0,1] but may be higher/lower.
        If u0<0, we assume the image wraps around to the right side of img.
        Likewise if u1>1 it wraps around the left side of img.
        This is useful if you want a region of earth going over the 180/-180 line.
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
        pieces = []
        if u0 < 0:  # wraps around the left side
            underflow = int((1 + u0) * img.shape[1])
            piece = img[lat[0] : lat[1], underflow:]
            pieces.append(piece)

        # in the 0..1 region
        piece = img[lat[0] : lat[1], max(0, lon[0]) : lon[1]]
        pieces.append(piece)

        if u1 > 1:  # wraps around the right side
            overflow = int((u1 - 1) * img.shape[1])
            piece = img[lat[0] : lat[1], 0:overflow]
            pieces.append(piece)

        arr = np.hstack(pieces)

        return arr

    elif isinstance(img, Image.Image):
        crop_regions = []
        if u0 < 0:  # wraps around the left side
            crop_regions.append(
                (
                    int((1 + u0) * img.size[0]),
                    int(v0 * img.size[1]),
                    int(img.size[0]),
                    int(v1 * img.size[1]),
                )
            )
        # in the 0..1 region
        crop_regions.append(
            (
                int(max(0, u0) * img.size[0]),
                int(v0 * img.size[1]),
                int(min(1, u1) * img.size[0]),
                int(v1 * img.size[1]),
            )
        )
        if u1 > 1:  # wraps around the left side
            crop_regions.append(
                (
                    0,
                    int(v0 * img.size[1]),
                    int((u1 - 1) * img.size[0]),
                    int(v1 * img.size[1]),
                )
            )

        max_height = crop_regions[0][3] - crop_regions[0][1]
        total_width = sum(i[2] - i[0] for i in crop_regions)
        new_im = Image.new("RGB", (total_width, max_height))

        x_offset = 0
        for area in crop_regions:
            im = img.crop(area)
            new_im.paste(im, (x_offset, 0))
            x_offset += im.size[0]

        return new_im

    else:
        print("Unknown type: ", type(img))


def crop_img_lon_lat(img, top_left, bottom_right):
    """
    Crop an equirectangular image (PIL or numpy array) based on corner coords
    Provide coords as lat/lon coords in decimal degrees

    Parameters
    ----------
    img: np.ndarray or PIL.ImageFile.ImageFile
        The image to be cropped (numpy or PIL).
    top_left: tuple
        (lon, lat) Top left coordinates of the crop box.
        If top_left or bottom_right are None, it returns the mask of the full earth.
    bottom_right: tuple
        (lon, lat) Bottom right coordinates of the crop box.
        If top_left or bottom_right are None, it returns the mask of the full earth.
    Returns:
    ----------
    np.ndarray or PIL.ImageFile.ImageFile
    """
    a = lonlat_to_uv(*top_left)
    b = lonlat_to_uv(*bottom_right)
    return crop_img_uv(img, (a, b))


def sphere_mesh(radius=1.0, quality=256):
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
    """
    # Generate cube face grid
    lv = lavavu.Viewer()
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

    return sdata


def cubemap_sphere_vertices(
    radius=1.0,
    resolution=None,
    heightmaps=None,
    vertical_exaggeration=1.0,
    topo_exaggeration=1,
    bathy_exaggeration=1,
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
    topo_exaggeration: number
        Multiplier to topography height only
    bathy_exaggeration: number
        Multiplier to bathymetry height only
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

    # For each cube face...
    minmax = []
    eminmax = []
    for f in ["F", "R", "B", "L", "U", "D"]:
        # For texcoords
        tij = np.linspace(0.0, 1.0, resolution, dtype="float32")
        tii, tjj = np.meshgrid(tij, tij)  # 2d grid
        # For vertices
        ij = np.linspace(-1.0, 1.0, resolution, dtype="float32")
        ii, jj = np.meshgrid(ij, ij)  # 2d grid
        zz = np.zeros(shape=ii.shape, dtype="float32")  # 3rd dim
        if f == "F":  ##
            vertices = np.dstack((ii, jj, zz + 1.0))
            tc = np.dstack((tii, tjj))
        elif f == "B":
            vertices = np.dstack((ii, jj, zz - 1.0))
            tc = np.dstack((1.0 - tii, tjj))
        elif f == "R":
            vertices = np.dstack((zz + 1.0, jj, ii))
            tc = np.dstack((1.0 - tii, tjj))
        elif f == "L":  ##
            vertices = np.dstack((zz - 1.0, jj, ii))
            tc = np.dstack((tii, tjj))
        elif f == "U":
            vertices = np.dstack((ii, zz + 1.0, jj))
            tc = np.dstack((tii, 1.0 - tjj))
        elif f == "D":  ##
            vertices = np.dstack((ii, zz - 1.0, jj))
            tc = np.dstack((tii, tjj))
        # Normalise the vectors to form spherical patch  (normalised cube)
        V = vertices.ravel().reshape((-1, 3))
        norms = np.sqrt(np.einsum("...i,...i", V, V))
        norms = norms.reshape(resolution, resolution, 1)
        verts = vertices / norms

        # Scale and apply surface detail?
        if heightmaps:
            # Optional split vertical exaggeration for bathymetry and topography
            if topo_exaggeration != 1 or bathy_exaggeration != 1:
                # Load ocean mask for split topo/bathy exag
                masktype = "oceanmask"
                res = settings.TEXRES
                mask_tile = f"{settings.DATA_PATH}/landmask/cubemap_{res}/{f}_{masktype}_{res}.png"
                image = Image.open(mask_tile)
                outres = (resolution, resolution)
                if image.size != outres:
                    image = image.resize(outres, Image.Resampling.LANCZOS)
                mask = np.array(image)
                if f != "U":
                    mask = np.flipud(mask)
                if f in ["R", "B"]:  # , 'U']:
                    mask = np.fliplr(mask)
                exag = np.zeros(shape=mask.shape, dtype=float)
                exag += topo_exaggeration  # Set array to land exag
                exag[mask == 0] = bathy_exaggeration  # Apply ocean exag with mask
                # Apply radius and heightmap
                hmap = (heightmaps[f].reshape(outres) * exag).reshape(
                    (resolution, resolution, 1)
                )
                verts *= hmap + radius
                eminmax += [hmap.min(), hmap.max()]
            else:
                # Apply radius and heightmap
                verts *= heightmaps[f] * vertical_exaggeration + radius
                eminmax += [
                    heightmaps[f].min() * vertical_exaggeration,
                    heightmaps[f].max() * vertical_exaggeration,
                ]
            minmax += [heightmaps[f].min(), heightmaps[f].max()]
        else:
            # Apply radius only
            verts *= radius
        sdata[f] = verts
        sdata[f + "_texcoord"] = tc

    # Save height range
    if len(minmax) == 0:
        minmax = [0, vertical_exaggeration]
        eminmax = [0, 1.0]
    minmax = np.array(minmax)
    eminmax = np.array(eminmax)
    sdata["range"] = (minmax.min(), minmax.max())
    # Also exaggerated range
    sdata["exag_range"] = (eminmax.min(), eminmax.max())

    # Hemisphere crop?
    half = resolution // 2
    if hemisphere == "N":  # U
        del sdata["D"]  # Delete south
        for f in ["F", "R", "B", "L"]:
            sdata[f] = sdata[f][half::, ::, ::]  # Crop bottom section
            sdata[f + "_texcoord"] = sdata[f + "_texcoord"][half::, ::, ::]
    elif hemisphere == "S":  # D
        del sdata["U"]  # Delete north
        for f in ["F", "R", "B", "L"]:
            sdata[f] = sdata[f][0:half, ::, ::]  # Crop top section
            sdata[f + "_texcoord"] = sdata[f + "_texcoord"][0:half, ::, ::]
    elif hemisphere == "E":  # R
        del sdata["L"]  # Delete W
        for f in ["F", "B", "U", "D"]:
            sdata[f] = sdata[f][::, half::, ::]  # Crop left section
            sdata[f + "_texcoord"] = sdata[f + "_texcoord"][::, half::, ::]
    elif hemisphere == "W":  # L
        del sdata["R"]  # Delete E
        for f in ["F", "B", "U", "D"]:
            sdata[f] = sdata[f][::, 0:half, ::]  # Crop right section
            sdata[f + "_texcoord"] = sdata[f + "_texcoord"][::, 0:half, ::]
    elif hemisphere == "EW":  # B
        del sdata["F"]  # Delete prime meridian
        for f in ["R", "L"]:
            sdata[f] = sdata[f][::, 0:half, ::]  # Crop right section
            sdata[f + "_texcoord"] = sdata[f + "_texcoord"][::, 0:half, ::]
        for f in ["U", "D"]:
            sdata[f] = sdata[f][0:half, ::, ::]  # Crop top section
            sdata[f + "_texcoord"] = sdata[f + "_texcoord"][0:half, ::, ::]
    elif hemisphere == "WE":  # F
        del sdata["B"]  # Delete antimeridian
        for f in ["R", "L"]:
            sdata[f] = sdata[f][::, half::, ::]  # Crop left section
            sdata[f + "_texcoord"] = sdata[f + "_texcoord"][::, half::, ::]
        for f in ["U", "D"]:
            sdata[f] = sdata[f][half::, ::, ::]  # Crop bottom section
            sdata[f + "_texcoord"] = sdata[f + "_texcoord"][half::, ::, ::]

    return sdata


def load_topography_cubemap(
    resolution=None,
    radius=6.371,
    vertical_exaggeration=1,
    topo_exaggeration=1,
    bathy_exaggeration=1,
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
    topo_exaggeration: number
        Multiplier to topography height only
    bathy_exaggeration: number
        Multiplier to bathymetry height only
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
    process_gebco(cubemap=True, resolution=resolution)  # Ensure data exists
    fn = f"{settings.DATA_PATH}/gebco/gebco_cubemap_{resolution}.npz"
    if not os.path.exists(fn):
        raise (Exception("GEBCO data not found"))
    heights = np.load(fn)

    # Apply to cubemap sphere
    return cubemap_sphere_vertices(
        radius,
        resolution,
        heights,
        vertical_exaggeration,
        topo_exaggeration,
        bathy_exaggeration,
        hemisphere=hemisphere,
    )


def load_topography(
    resolution=None, subsample=1, top_left=None, bottom_right=None, bathymetry=True
):
    """
    Load topography from pre-saved equirectangular data, can be cropped for regional plots

    Parameters
    ----------
    resolution:
        Resolution of the topography in the Y direction.
        defaults to settings.FULL_RES_Y (default 10800)
        Note: res_x = 2 * res_y
    subsample: int
        Selects every Nth item from the mask.
        e.g. subsample=2 takes every second lat and lon value.
    top_left: tuple
        (lon, lat) Top left coordinates of the crop box.
        If top_left or bottom_right are None, it returns the mask of the full earth.
    bottom_right: tuple
        (lon, lat) Bottom right coordinates of the crop box.
        If top_left or bottom_right are None, it returns the mask of the full earth.
    bathymetry:
        If the ocean floor should be shown.
    """
    if resolution is None:
        resolution = settings.FULL_RES_Y
    heights = None
    if heights is None:
        process_gebco(cubemap=False, resolution=resolution)  # Ensure data exists
        basefn = f"gebco_equirectangular_{resolution * 2}_x_{resolution}.npz"
        fn = f"{settings.DATA_PATH}/gebco/{basefn}"
        if not os.path.exists(fn):
            raise (Exception("GEBCO data not found"))
        else:
            heights = np.load(fn)
            heights = heights["elevation"]

    if subsample > 1:
        heights = heights[::subsample, ::subsample]
    if top_left and bottom_right:
        heights = crop_img_lon_lat(
            heights, top_left=top_left, bottom_right=bottom_right
        )

    # Bathymetry?
    if not bathymetry or not isinstance(bathymetry, bool):
        # Ensure resolution matches topo grid res
        # res_y = resolution//4096 * 10800
        # res_y = max(resolution,2048) // 2048 * 10800
        mask = load_mask(
            res_y=resolution,
            subsample=subsample,
            top_left=top_left,
            bottom_right=bottom_right,
            masktype="oceanmask",
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


def lonlat_minmax(a, b):
    """
    Takes two lon/lat tuples (lon, lat).
    Returns two tuples containing the smallest, largest of each lon/lat.
    """

    lon1, lat1 = a
    lon2, lat2 = b
    if lon1 > lon2:
        lon1, lon2 = lon2, lon1
    if lat1 > lat2:
        lat1, lat2 = lat2, lat1
    return (lon1, lat1), (lon2, lat2)


def plot_region(
    lv=None,
    top_left=None,
    bottom_right=None,
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
    **kwargs,
):
    """
    Plots a flat region of the earth with topography cropped to specified region (lat/lon bounding box coords)
    uses bluemarble textures by default and sets up seasonal texture blending based on given or current date and time

    Uses lat/lon as coordinate system, so no use for polar regions, scales heights to equivalent
    TODO: support using km as unit or other custom units instead with conversions from lat/lon

    TODO: top_left/bottom_right: defaults to full earth if None passed, write a test for this

    TODO: Implement lighting, waves, shaders arguments (currently unused)

    TODO: Implement bathy_exaggeration and topo_exaggeration
    Note:
    If you want to plot data, you should use `plot_region_data()` after calling this function.
    If you want to plot data in a region, but continue to display the entire 3D earth, you should use `earth_2d_plot()` instead.

    Parameters
    ----------
    lv: lavavu.Viewer
        The viewer object to plot with
    top_left: tuple
        (lon, lat) Top left coordinates of the crop box.
    bottom_right: tuple
        (lon, lat) Bottom right coordinates of the crop box.
    vertical_exaggeration: number
        Multiplier to topography/bathymetry height
    texture: str
        Path to textures, face label and texres will be applied with .format(), eg:
        texture='path/{face}_mytexture_{texres}.png'
        with: texture.format(face='F', texres=settings.TEXRES)
        to:'path/F_mytexture_1024.png'
    name: str
        Append this label to each face object created
    vertical_exaggeration: number
        Multiplier to topography/bathymetry height
    when: datetime
        Provide a datetime object to set the month for texture sets that vary over the year and time for
        position of sun and rotation of earth when calculating sun light position
    blendtex: bool
        When the texture set has varying seasonal images, enabling this will blend between the current month and next
        months texture to smoothly transition between them as the date changes, defaults to True
    bathymetry: bool
        If the ocean floor should be shown
    name: str
        The name of the surface
    uniforms: dict
        Provide a set of uniform variables, these can be used to pass data to a custom shader
    background: str
        Provide a background colour string, X11 colour name or hex RGB
    kwargs:
        Additional lavavu arguments.
        Covers global props, object props and uniform values
    """

    if top_left and bottom_right:
        top_left, bottom_right = lonlat_minmax(top_left, bottom_right)

    if lv is None:
        lv = lavavu.Viewer(
            border=False,
            axis=False,
            resolution=[1280, 720],
            background=background,
            fliptexture=False,
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
    topo = load_topography(
        top_left=top_left, bottom_right=bottom_right, bathymetry=bathymetry
    )
    height = np.array(topo)

    # Scale and apply vertical exaggeration
    height = height * MtoLL * vertical_exaggeration

    D = [height.shape[1], height.shape[0]]

    sverts = np.zeros(shape=(height.shape[0], height.shape[1], 3))
    lon0, lat0 = top_left if top_left else [-90, 0]
    lon1, lat1 = bottom_right if bottom_right else [90, 360]
    # TODO: Support crossing zero in longitude
    # Will probably not support crossing poles
    xy = lv.grid2d(corners=((lon0, lat1), (lon1, lat0)), dims=D)
    sverts[::, ::, 0:2] = xy
    sverts[::, ::, 2] = height[::, ::]
    assert len(sverts)

    # Default mask
    # mask_tex = f"{settings.DATA_PATH}/landmask/world.watermask.21600x10800.png"
    mask_tex = f"{settings.DATA_PATH}/landmask/world.oceanmask.21600x10800.png"

    if texture == "bluemarble":
        # TODO: support cropping tiled high res blue marble textures
        # Also download relief textures if not found or call process_bluemarble
        process_bluemarble(when, blendtex=blendtex)
        colour_tex = f"{settings.DATA_PATH}/bluemarble/source_full/world.200412.3x21600x10800.jpg"
        uniforms["bluemarble"] = True
    elif texture == "relief":
        process_relief()  # Ensure images available
        colour_tex = f"{settings.DATA_PATH}/relief/4_no_ice_clouds_mts_16k.jpg"
    else:
        colour_tex = texture

    process_landmask(texture)
    lv.texture("landmask", mask_tex)

    surf = lv.triangles(
        name, vertices=sverts, uniforms=uniforms, cullface=True, opaque=True
    )  # , fliptexture=False)

    img = Image.open(colour_tex)
    if top_left and bottom_right:
        cropped_img = crop_img_lon_lat(
            img, top_left=top_left, bottom_right=bottom_right
        )
        arr = np.array(cropped_img)
    else:
        arr = np.array(img)

    surf.texture(arr, flip=False)
    return lv


def plot_earth(
    lv=None,
    radius=6.371,
    vertical_exaggeration=10,
    topo_exaggeration=1,
    bathy_exaggeration=1,
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
    **kwargs,
):
    """
    Plots a spherical earth using a 6 face cubemap mesh with bluemarble textures
    and sets up seasonal texture blending and optionally, sun position,
    based on given or current date and time

    Parameters
    ----------
    lv: lavavu.Viewer
        The viewer object to plot with
    radius: float
        Radius of the sphere, defaults to 6.371 Earth's approx radius in Mm
    vertical_exaggeration: number
        Multiplier to topography/bathymetry height
    topo_exaggeration: number
        Multiplier to topography height only
    bathy_exaggeration: number
        Multiplier to bathymetry height only
    texture: str
        Path to textures, face label and texres will be applied with .format(), eg:
        texture="path/{face}_mytexture_{texres}.png"
        with: texture.format(face="F", texres=settings.TEXRES)
        to:"path/F_mytexture_1024.png"

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
    kwargs:
        Additional lavavu arguments.
        Covers global props, object props and uniform values
    """
    if lv is None:
        lv = lavavu.Viewer(
            border=False,
            axis=False,
            resolution=[1280, 720],
            background=background,
            fliptexture=False,
        )

    topo = load_topography_cubemap(
        settings.GRIDRES,
        radius,
        vertical_exaggeration,
        topo_exaggeration,
        bathy_exaggeration,
        hemisphere=hemisphere,
    )
    if when is None:
        when = datetime.datetime.now()
    month = when.strftime("%B")

    # Custom uniforms / additional textures
    uniforms["radius"] = radius

    # Land/ocean mask - get data
    process_landmask(texture)

    if texture == "bluemarble":
        texture = "{basedir}/bluemarble/cubemap_{texres}/{face}_blue_marble_{month}_{texres}.jpg"
        uniforms["bluemarble"] = True
        if waves is None:
            waves = True
        # Use the bluemarble land/water mask
        landmask = "{basedir}/landmask/cubemap_{texres}/{face}_watermask_{texres}.png"
    elif texture == "relief":
        process_relief()  # Ensure images available
        texture = "{basedir}/relief/cubemap_{texres}/{face}_relief_{texres}.jpg"
        # Use the relief land/water mask
        landmask = "{basedir}/landmask/cubemap_{texres}/{face}_relief_{texres}.png"

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
    hrange = np.array(topo["exag_range"])
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
        tc = topo[f + "_texcoord"]

        texfn = texture.format(
            basedir=settings.DATA_PATH, face=f, texres=settings.TEXRES, month=month
        )
        uniforms["landmask"] = landmask.format(
            basedir=settings.DATA_PATH, face=f, texres=settings.TEXRES
        )

        lv.triangles(
            name=f + name,
            vertices=verts,
            texcoords=tc,
            texture=texfn,
            fliptexture=".jpg" in texfn,
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
        lv.set_properties(
            diffuse=0.6,
            ambient=0.6,
            specular=0.3,
            shininess=0.04,
            light=[1, 1, 0.98, 1],
            lightpos=lp,
        )

    if hemisphere == "N":
        lv.rotation(90.0, 0.0, 0.0)
    elif hemisphere == "S":
        lv.rotation(-90.0, 0.0, 0.0)
    elif hemisphere == "E":
        lv.rotation(0.0, -90.0, 0.0)
    elif hemisphere == "W":
        lv.rotation(0.0, 90.0, 0.0)
    elif hemisphere == "EW":
        lv.rotation(0.0, 180.0, 0.0)
    elif hemisphere == "WE":
        lv.rotation(0.0, 0.0, 0.0)

    lv.render()  # Required to apply changes
    return lv


def lonlat_grid(longitudes, latitudes, resolution=None):
    """
    Creates a 2D geo grid from  corner coords and a sampling resolution
    Returns the longitude then latitude arrays

    Parameters
    ----------
    longitudes: ndarray or tuple
        min/max longitudes as a tuple
    latitudes: ndarray or tuple
        min/max latitudes as a tuple
    resolution: 2-tuple
        Resolution of the 2d grid: (longitude-res, latitude-res)

    Returns
    -------
    tuple (ndarray,ndarray): the grid as lon,lat
    """


def lonlat_grid_3D(
    longitudes, latitudes, altitude=0.001, resolution=None, wrap_longitude=False
):
    """
    Creates a grid of 3D vertices representing a regional 2D grid converted from lat,lon coords
    Used for plotting data arrays on a 3D earth plot
    Can provide corner coords and a sampling resolution or just the raw lat,lon coords

    Parameters
    ----------
    longitudes: ndarray or tuple
        List of longitude values or min/max longitude as a tuple
    latitudes: ndarray or tuple
        List of latitude values or min/max latitude as a tuple
    altitudes: float / ndarray
        height above sea level in Mm
        Will fill this value in a fixed height grid
        If an array is passed, will use this as the height map
    resolution: 2-tuple
        Resolution of the 2d grid, if provided will use corner points from
        longitudes,latitudes only and sample a regular grid between them of
        provided resolution
    wrap_longitude: boolean
        If the grid requires an extra coord to wraps the earth in longitude
        coords then set this to True.
        This will add a duplicate of the first longitude value as the last
        value. Useful for cell centred grids allowing passing original coord
        range, without this a manually calculated additional coord might not
        interpolate correctly on the grid.

    Returns
    -------
    ndarray: the 3D vertices of the grid
    """

    x = np.array(longitudes)
    y = np.array(latitudes)
    if resolution is not None:
        # Use corners and create a sampling grid of this resolution
        x = np.linspace(x[0], x[-1], resolution[0])
        y = np.linspace(y[0], y[-1], resolution[1])

    if wrap_longitude:
        # Joining longitude coord to wrap when using cell centre grid
        x[-1] = x[0]  # + 360
        # x = np.append(x, x[0])

    lon, lat = np.meshgrid(x, y, indexing="ij")

    arrays = lonlat_to_3D(lon, lat, altitude)

    return np.dstack(arrays)


def earth_2d_plot(
    lv,
    data=None,
    colourmap="viridis",
    opacitymap=None,
    opacitypower=1,
    longitudes=None,
    latitudes=None,
    altitude=1e-6,
    resolution=None,
    **kwargs,
):
    """
    Plots a surface on the 3D earth given an xarray DataArray

    Calls lonlat_grid_3D, creates a mesh of required resolution and plots the data as a texture

    Parameters
    ----------
    lv: lavavu.Viewer
        The viewer object to plot with
    data: xarray.DataArray, ndarray
        The data to plot, either a numpy array, in which case latitudes + longitudes must be passed to define the corners/grid
        Otherwise xarray data should only have two coords, if more need to select down to required data in 2d
        Assumes the first coordinate of the array is longitude and second is latitude
        If order is reversed, need to transpose before passing, eg: data.transpose('latitude', 'longitude')
        If omitted, will skip texturing
    colourmap: str or list or object
        If a string, provides the name of a matplotlib colourmap to use
        If a list is passed, should contain RGB[A] colours as lists or tuples in range [0,1] or [0,255],
        a matplotlib.colors.LinearSegmentedColormap will be created from the list
        Otherwise, will assume is a valid matplotlib colormap already
    opacitymap: bool or numpy.ndarray
        Set to true to use values as an opacity map, top of range will be opaque, bottom transparent
        Provide an array to use a different opacity map dataset
    opacitypower: float
        Power to apply to values when calculating opacity map,
        eg: 0.5 equivalent to opacity = sqrt(value), 2 equivalent to opacity = value*value
    longitudes: ndarray or tuple
        List of longitude values or min/max longitude as a tuple
    latitudes: ndarray or tuple
        List of latitude values or min/max latitude as a tuple
    altitudes: float
        height above sea level in Mm, defaults to 1m above sea level
        Will fill this value in a fixed height grid
    resolution: 2-tuple
        Resolution of the 2d grid, if provided will use corner points from
        data only and sample a regular grid between them of provided resolution
    kwargs:
        These kwargs are passed through to lv.surface().
    Returns
    -------
    lavavu.Object: the drawing object created

    Example
    -------

    >>> import lavavu, accessvis
    >>> lv = lavavu.Viewer()
    >>> surf = accessvis.earth_2d_plot(data, resolition=(100,100), colourmap='viridis')
    """
    if data is not None and latitudes is None or longitudes is None:
        keys = list(data.coords.keys())
        if len(keys) < 2:
            print("Data must have 2 dimensions in order longitude, latitude")
            return None
        longitudes = data[keys[0]]
        latitudes = data[keys[1]]

    # Set default blank texture - forces texcoord load
    if "texture" not in kwargs:
        kwargs["texture"] = str(settings.INSTALL_PATH / "data" / "blank.png")
    # Set default transparent colour
    if "colour" not in kwargs:
        kwargs["colour"] = "rgba(0,0,0,0)"

    lon = np.array(longitudes)
    lat = np.array(latitudes)
    if len(lon.shape) == 2 and len(lat.shape) == 2:
        # Already passed 2d grid of lon/lat values
        # Just need to convert to 3D coords
        arrays = lonlat_to_3D(lon, lat, altitude)
        grid = np.dstack(arrays)
    elif len(lon.shape) == 1 and len(lat.shape) == 1:
        # Passed 1d arrays of lon/lat, create a grid
        grid = lonlat_grid_3D(longitudes, latitudes, altitude, resolution)
    else:
        raise (ValueError("Unknown data format for latitude and longitude params"))

    surf = lv.surface(vertices=grid, **kwargs)
    if data is not None:
        imgarr = array_to_rgba(
            data, colourmap=colourmap, opacitymap=opacitymap, opacitypower=opacitypower
        )
        surf.texture(imgarr, flip=False)
    return surf


def update_earth_datetime(
    lv, when, name="", texture=None, sunlight=False, blendtex=True
):
    """
    Update date/time for texture blending and lighting

    Parameters
    ----------
    lv: lavavu.Viewer
        The earth viewer object to plot with.
    when: datetime.datetime
        timestamp the earth should reflect.
    name: str
        the name of the surfaces the texture should be applied to.
    texture:
        "bluemarble" or path to the file containing the texture.
    sunlight: bool
        True means the light source should move to the correct position to reflect the given time.
    blendtex: bool
        If the bluemarble texture should be blended between months to approximate the given date.
    """

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
        texture = "{basedir}/bluemarble/cubemap_{texres}/{face}_blue_marble_{month}_{texres}.jpg"

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
        if blendtex:
            assert os.path.exists(texfn2)
        o = f + name
        if o in lv.objects:
            obj = lv.objects[o]
            uniforms = obj["uniforms"]

            # if not "blendTex" in uniforms or uniforms["blendTex"] != texfn2:
            if obj["texture"] != texfn:
                obj["texture"] = texfn  # Not needed, but set so can be checked above
                obj.texture(texfn, flip=".jpg" in texfn)

            if blendtex and (
                "blendTex" not in uniforms or uniforms["blendTex"] != texfn2
            ):
                uniforms["blendTex"] = texfn2
                obj.texture(texfn2, flip=".jpg" in texfn2, label="blendTex")

            if not blendtex:
                factor = -1.0  # Disable blending multiple textures
            uniforms["blendFactor"] = factor

            obj["uniforms"] = uniforms

    lv.render()  # Required to render a frame which fixes texture glitch
    if sunlight:
        lv.set_properties(lightpos=sun_light(time=when))


def update_earth_texture(
    lv, label, texture, flip=False, shared=True, name="", **kwargs
):
    """
    Update texture values for a specific texture on earth model

    Parameters
    ----------
    lv: lavavu.Viewer
        The earth viewer object to plot with.
    label: str
    flip: bool
        Whether or not the textures should be flipped.
    shared: bool
        If the texture is shared between faces.
    name:
        The name of the object which the kwargs should be passed through to.
    kwargs:
        lavavu kwargs to pass through to the uniforms.
    """

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

    # lv.render()  # Required to render a frame which fixes texture glitch


def update_earth_values(lv, name="", flip=False, **kwargs):
    """
    Update uniform values on earth objects via passed kwargs

    Parameters
    ----------
    lv: lavavu.Viewer
        The earth viewer object to plot with.
    name: str
        The name of the object which the kwargs should be passed through to.
    flip: bool
        Whether or not the textures should be flipped.
    kwargs:
        Lavavu arguments to pass through to the textures.

    """
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

    # lv.render()  # Required to render a frame which fixes texture glitch


def vec_rotate(v, theta, axis):
    """
    Rotate a 3D vector about an axis by given angle

    Parameters
    ----------
    v : list/numpy.ndarray
        The 3 component vector
    theta : float
        Angle in radians
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


def lonlat_normal_vector(lon, lat):
    """
    If standing at (lat,lon), this unit vector points directly above.
    """
    return normalise(lonlat_to_3D(lat=lat, lon=lon))


def lonlat_vector_to_east(lon, lat):
    """
    If standing at (lat,lon), this unit vector points east.
    """
    start = lonlat_to_3D(lat=lat, lon=lon)
    north = lonlat_to_3D(lat=90, lon=0)
    cross = np.cross(north, start)
    if np.any(cross):  # if start != north
        return normalise(cross)
    return np.array([1, 0, 0])


def lonlat_vector_to_north(lon, lat):
    """
    If standing at (lat,lon), this unit vector points north.
    """
    start = lonlat_to_3D(lat=lat, lon=lon)
    east = lonlat_vector_to_east(lat=lat, lon=lon)
    return normalise(np.cross(start, east))


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
            ltime = datetime.datetime.now(datetime.timezone.utc).astimezone()
            ltz = ltime.tzinfo
            if now or time is None:
                # Use local time in utc zone
                time = ltime
                # time = datetime.datetime.now(tz=ltz)

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
    opacitypower=1,
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
    opacitypower: float
        Power to apply to values when calculating opacity map,
        eg: 0.5 equivalent to opacity = sqrt(value), 2 equivalent to opacity = value*value
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
    elif opacitymap is True:
        oarray = array
        if opacitypower != 1:
            oarray = np.power(oarray, opacitypower)
        # Mask NaN
        oarray = np.nan_to_num(oarray)
        oarray = (oarray * 255).astype(np.uint8)
        rgba[::, ::, 3] = oarray
    elif hasattr(opacitymap, "__array__"):
        oarray = normalise_array(opacitymap)
        if flip:
            oarray = np.flipud(oarray)
        if opacitypower != 1:
            oarray = np.power(oarray, opacitypower)
        # Mask NaN
        oarray = np.nan_to_num(oarray)
        oarray = (oarray * 255).astype(np.uint8)
        rgba[::, ::, 3] = oarray
    elif opacitymap:
        raise TypeError("Unknown opacitymap type: Expected bool or ndarray")
    else:
        # Opaque, Mask out NaN
        oarray = ~np.isnan(array) * 255
        rgba[::, ::, 3] = oarray.astype(np.uint8)

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


def load_mask(
    res_y=None, masktype="watermask", subsample=1, top_left=None, bottom_right=None
):
    """
    Loads watermask/oceanmask

    Water mask / Ocean mask - using landmask_new for better quality

    https://neo.gsfc.nasa.gov/archive/bluemarble/bmng/landmask/

    Much cleaner oceanmasks without edge artifacts and errors,
    but only available in tif.gz format at full res

    https://neo.gsfc.nasa.gov/archive/bluemarble/bmng/landmask_new/
    https://neo.gsfc.nasa.gov/archive/bluemarble/bmng/landmask/world.watermask.21600x21600.A1.png

    Parameters
    ----------
    res_y:
        defaults to settings.FULL_RES_Y (default 10800)
        Resolution of the oceanmask png file.
        res_x = 2 * res_y
    masktype: str
        "relief": lower quality
        "oceanmask" or "watermask": better quality
    subsample: int
        Selects every Nth item from the mask.
        e.g. subsample=2 takes every second lat and lon value.
    top_left: tuple
        (lon, lat) Top left coordinates of the crop box.
        If top_left or bottom_right are None, it returns the mask of the full earth.
    bottom_right: tuple
        (lon, lat) Bottom right coordinates of the crop box.
        If top_left or bottom_right are None, it returns the mask of the full earth.
    """
    if res_y is None:
        res_y = settings.FULL_RES_Y

    # Calculate full image res to use for specified TEXRES
    # (full equirectangular image - 'relief' or watermask/oceanmask for bluemarble
    if masktype == "relief":
        ffn = f"{settings.DATA_PATH}/landmask/landmask_16200_8100.png"
    else:
        ffn = f"{settings.DATA_PATH}/landmask/world.{masktype}.{2 * res_y}x{res_y}.png"
    image = Image.open(ffn)
    mask = np.array(image)

    if subsample > 1:
        mask = mask[::subsample, ::subsample]
    if top_left and bottom_right:
        return crop_img_lon_lat(mask, top_left=top_left, bottom_right=bottom_right)
    return mask


def process_relief(overwrite=False, redownload=False):
    """
    Download and process relief map images

    overwrite: bool
        Always re-process from source images overwriting any existing
    redownload: bool
        Always download and overwrite source images, even if they exist
    """

    if settings.TEXRES > 8192:
        print("WARNING: 16K textures not available for relief mode")
        settings.TEXRES = 8192

    # Check for processed imagery
    # print(midx,month_name,settings.TEXRES)
    pdir = f"{settings.DATA_PATH}/relief/cubemap_{settings.TEXRES}"
    os.makedirs(pdir, exist_ok=True)
    cubemaps = len(glob.glob(f"{pdir}/*_relief_{settings.TEXRES}.jpg"))
    # print(cur_month, next_month)
    if not overwrite and cubemaps == 6:
        return  # Processed images present

    """
    Download pre-processed data from DATA_URL
    """
    try:
        for f in ["F", "R", "B", "L", "U", "D"]:
            tfn = f"{f}_relief_{settings.TEXRES}.jpg"
            url = f"{settings.DATA_URL}/relief/cubemap_{settings.TEXRES}/{tfn}"
            download(url, pdir)
        return
    except (Exception) as e:
        print(f"Error downloading: {str(e)} attempting to generate files")

    # Below should never normally be executed as we now download from url above
    # Code still required in case we need to regenerate the data from sources

    # Check for source images, download if not found
    colour_tex = "4_no_ice_clouds_mts_16k.jpg"
    sdir = f"{settings.DATA_PATH}/relief"
    src = f"{sdir}/{colour_tex}"
    if redownload or not os.path.exists(src):
        print("Downloading missing source images...")
        url = f"http://shadedrelief.com/natural3/ne3_data/16200/textures/{colour_tex}"
        download(url, sdir, overwrite=redownload)

    # Open source image
    full = np.array(Image.open(f"{sdir}/{colour_tex}"))

    # Split the colour texture image into cube map tiles
    # Export individial textures
    with closing(pushd(pdir)):
        textures = split_tex(full, settings.TEXRES)
        # Write colour texture tiles
        for f in ["F", "R", "B", "L", "U", "D"]:
            tfn = f"{f}_relief_{settings.TEXRES}.jpg"
            print(tfn)
            if overwrite or not os.path.exists(tfn):
                # tex = lavavu.Image(data=textures[f])
                # tex.save(tfn)
                tex = Image.fromarray(textures[f])
                tex.save(tfn, quality=95)


def process_bluemarble(when=None, overwrite=False, redownload=False, blendtex=True):
    """
    Download and process NASA Blue Marble next gen imagery

    when: datetime.datetime
        If None, it defaults to the current time.
    overwrite: bool
        Always re-process from source images overwriting any existing
    redownload: bool
        Always download and overwrite source images, even if they exist
    blendtex: bool
        When texture blending enabled we use images from current and next month,
        so need to check for both
    """
    midx = 0
    if when is None:
        when = datetime.datetime.now()
    midx = when.month
    midx2 = midx + 1 if midx < 12 else 1
    month_name = when.strftime("%B")
    month_name2 = datetime.date(2004, midx2, 1).strftime("%B")
    # Check for processed imagery
    # print(midx,month_name,settings.TEXRES)
    pdir = f"{settings.DATA_PATH}/bluemarble/cubemap_{settings.TEXRES}"
    os.makedirs(pdir, exist_ok=True)
    cur_month = len(
        glob.glob(f"{pdir}/*_blue_marble_{month_name}_{settings.TEXRES}.jpg")
    )
    next_month = len(
        glob.glob(f"{pdir}/*_blue_marble_{month_name2}_{settings.TEXRES}.jpg")
    )
    # print(cur_month, next_month)
    if not overwrite and cur_month == 6 and (not blendtex or next_month == 6):
        return  # Full month processed images present
    if (
        not overwrite
        and len(glob.glob(f"{pdir}/*_blue_marble_*_{settings.TEXRES}.jpg")) == 6 * 12
    ):
        return  # Full year processed images present

    """
    Download pre-processed data from DATA_URL
    """
    try:
        for f in ["F", "R", "B", "L", "U", "D"]:
            tfn = f"{f}_blue_marble_{month_name}_{settings.TEXRES}.jpg"
            url = f"{settings.DATA_URL}/bluemarble/cubemap_{settings.TEXRES}/{tfn}"
            download(url, pdir)
            tfn = f"{f}_blue_marble_{month_name2}_{settings.TEXRES}.jpg"
            url = f"{settings.DATA_URL}/bluemarble/cubemap_{settings.TEXRES}/{tfn}"
            download(url, pdir)

        # Full equirectangular images, used for regional crops
        ddir = f"{settings.DATA_PATH}/bluemarble/source_full"
        for m in range(1, 13):
            url = f"{settings.DATA_URL}/bluemarble/source_full/world.2004{m:02}.3x21600x10800.jpg"
            download(url, ddir)
        return
    except (Exception) as e:
        print(f"Error downloading: {str(e)} attempting to generate files")

    # Below should never normally be executed as we now download from url above
    # Code still required in case we need to regenerate the data from sources

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

        # Export individial textures
        if (
            overwrite
            or len(glob.glob(f"{pdir}/*_blue_marble_{month}_{settings.TEXRES}.jpg"))
            != 6
        ):
            with closing(pushd(pdir)):
                print(" - Splitting")
                textures = split_tex(full, settings.TEXRES)
                # Write colour texture tiles
                for f in ["F", "R", "B", "L", "U", "D"]:
                    tfn = f"{f}_blue_marble_{month}_{settings.TEXRES}.jpg"
                    print(" - ", tfn)
                    if overwrite or not os.path.exists(tfn):
                        # tex = lavavu.Image(data=textures[f])
                        # tex.save(tfn)
                        tex = Image.fromarray(textures[f])
                        tex.save(tfn, quality=95)


def process_landmask(texture, overwrite=False, redownload=False):
    """
    Download and process ocean/water mask imagery

    texture: str
        "relief": lower quality but smaller.
        Otherwise it uses both watermask and oceanmask to generate the mask
    overwrite: bool
        Always re-process from source images overwriting any existing
    redownload: bool
        Always download and overwrite source images, even if they exist
    """
    res_y = settings.FULL_RES_Y
    res = settings.TEXRES

    # Load the relief mask
    if texture == "relief":
        if settings.TEXRES > 8192:
            print("WARNING: 16K textures not available for relief mode")
            res = settings.TEXRES = 8192

        sdir = f"{settings.DATA_PATH}/landmask"
        # Full equirectangular image
        maskfn = f"{sdir}/landmask_16200_8100.png"
        mask = None
        if not os.path.exists(maskfn):
            url = f"{settings.DATA_URL}/landmask/landmask_16200_8100.png"
            try:
                download(url, sdir, overwrite=redownload)
            except (Exception):
                url = "http://shadedrelief.com/natural3/ne3_data/16200/masks/water_16k.png"
                download(url, sdir, overwrite=redownload)
                image = Image.open(f"{sdir}/water_16k.png")
                mask = 255 - np.array(image)
                # Save inverted image
                mimg = Image.fromarray(mask)
                mimg.save(maskfn)

        filespec = f"{settings.DATA_PATH}/landmask/cubemap_{settings.TEXRES}/*_relief_{settings.TEXRES}.png"
        if redownload or len(glob.glob(filespec)) < 6:
            try:
                # Cubemaps
                for f in ["F", "R", "B", "L", "U", "D"]:
                    url = f"{settings.DATA_URL}/landmask/cubemap_{res}/{f}_relief_{res}.png"
                    download(url, f"{sdir}/cubemap_{res}")
            except (Exception) as e:
                print(f"Error downloading: {str(e)} attempting to generate files")

                if mask is None:
                    mask = np.array(Image.open(maskfn))

                # Split the full res image into cube map tiles
                pdir = f"{settings.DATA_PATH}/landmask/cubemap_{res}"
                with closing(pushd(pdir)):
                    textures = split_tex(mask, settings.TEXRES)
                    # Write colour texture tiles
                    for f in ["F", "R", "B", "L", "U", "D"]:
                        tfn = f"{f}_relief_{res}.png"
                        print(tfn)
                        if overwrite or not os.path.exists(tfn):
                            # tex = lavavu.Image(data=textures[f])
                            # tex.save(tfn)
                            tex = Image.fromarray(textures[f].squeeze())
                            tex.save(tfn)

    else:
        # Load full water mask images for bluemarble
        for masktype in ["watermask", "oceanmask"]:
            # First check for old files (will be flipped in R,B,U tiles)
            testfile = (
                f"{settings.DATA_PATH}/landmask/cubemap_{res}/F_{masktype}_{res}.png"
            )
            outdated = False
            if os.path.exists(testfile):
                d0 = datetime.datetime.fromtimestamp(os.path.getmtime(testfile))
                delta = datetime.datetime(2025, 5, 13) - d0
                if delta.days > 0:
                    print(f"Mask files out of date {delta.days} days, redownloading")
                    outdated = True

            # Calculate full image res to use for specified TEXRES
            filespec = f"{settings.DATA_PATH}/landmask/world.{masktype}.{2 * res_y}x{res_y}.png"
            filespec_cm = (
                f"{settings.DATA_PATH}/landmask/cubemap_{res}/*_{masktype}_{res}.png"
            )
            if (
                redownload
                or outdated
                or len(glob.glob(filespec)) < 1
                or len(glob.glob(filespec_cm)) < 6
            ):
                """
                Download pre-processed data from DATA_URL
                """
                try:
                    # Full iamges
                    url = f"{settings.DATA_URL}/landmask/world.{masktype}.{2 * res_y}x{res_y}.png"
                    download(url, f"{settings.DATA_PATH}/landmask/")
                    # Cubemaps
                    for f in ["F", "R", "B", "L", "U", "D"]:
                        url = f"{settings.DATA_URL}/landmask/cubemap_{res}/{f}_{masktype}_{res}.png"
                        download(
                            url,
                            f"{settings.DATA_PATH}/landmask/cubemap_{res}",
                            overwrite=outdated,
                        )
                    # All succeeded
                    return

                except (Exception) as e:
                    print(f"Error downloading: {str(e)} attempting to generate files")

                try:
                    # Get the tiled high res images
                    os.makedirs(
                        settings.DATA_PATH / "landmask/source_tiled", exist_ok=True
                    )
                    filespec = f"{settings.DATA_PATH}/landmask/source_tiled/world.{masktype}.21600x21600.*.tif.gz"
                    if len(glob.glob(filespec)) < 8:
                        # Download tiles
                        for t in bm_tiles:
                            # https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73967/world.200402.3x21600x21600.A1.jpg
                            # Original url, now using our own copy
                            url = f"https://neo.gsfc.nasa.gov/archive/bluemarble/bmng/landmask_new/world.{masktype}.21600x21600.{t}.tif.gz"
                            # url = f"{settings.DATA_URL}/landmask/source_tiled/world.{masktype}.21600x21600.{t}.tif.gz"
                            # print(url)
                            download(url, f"{settings.DATA_PATH}/landmask/source_tiled")
                except (Exception) as e:
                    print(
                        f"Error downloading source data: {str(e)}, unable to continue"
                    )
                    raise

                # Combine 4x2 image tiles into single image
                # [A1][B1][C1][D1]
                # [A2][B2][C2][D2]
                mask = np.zeros(shape=(43200, 86400), dtype=np.uint8)
                for t in bm_tiles:
                    x = ord(t[0]) - ord("A")
                    y = 1 if int(t[1]) == 2 else 0
                    filespec = f"{settings.DATA_PATH}/landmask/source_tiled/world.{masktype}.21600x21600.{t}.tif.gz"
                    paste_image(filespec, x, y, mask)

                # Split the full res image into cube map tiles
                pdir = f"{settings.DATA_PATH}/landmask/cubemap_{res}"
                with closing(pushd(pdir)):
                    textures = split_tex(mask, settings.TEXRES)
                    # Write colour texture tiles
                    for f in ["F", "R", "B", "L", "U", "D"]:
                        tfn = f"{f}_{masktype}_{res}.png"
                        print(tfn)
                        if overwrite or not os.path.exists(tfn):
                            # tex = lavavu.Image(data=textures[f])
                            # tex.save(tfn)
                            tex = Image.fromarray(textures[f].squeeze())
                            tex.save(tfn)

                # Save full mask in various resolutions
                for outres in [(86400, 43200), (43200, 21600), (21600, 10800)]:
                    r_fn = f"{settings.DATA_PATH}/landmask/world.{masktype}.{outres[0]}x{outres[1]}.png"
                    if not os.path.exists(r_fn):
                        # Create medium res mask image
                        mimg = Image.fromarray(mask)
                        if mimg.size != outres:
                            mimg = mimg.resize(outres, Image.Resampling.LANCZOS)
                        mimg.save(r_fn)


def process_gebco(cubemap, resolution, overwrite=False):
    """
    # Full res GEBCO .nc grid

    This function generates cubemap sections at the desired resolution from the full res GEBCO dataset.

    https://www.bodc.ac.uk/data/hosted_data_systems/gebco_gridded_bathymetry_data/

    Combined topo / bath full dataset (86400 x 43200):
    - See https://download.gebco.net/
    - NC version: https://www.bodc.ac.uk/data/open_download/gebco/gebco_2020/zip/
    - Sub-ice topo version: https://www.bodc.ac.uk/data/open_download/gebco/gebco_2023_sub_ice_topo/zip/

    Parameters
    ----------
    cubemap: bool
        Whether to download cubemap or equirectangular data.
    resolution: int
        Resolution of each cube face (e.g. 1024).
    overwrite: bool
        True means the files should be redownloaded.
    """

    if cubemap:
        fspec = f"{settings.DATA_PATH}/gebco/gebco_cubemap_{resolution}.npz"
        if not overwrite and len(glob.glob(fspec)) == 1:
            return  # Processed data exists
    else:
        fspec = f"{settings.DATA_PATH}/gebco/gebco_equirectangular_{resolution * 2}_x_{resolution}"
        if not overwrite and len(glob.glob(fspec)) == 1:
            return  # Processed data exists

    """
    Download pre-processed data from DATA_URL
    """
    try:
        if cubemap:
            url = f"{settings.DATA_URL}/gebco/gebco_cubemap_{resolution}.npz"
        else:
            url = f"{settings.DATA_URL}/gebco/gebco_equirectangular_{resolution * 2}_x_{resolution}.npz"
        ddir = f"{settings.DATA_PATH}/gebco"
        download(url, ddir)
        return
    except (Exception) as e:
        print(f"Error downloading: {str(e)} attempting to generate files")

    # Below should never normally be executed as we now download from url above
    # Code still required in case we need to regenerate the data from sources

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
        heights = split_tex(height, settings.GRIDRES, flipud=True)
        np.savez_compressed(fn, **heights)


class EarthTracers(tracers.Tracers):
    """
    Override the tracer class get_positions to use 3D coords
    Allow us to track particles and pass in data in lon,lat,alt but do
    the final plotting in 3D cartesian coords

    EarthTracers(grid, count=1000, lowerbound=None, upperbound=None, limit=None, age=4, respawn_chance=0.2, speed_multiply=1.0, height=0.0, label='', viewer=lv)
    """

    def get_positions(self):
        lon = self.positions[::, 0]
        lat = self.positions[::, 1]
        if self.dims > 2:
            alt = self.positions[::, 2]
            positions = lonlat_to_3D(lon=lon, lat=lat, alt=alt).T
        else:
            positions = lonlat_to_3D(lon=lon, lat=lat, alt=self.height).T
        return positions


def plot_vectors_xr(
    lv,
    xr_coordinates: list[dict],
    u=None,
    v=None,
    w=None,
    lat_name="lat",
    lon_name="lon",
    alt_name="alt",
    rescale_alt=True,
    alt_scale_factor=1.0,
    label="xr_vectors",
    **kwargs,
):
    """
    Plots vectors the earth.

    Parameters
    ----------
    lv: lavavu.Viewer
        The viewer object to plot with.
    xr_coordinates: list[dict]
        Each element is a dict of coord_name:approx_coord_value.
        It automatically uses the nearest coordinate values.
    u: xr.DataArray
        vector magnitude in the east direction
    v: xr.DataArray
        vector magnitude in the north direction
    w: xr.DataArray
        vector magnitude in the upwards direction
    lat_name: str
        The name of the latitude coordinate (e.g. lat, latitude)
    lon_name: str
        The name of the longitude coordinate (e.g. lon, longitude)
    alt_name: str
        The name of the altitude coordinate (e.g. alt, altitude)
        If it cannot find match alt_name, altitude defaults 0.
    rescale_alt: bool
        If True, it re-scales the altitude to make it higher off the surface.
        If False, it uses the provided altitude (or zero if missing).
        If False, the altitude coordinate must be in meters above sea level.
    alt_scale_factor: float
        How far above the ground should the top-most arrow be?
        Defaults to 1 (million meters).
    label: str
        The name of the lavavu vectors.
        Change this if plotting multiple sets of vectors at the same time.
    kwargs:
        Lavavu vector properties, e.g. colour, scalevectors.
        https://lavavu.github.io/Documentation/Property-Reference.html#object-vector

    Returns
    -------

    lavavu.Object:
        The object defining all the vectors in lavavu
    """

    vertices = []  # where the arrow base is
    vectors = []  # direction the arrow is pointing

    # Here we find all unique coordinates,
    # then reducing the size of u,v,w to only these values.
    # Then I load this DataArray Subset.
    # Otherwise it takes too long to run.
    coord_indexers = defaultdict(list)
    for d in xr_coordinates:
        for key, val in d.items():
            coord_indexers[key].append(val)
    coord_indexers = {key: np.unique(val) for key, val in coord_indexers.items()}

    if u is not None:
        u = u.sel(
            method="nearest", indexers=coord_indexers
        ).load()  # making u smaller and loading for speed
        any_xr = u  # We pick any xarray dataset and use this to get lat/lon/alt later
    if v is not None:
        v = v.sel(method="nearest", indexers=coord_indexers).load()
        any_xr = v
    if w is not None:
        w = w.sel(method="nearest", indexers=coord_indexers).load()
        any_xr = w
    if u is None and v is None and w is None:
        raise ValueError("You must provide at least one of u,v,w.")

    if rescale_alt and alt_name in any_xr.coords:
        min_alt = np.min(any_xr[alt_name]).to_numpy()
        max_alt = np.max(any_xr[alt_name]).to_numpy()
    else:
        min_alt = 0
        max_alt = 1

    # Looping through to get vertex/vector for each arrow.
    for coords in xr_coordinates:
        arr = any_xr.sel(method="nearest", **coords)
        lat = arr[lat_name].to_numpy()
        lon = arr[lon_name].to_numpy()
        try:
            alt = arr[alt_name].to_numpy()
        except KeyError:
            # If no altitude, the height is controlled by alt_scale_factor.
            alt = max_alt

        # Basis vector directions, on the 3d model.
        normal = lonlat_normal_vector(lat=lat, lon=lon)
        east = lonlat_vector_to_east(lat=lat, lon=lon)
        north = lonlat_vector_to_north(lat=lat, lon=lon)

        if rescale_alt:
            # The bottom-most vector is ground level, top-most is at a height of alt_scale_factor above ground
            relative_height = (alt - min_alt) / (max_alt - min_alt)
            vert = (
                lonlat_to_3D(lat=lat, lon=lon)
                + normal * relative_height * alt_scale_factor
            )
            vertices.append(vert)
        else:
            vert = lonlat_to_3D(lat=lat, lon=lon, alt=alt * 1e-6)
            # 1e-6 is to convert from meters to Million meters (3d Scale)
            vertices.append(vert)

        # If array is missing/is nan, we set the vector in that direction to 0.
        if u is None:
            uu = 0
        else:
            uu = np.nan_to_num(u.sel(method="nearest", **coords))
        if v is None:
            vv = 0
        else:
            vv = np.nan_to_num(v.sel(method="nearest", **coords))
        if w is None:
            ww = 0
        else:
            ww = np.nan_to_num(w.sel(method="nearest", **coords))
        vec = east * uu + north * vv + normal * ww
        vectors.append(vec)

    # creating the vectors object if it does not exist.
    try:
        lv_vectors = lv.objects[label]
        lv_vectors.clear()
    except KeyError:
        lv_vectors = lv.vectors(label, colour="red", lit=False)

    for prop, val in kwargs.items():  # Vector settings (e.g. colour)
        lv_vectors[prop] = val

    # Plotting the vectors
    lv_vectors.vertices(vertices)
    lv_vectors.vectors(vectors)

    return lv_vectors


def plot_shapefile(lv, fn, features=None, alt=1e-6, label="shape_", **kwargs):
    """
    Uses a shapefile to display a boundary.

    Parameters
    ----------
    lv: lavavu.Viewer
        The viewer object to plot with.
    fn: str
        The path to the shapefile.
    features: list[str]|str|None
        Which shapefile shapes to display.
        Multiple countries/states/regions may be in the same file. Select desired shapes.
        str: only display this shape.
        list[str]: display all shapes in the list.
        None: display all shapes.
        Note: it is case sensitive and will silently skip features it cannot find.
        e.g. ["Australian Capital Territory", "Northern Territory"]
    alt: float
        The height above the earth that lines should hover (million meters).
    label: str
        The prefix of the name of the lavavu lines.
        Change this if plotting multiple regions with the same name.
    kwargs:
        Lavavu line properties, e.g. colour, linewidth.
        https://lavavu.github.io/Documentation/Property-Reference.html#object-line

    Returns
    -------
    dict[str, lavavu.Object]
        key: the label of each lavavu object
        value: The displayed lavavu object
    """
    import shapefile  # shapefile is an optional dependency.

    sf = shapefile.Reader(fn)

    objects = dict()

    if isinstance(features, str):
        features = [str]

    # Iterating through each shape, e.g. each state
    for entry, shape in zip(sf.iterShapeRecords(), sf.shapes()):
        # finding name
        if features is None:
            name = label + str(entry.record[0])
        else:
            for rec in entry.record:
                if features is None or str(rec) in features:
                    name = label + str(rec)  # a name is found
                    break
            else:  # No name is found. Continue to next entry
                continue

        # converting to 3d Coordinates.
        lons, lats = zip(*shape.points)
        verts = lonlat_to_3D(lat=lats, lon=lons, alt=alt).T

        # Iterate through each part, e.g. each island.
        for i, (p1, p2) in enumerate(zip(shape.parts[:-1], shape.parts[1:])):
            vv = verts[p1:p2]
            points = lv.lines(name + "_" + str(i), **kwargs, link=True, depthtest=True)
            points.vertices(vv)

            objects[name + "_" + str(i)] = points
    return objects


def plot_cross_section(
    lv,
    data,
    start,
    end,
    resolution=100,
    label="cross-section",
    **kwargs,
):
    """
    Adds a vertical cross section of data.

    Parameters
    ----------
    lv: lavavu.Viewer
        The viewer object to plot with.
    data: np.ndarray
        An array of shape [width, height, RGB(A)].
    start:
        (lon, lat, altitude) or (lon, lat).
        data[0,0] corrosponds to the start position.
        Default altitude is 1 (million meters).
    end: tuple[float, float, Optional[float]]
        (lon, lat, altitude) or (lon, lat).
        data[-1,-1] corrosponds to the end position.
        Default altitude is 0 (million meters).
    resolution: int
        The number of mesh points between start and end.
        Increase if start/end cover a large range/if you see corners.
    label: str
        The name of the lavavu surface.
        Change this if plotting multiple cross sections.
    kwargs:
        Lavavu surface properties, e.g. lit=False.
        https://lavavu.github.io/Documentation/Property-Reference.html#object-surface

    Returns
    -------
    lavavu.Object
        The lavavu surface being created.
    """

    surf = lv.triangles(label, colour="rgba(255,255,255,0)", **kwargs)

    try:
        max_alt = start[2]
    except IndexError:
        max_alt = 1

    try:
        min_alt = end[2]
    except IndexError:
        min_alt = 0

    # Support crossing zero in longitude
    if start[0] > end[0]:
        end[0] += 360

    # Calculating the position of all vertices.
    lats = np.linspace(start[1], end[1], resolution)
    lons = np.linspace(start[0], end[0], resolution)
    lower = lonlat_to_3D(lat=lats, lon=lons, alt=min_alt)
    upper = lonlat_to_3D(lat=lats, lon=lons, alt=max_alt)
    vertices = np.dstack((lower, upper)).T

    surf.vertices(vertices)

    surf.texture(data)
    # lv.reload() #Necessary?
    return surf


def plot_region_data(lv, data, start, end, label="data-surface", **kwargs):
    """
    Plots data on a 2D region of the earth.
    Use this to plot data after plot_region() is called.
    Note: If you want to plot data in a region, but continue to display the entire earth, you may want to use earth_2d_plot() instead.

    Parameters
    ----------
    lv: lavavu.Viewer
        The viewer object to plot with.
    data: np.ndarray
        An array of shape [width, height, RGB(A)].
    start: tuple[number, number]
        (lon, lat) or (lon, lat, alt)
        data[0,0] corresponds to the start position.
        alt is optional and defaults to 1e-5.
    end: tuple[number, number]
        (lon, lat)
        data[-1,-1] corresponds to the end position.
    alt: number
        height above sea level to display the data.
    label: str
        The name of the lavavu surface.
        Change this if plotting multiple surfaces.
    kwargs:
        Lavavu surface properties, e.g. lit=False.
        https://lavavu.github.io/Documentation/Property-Reference.html#object-surface

    Returns
    -------
    lavavu.Object
        The lavavu surface being created.
    """

    if len(start) == 3:
        alt = start[2]
    elif len(end) == 3:
        alt = end[2]
    else:
        alt = 1e-5

    surf = lv.triangles(
        label, colour="rgba(255,255,255,0)", **kwargs  # allows transparent data
    )

    lons = np.linspace(start[0], end[0], 5)
    lats = np.linspace(start[1], end[1], 5)
    lons, lats = np.meshgrid(lons, lats)
    alts = np.full_like(lons, alt)

    vertices = np.stack([lons, lats, alts], axis=-1)
    surf.vertices(vertices)

    surf.texture(data)

    return surf
