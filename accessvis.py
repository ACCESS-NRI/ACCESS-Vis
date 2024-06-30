#https://github.com/sunset1995/py360convert
# #!pip install py360convert
import numpy as np
import py360convert
from PIL import Image
import os
Image.MAX_IMAGE_PIXELS = 1061683200

def latlon_to_3D(lat, lon, alt=0):
    """
    Convert lat/lon coord to 3D coordinate for visualisation
    Uses simple spherical earth rather than true ellipse
    see http://www.mathworks.de/help/toolbox/aeroblks/llatoecefposition.html
    https://stackoverflow.com/a/20360045
    """
    return latlon_to_3D_true(lat, lon, alt, flattening=0.0)

def latlon_to_3D_true(lat, lon, alt=0, flattening=1.0/298.257223563):
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
    FF     = (1.0-np.float64(flattening))**2
    C      = 1/np.sqrt(cos_lat**2 + FF * sin_lat**2)
    S      = C * FF

    x = (rad * C + alt) * cos_lat * np.cos(lon_r)
    y = (rad * C + alt) * cos_lat * np.sin(lon_r)
    z = (rad * S + alt) * sin_lat

    #Coord order swapped to match our coord system
    return np.array([y, z, x])
    #return (x, y, z)

def split_tex(data, res, flip=[]):
    """
    Convert a texture image from equirectangular to a set of 6 cubemap faces
    (requires py360convert)
    """
    if len(data.shape) == 2:
        data = data.reshape(data.shape[0],data.shape[1],1)
    channels = data.shape[2]
    #Convert equirectangular to cubemap
    out = py360convert.e2c(data, face_w=res, mode='bilinear', cube_format='dict')
    tiles = {}
    for o in out:
        #print(o, out[o].shape)
        tiles[o] = out[o].reshape(res,res,channels)
        if True: #o in flip:
            tiles[o] = np.flipud(tiles[o])
            #tiles[o] = np.fliplr(tiles[o])
    return tiles

def draw_latlon_grid(base_img, out_fn, lat=30, lon=30, linewidth=5, colour=0):
    """
    Create lat/lon grid image over provided base image
    """
    from PIL import Image
    
    #Open base image
    image = Image.open(base_img)

    # Set the gridding interval
    x_div = 360. / lat #degrees grid in X [0,360]
    y_div = 180. / lon #degree grid in Y [-90,90]
    interval_x = round(image.size[0] / x_div)
    interval_y = round(image.size[1] / y_div)

    #Vertical lines
    l = round(linewidth / 2)
    for i in range(0, image.size[0], interval_x):
        for j in range(image.size[1]):
            for k in range(-l,l):
                if i+k < image.size[0]:
                    image.putpixel((i+k, j), colour)
    #Horizontal lines
    for i in range(image.size[0]):
        for j in range(0, image.size[1], interval_y):
            #image.putpixel((i, j), colour)
            for k in range(-l,l):
                if j+k < image.size[1]:
                    image.putpixel((i, j+k), colour)
        
    display(image)
    image.save(out_fn)

def latlon_to_uv(lat, lon):
    """
    Convert a decimal longitude, latitude coordinate
    to a tex coord in an equirectangular image
    """
    #X/u E-W Longitude - [-180,180]
    u = 0.5 + lon/360.0

    #Y/v N-S Latitude  - [-90,90]
    v = 0.5 - lat/180.0

    return u,v

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

def crop_img_uv(img, top_left, bottom_right):
    """
    Crop an image (PIL or numpy array) based on corner coords
    Provide coords as texture coords in [0,1]
    """
    u0 = top_left[0]
    u1 = bottom_right[0]
    v0 = top_left[1]
    v1 = bottom_right[1]
    #Swap coords if order incorrect
    if u0 > u1:
        u0, u1 = u1, u0
    if v0 > v1:
        v0, v1 = v1, v0
    #Supports numpy array or PIL image
    if isinstance(img, np.ndarray):
        #Assumes [lat][lon]
        lat = int(v0*img.shape[0]), int(v1*img.shape[0])
        lon = int(u0*img.shape[1]), int(u1*img.shape[1])
        print(lat, lon)
        return img[lat[0] : lat[1], lon[0] : lon[1]]
    elif hasattr(img, 'crop'):
        area = (int(u0*img.size[0]), int(v0*img.size[1]), int(u1*img.size[0]), int(v1*img.size[1]))
        return img.crop(area)
    else:
        print("Unknown type: ", type(img))

def crop_img_lat_lon(img, top_left, bottom_right):
    """
    Crop an equirectangular image (PIL or numpy array) based on corner coords
    Provide coords as lat/lon coords in decimal degrees
    """
    a = latlon_to_uv(*top_left)
    b = latlon_to_uv(*bottom_right)
    return crop_img_uv(img, a, b)

