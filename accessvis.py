#https://github.com/sunset1995/py360convert
# #!pip install py360convert
import numpy as np
import py360convert

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
    #Create lat/lon grid image over provided base image
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

