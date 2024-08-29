# creates a geotiff based on a function for testing purposes

import numpy as np
import rasterio as rio
from rasterio.transform import from_origin
from math import cos, pi, sin

print("Starting")

# paramters
input_data_file = "Development//Mark//sandbox_topo.npy"
output_file = "Development//Mark//generated.tif"
CRS = "EPSG:4326"
top_left_x_coord = -61.66 # degrees for geographic, metres for projected (eg UTM)
top_left_y_coord = 16.04
x_resolution = .01 # degrees for geographic, metres for projected (eg UTM)
y_resolution =.01

# load data
sandbox_data = np.load(input_data_file)

x_max = np.shape(sandbox_data)[0]
y_max = np.shape(sandbox_data)[1]
x_mid = int(x_max)/2
y_mid = int(y_max)/2
for x in range (x_max):
    x1 = (x - x_mid)/x_max*2
    for y in range(y_max):
        y1 = (y - y_mid)/y_max*2
        # v = x1**2 * cos(pi*x1)*y1**2 * cos(pi*y1)*100
        v = (cos(pi*x1) + cos(pi*y1)) * 100
        v = 2-(cos(pi/2*x1) + cos(pi/2*y1)) + (cos(pi*x1) + cos(pi*y1))/2
        sandbox_data[x,y] = v
        # if x == 100:
        #     print(v) 
#np.savetxt('Mark/sb1.txt', sandbox_data, delimiter=',')
# sandbox_data = np.where(sandbox_data <0, sandbox_data, 0)

# Define the geospatial information
transform = from_origin(top_left_x_coord, top_left_y_coord, x_resolution, y_resolution)

# Specify the profile
profile = {
    "driver": "GTiff",
    "height": sandbox_data.shape[0],
    "width": sandbox_data.shape[1],
    "count": 1,  # number of bands
    "dtype": "float32",
    "crs": CRS,  
    "transform": transform, 
    "no_data": -1
}
# Write the NumPy array to a GeoTIFF file
with rio.open(output_file, 'w', **profile) as dst:
    dst.write(sandbox_data, 1)

print("File created successfully!")