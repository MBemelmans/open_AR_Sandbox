# creates a geotiff based on an array loaded from file.
import numpy as np
import rasterio as rio
from rasterio.transform import from_origin

print("Starting")

# paramters
input_data_file = "Development//Mark//sandbox_topo.npy"
output_file = "Development//Mark//sandbox.tif"

# arbitrary CRS, coordinates and resolution
CRS = "EPSG:4326"
top_left_x_coord = -61.66 # degrees for geographic, metres for projected (eg UTM)
top_left_y_coord = 16.04
x_resolution = .001 # degrees for geographic, metres for projected (eg UTM)
y_resolution =.001

# load data
sandbox_data = np.load(input_data_file)

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