# runs richdem fill routine, created a 'filled' file
# also creates a geotiff showing the regions that are filled

import numpy as np
import rasterio as rio
from rasterio.rio.helpers import resolve_inout
import richdem as rd

print("Starting")

# paramters
input_data_file = "Mark//trimmed.tif"
output_file_results = "Mark//fill_results.tif"
output_file_diff = "Mark//diff_results.tif"

# load data
f = rio.open(input_data_file)
v = f.read(1)
v_original = np.copy(v)
profile = f.profile
trans = f.transform
f.close()

print("Setting Rich DEM array...")
if not profile['nodata'] == None:
    rd_v = rd.rdarray(v, no_data=profile["nodata"])
else:
    rd_v = rd.rdarray(v, no_data=-9999)

print("Filling depressions...")
rd.FillDepressions(rd_v, epsilon=True, in_place=True)

print("Writing output...")
resolve_inout(overwrite=True)
profile.update({"dtype": "float32"})

# Write the NumPy array to a GeoTIFF file
with rio.open(output_file_results, 'w', **profile) as dst:
    dst.write(rd_v, 1)

diff = v_original - np.array(rd_v)
diff = np.where(diff!=0,1,0)

# Write the NumPy array to a GeoTIFF file
profile.update({"dtype": "uint8", "nodata": 0})
with rio.open(output_file_diff, 'w', **profile) as dst:
    dst.write(diff.astype(rio.uint16), 1)

print("Files created successfully!")