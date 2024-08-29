# runs richdem flow and streams routine, creates a flow file and a streams file

import numpy as np
import rasterio as rio
from rasterio.rio.helpers import resolve_inout
import richdem as rd

print("Starting")

# paramters
input_data_file = "Mark//trimmed.tif"
stream_file = "Mark//stream.tif"
flow_file = "Mark//flow.tif"
stream_threshold = 100

# load data
f = rio.open(input_data_file)
v = f.read(1)
profile = f.profile
trans = f.transform
f.close()

print("Setting Rich DEM array...")
if not profile['nodata'] == None:
    rd_v = rd.rdarray(v, no_data=profile["nodata"])
else:
    rd_v = rd.rdarray(v, no_data=-9999)

resolve_inout(overwrite=True)
# profile.update({"dtype": "float32"})

#rd_dem_a.geotransform = (0,1,0,0,0,-1) # Defines affine. Just prevents warning messages. No material difference

print("Creating flows...")
rd_flow_v = rd.FlowProportions(dem=rd_v, method="D8")

print("Determining accumulations...")
rd_acc_v = rd.FlowAccumulation(dem=rd_v, method="D8")

print("Filtering streams...")
rd_str_v = np.where(rd_acc_v > stream_threshold, 1, 0)

# this changes the flow directions to match QGIS r.stream.extract. Needed to run in laharz
rd_flow_v = np.argmax(rd_flow_v == 1, axis=2)
rd_flow_v = (12 - rd_flow_v) % 8 +1 # to match QGIS r.stream.extract

print("Writing output...")

with rio.open(stream_file, 'w', **profile) as dst:
    dst.write(rd_str_v, 1)

with rio.open(flow_file, 'w', **profile) as dst:
    dst.write(rd_flow_v, 1)

print("Files created successfully!")