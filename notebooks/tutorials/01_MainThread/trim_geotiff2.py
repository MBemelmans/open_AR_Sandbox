# trims geo tiff based on parameters/ranges

import numpy as np
import rasterio as rio

print("Starting")

# paramters
input_data_file = "Development//Mark//sandbox.tif"
output_file = "Development//Mark//trimmed.tif"
boundary_file = "Development//Mark//boundaries.tif"
option = 'Trim' #Simple or Trim
# option = 'Simple' #Simple or Trim

# load data
f = rio.open(input_data_file)
v = f.read(1)
profile = f.profile
trans = f.transform
f.close()

x_max = v.shape[1]
y_max = v.shape[0]


# set ranges % of size of DEM, top, bottom, left, right. Can have multiple ranges for each
# 2 values for for up to but not including, subject to integer rounding.
t1 = [.05]
t2 = [.10]
b1 = [.9]
b2 = [1]
l1 = [.0]
l2 = [.05]
r1 = [.90]
r2 = [.91]

# set range for values above. Can have multiple ranges for each
tr = [[0, x_max]]
br = [[0, x_max]]
lr = [[0, y_max]]
rr = [[0, y_max]]

# boundaries
b = np.zeros_like(v)

for i, j in enumerate(br): # go through all ranges
        
    #range limits
    by1 = int(y_max * b1[i])
    by2 = int(y_max * b2[i])
    b[by1:by2, br[i][0]: br[i][1]] = 1

for i, j in enumerate(tr): # go through all ranges

    ty1 = int(y_max * t1[i])
    ty2 = int(y_max * t2[i])
    b[ty1:ty2, tr[i][0]: tr[i][1]] = 1

for i, j in enumerate(lr): # go through all ranges

    lx1 = int(x_max * l1[i])
    lx2 = int(x_max * l2[i])
    b[lr[i][0]: lr[i][1], lx1: lx2] = 1

for i, j in enumerate(rr): # go through all ranges
    rx1 = int(x_max * r1[i])
    rx2 = int(x_max * r2[i])
    b[rr[i][0]: rr[i][1], rx1: rx2] = 1

profile.update({"nodata": 0})

# Write the NumPy array to a GeoTIFF file
with rio.open(boundary_file, 'w', **profile) as dst:
    dst.write(b, 1)


if option == 'Simple':
    #bottom
    by1 = int(y_max * b1[0])
    by2 = int(y_max * b2[0])

    min_value = np.min(v[by1:by2])

    # Find the indexes of the minimum value
    indexes = np.argwhere(v == min_value)
    i = 0
    while not by1 <= indexes[i][0] <= by2:
        i+=1
    b = indexes[i,0]

    #top
    ty1 = int(t1[0] * y_max)
    ty2 = int(t2[0] * y_max)

    min_value = np.min(v[ty1:ty2])

    # Find the indexes of the minimum value
    indexes = np.argwhere(v == min_value)
    i = 0
    while not ty1 <= indexes[i][0] <= ty2:
        i+=1
    t = indexes[i,0]

    #left
    lx1 = int(x_max * l1[0])
    lx2 = int(x_max * l2[0])

    min_value = np.min(v[:, lx1:lx2])

    # Find the indexes of the minimum value
    indexes = np.argwhere(v == min_value)
    i = 0
    while not lx1 <= indexes[i][1] <= lx2:
        i+=1
    l = indexes[i,1]

    #right
    rx1 = int(x_max * r1[0])
    rx2 = int(x_max * r2[0])

    min_value = np.min(v[:, rx1:rx2])

    # Find the indexes of the minimum value
    indexes = np.argwhere(v == min_value)
    i = 0
    while not rx1 <= indexes[i][1] <= rx2:
        i+=1
    r = indexes[i,1]

    v[0:t,:] = -9999
    v[b+1:y_max, :] = -9999
    v[:, 0:l] = -9999
    v[:, r+1:x_max] = -9999
    profile.update({"nodata": -9999})
    
    # Write the NumPy array to a GeoTIFF file
    with rio.open(output_file, 'w', **profile) as dst:
        dst.write(v, 1)


elif option == "Trim":
    #bottom
    for i, j in enumerate(br): # go through all ranges

        # bottom            
        #range limits
        by1 = int(y_max * b1[i])
        by2 = int(y_max * b2[i])

        # go through range
        for k in range(br[i][0], br[i][1], 1):
            min = np.min(v[by1:by2, k])
            l = np.where(np.isclose(v[by1:by2, k], min, atol = 1e-10))[0][0] + by1
            v[l:y_max,k] = -9999

        # top
        #range limits
        ty1 = int(y_max * t1[i])
        ty2 = int(y_max * t2[i])

        # go through range
        for k in range(tr[i][0], tr[i][1], 1):
            min = np.min(v[ty1:ty2, k])
            l = np.where(np.isclose(v[ty1:ty2, k], min, atol = 1e-10))[0][0] + ty1
            v[0:l+1, k] = -9999

        # left
        #range limits
        lx1 = int(x_max * l1[i])
        lx2 = int(x_max * l2[i])

        # go through range
        for k in range(lr[i][0], lr[i][1], 1):
            min = np.min(v[k, lx1:lx2])
            l = np.where(np.isclose(v[k, lx1:lx2], min, atol = 1e-10))[0][0] + lx1
            v[k, 0:l+1] = -9999

        # right
        #range limits
        rx1 = int(x_max * r1[i])
        rx2 = int(x_max * r2[i])

        # go through range
        for k in range(rr[i][0], rr[i][1], 1):
            min = np.min(v[k, rx1:rx2])
            l = np.where(np.isclose(v[k, rx1:rx2], min, atol = 1e-10))[0][0] + rx1
            v[k, l:x_max] = -9999


    profile.update({"nodata": -9999})

    # Write the NumPy array to a GeoTIFF file
    with rio.open(output_file, 'w', **profile) as dst:
        dst.write(v, 1)

print("Finished!")