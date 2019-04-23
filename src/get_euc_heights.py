#!/usr/bin/env python

"""
Cluster the height of Eucs using K-Means clustering

Data from -> https://landscape.jpl.nasa.gov/

That's all folks.
"""

__author__ = "Martin De Kauwe"
__version__ = "1.0 (23.04.2019)"
__email__ = "mdekauwe@gmail.com"

import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
from sklearn.cluster import KMeans
import gdal

def get_heights(fdir, fn):

    src_ds = gdal.Open(os.path.join(fdir, fn))
    band = src_ds.GetRasterBand(1)

    cols = src_ds.RasterXSize
    rows = src_ds.RasterYSize
    transform = src_ds.GetGeoTransform()
    xOrigin = transform[0]
    yOrigin = transform[3]
    pixelWidth = transform[1]
    pixelHeight = -transform[5]
    ulx, xres, xskew, uly, yskew, yres  = src_ds.GetGeoTransform()
    lrx = ulx + (src_ds.RasterXSize * xres)
    lry = uly + (src_ds.RasterYSize * yres)

    lats = np.linspace(uly, lry, rows)
    lons = np.linspace(ulx, lrx, cols)

    lonx, laty = np.meshgrid(lats, lons)

    latx = np.ones((len(lats),len(lons))).shape

    data = band.ReadAsArray(0, 0, cols, rows)

    """
    # gdalwarp -tr 0.00833333 0.00833333 Global_l3c_error_map.tif \
    #                                    Global_l3c_error_map_inter.tif
    fn_error = "Global_l3c_error_map_inter.tif"

    src_ds = gdal.Open(os.path.join(fdir, fn_error))
    band_error = src_ds.GetRasterBand(1)

    cols_error = src_ds.RasterXSize
    rows_error = src_ds.RasterYSize
    transform = src_ds.GetGeoTransform()
    (ulx_error, xres_error,
     xskew_error, uly_error,
     yskew_error, yres_error)  = src_ds.GetGeoTransform()
    lrx_error = ulx_error + (src_ds.RasterXSize * xres_error)
    lry_error = uly_error + (src_ds.RasterYSize * yres_error)
    lats_error = np.linspace(lry_error, uly_error, rows_error)
    data_error = band_error.ReadAsArray(0, 0, cols_error, rows_error)

    # Screen by error dataset, set to zero as we mask this below
    data = np.where(data_error < 0.0, 0.0, data)
    """

    idy = np.argwhere((lats>=-43.6345972634) & (lats<-10.6681857235))
    idx = np.argwhere((lons>=113.338953078) & (lons<153.569469029))

    aus = data[idy.min():idy.max(),idx.min():idx.max()]
    aus_lat = lats[idy.min():idy.max()]
    aus_lon = lons[idx.min():idx.max()]

    #plt.imshow(np.flipud(aus))
    #plt.colorbar()
    #plt.show()

    return (aus, aus_lat, aus_lon)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

if __name__ == "__main__":

    fdir = "data/euc_locations"
    fn = "euc_latlong.csv" # from Nathalie Butt
    df = pd.read_csv(os.path.join(fdir, fn))

    fdir = "data/height"
    #gdalwarp -tr 0.25 0.25 Simard_Pinto_3DGlobalVeg_JGR.tif \
    #                       Simard_Pinto_3DGlobalVeg_JGR_degraded.tif
    #fn = "Simard_Pinto_3DGlobalVeg_JGR_degraded.tif"
    fn = "Simard_Pinto_3DGlobalVeg_JGR.tif"

    (aus, aus_lat, aus_lon) = get_heights(fdir, fn)

    ofname = "data/height/euc_heights.csv"
    if os.path.exists(ofname):
        os.remove(ofname)

    f = open(ofname, "a")
    print("lat,lon,height", file=f)
    for i in range(len(df)):

        print(i,"/",len(df))
        lat = df["latitude"].values[i]
        lon = df["longitude"].values[i]

        r = find_nearest(aus_lat, lat)
        c = find_nearest(aus_lon, lon)

        print("%f,%f,%f" % (lat, lon, aus[r,c]), file=f)

    f.close()

    df = pd.read_csv(ofname)
    df = df[df.height>0.0]
    df.to_csv(ofname, index=False)
