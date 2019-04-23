#!/usr/bin/env python

"""
Add MAP and AI from CRU data

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

def add_ppt_and_ai(df):

    lats_needed = df["latitude"].values
    lons_needed = df["longitude"].values

    # Fix units to help match the 0.5 degree data
    lats_neededx = [float(x_round(float(i))) for i in lats_needed]
    lons_neededx = [float(x_round(float(i))) for i in lons_needed]

    nrows = 360
    ncols = 720

    latitudes = np.linspace(-89.75, 89.75, nrows)
    longitudes = np.linspace(-179.75, 179.75, ncols)

    idir = "/Users/mdekauwe/research/CRU_TS_v4_bioclim_MAT_MAP_AI"
    mapx = np.fromfile(os.path.join(idir,
                       "MAP_1960_2010.bin")).reshape(nrows, ncols)
    aix = np.fromfile(os.path.join(idir,
                      "AI_1960_2010.bin")).reshape(nrows, ncols)



    #dfo = df.copy()
    df["map"] = np.nan
    df["ai"] = np.nan

    for i in range(len(df)):

        lat = float(x_round(float(df["latitude"].values[i])))
        lon = float(x_round(float(df["longitude"].values[i])))

        r = np.where(latitudes==lat)[0][0]
        c = np.where(longitudes==lon)[0][0]

        df.loc[i,'map'] = mapx[r,c]
        df.loc[i,'ai'] = aix[r,c]
        #print(i, r, c, lat, lon, mapx[r,c], aix[r,c], df['map'][i], df['ai'][i])


    return (df)

def x_round(x):
    # Need to round to nearest .25 or .75 to match the locations in CRU
    val = round(x * 4.0) / 4.0
    valx = str(val).split(".")
    v1 = valx[0]
    v2 = valx[1]

    if v2 <= "25":
        v2 = "25"
    else:
        v2 = "75"
    valx = float("%s.%s" % (v1, v2))

    return (valx)


if __name__ == "__main__":

    fdir = "data/g1"
    odir = "data/g1/processed/"
    df = pd.read_csv(os.path.join(fdir, "g1_isotope_screened.csv"))
    df = add_ppt_and_ai(df)

    if not os.path.exists(odir):
        os.makedirs(odir)

    df.to_csv(os.path.join(odir, "g1_isotope_screened_mapai.csv"), index=False)

    df = pd.read_csv(os.path.join(fdir, "g1_leaf_gas_exchange.csv"),
                     encoding='latin-1')
    df = add_ppt_and_ai(df)

    if not os.path.exists(odir):
        os.makedirs(odir)

    df.to_csv(os.path.join(odir, "g1_leaf_gas_exchange_mapai.csv"), index=False)
