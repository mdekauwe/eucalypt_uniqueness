#!/usr/bin/env python

"""
Explore the austraits db a bit

That's all folks.
"""
__author__ = "Martin De Kauwe"
__version__ = "1.0 (17.07.2019)"
__email__ = "mdekauwe@gmail.com"

import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main(df):

    # drop empty rows...
    df = df[df.species_name != '']
    df = df.dropna(subset=['species_name'])

    # weird issue of mixed casting string & float, fix that
    df['species_name'] = df['species_name'].astype(str)

    species = np.unique(df.species_name)
    cnt = 0
    for spp in species:
        #print(cnt, spp)
        cnt += 1

    """
    dfx = df.dropna(subset=['leaf_area'])
    species = np.unique(dfx.species_name)
    cnt = 0
    for spp in species:
        print(cnt, spp)
        cnt += 1

    plt.boxplot(dfx.leaf_area * 1e-6)
    plt.show()
    """

    dfx = df.dropna(subset=['plant_height'])
    species = np.unique(dfx.species_name)
    cnt = 0
    for spp in species:
        print(cnt, spp)
        cnt += 1

    plt.hist(dfx.plant_height)
    plt.show()

if __name__ == "__main__":

    fdir = "data/austraits/"
    fn = "austraits_eucalypt_reoranised.csv"
    df = pd.read_csv(os.path.join(fdir, fn))

    main(df)
