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

def cluster_eucs():

    """
    # create kmeans object
    MYSEED = 5
    kmeans = KMeans(n_clusters=4, random_state=MYSEED)

    # fit kmeans object to data
    kmeans.fit(data)

    # print location of clusters learned by kmeans object
    #print(kmeans.cluster_centers_)

    y_km = kmeans.fit_predict(data)

    #plt.scatter(data[y_km ==0,0], data[y_km == 0,1], s=100, c='red')
    #plt.scatter(data[y_km ==1,0], data[y_km == 1,1], s=100, c='black')
    #plt.scatter(data[y_km ==2,0], data[y_km == 2,1], s=100, c='blue')
    #plt.scatter(data[y_km ==3,0], data[y_km == 3,1], s=100, c='cyan')
    #plt.show()

    clus_1_low, clus_1_hig = np.min(data[y_km ==0,0]), np.max(data[y_km ==0,0])
    clus_2_low, clus_2_hig = np.min(data[y_km ==1,0]), np.max(data[y_km ==1,0])
    clus_3_low, clus_3_hig = np.min(data[y_km ==2,0]), np.max(data[y_km ==2,0])
    clus_4_low, clus_4_hig = np.min(data[y_km ==3,0]), np.max(data[y_km ==3,0])

    # Clusters aren't sorted, fix this
    (clus_1_low, clus_2_low,
     clus_3_low, clus_4_low) = sorted( (clus_1_low, clus_2_low, clus_3_low,
                                        clus_4_low))
    # Clusters aren't sorted, fix this
    (clus_1_hig, clus_2_hig,
     clus_3_hig, clus_4_hig) = sorted( (clus_1_hig, clus_2_hig, clus_3_hig,
                                        clus_4_hig))

    print(clus_1_low, clus_2_low,clus_3_low, clus_4_low)
    print(clus_1_hig, clus_2_hig,clus_3_hig, clus_4_hig)
    """


if __name__ == "__main__":

    fdir = "data/height/"
    fn = "euc_heights.csv"
    df = pd.read_csv(os.path.join(fdir, fn))
