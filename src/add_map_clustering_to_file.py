#!/usr/bin/env python

"""
Cluster the isotope data/MAP space using K-Means clustering

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

def compare_within_hydroclimate(df):
    """
    Use K-Mean clustering to figure out the unique classes to compare across
    """

    df_euc = df[df['orig_spp'].str.contains('Eucalyptus')]

    # extracted from
    # http//www.naturalearthdata.com/download/110m/cultural/ne_110m_admin_0_countries.zip
    # https://gist.github.com/graydon/11198540
    df_euc = df_euc[(df_euc.latitude >= -43.6345972634) &
                    (df_euc.latitude <= -10.6681857235) &
                    (df_euc.longitude >= 113.338953078) &
                    (df_euc.longitude <= 153.569469029)]

    df1 = df_euc[['map','g1']]
    data = df1.as_matrix()

    # create kmeans object
    kmeans = KMeans(n_clusters=4)

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

    df["clust"] = np.nan
    for i in range(len(df)):

        if df.map[i] >= clus_1_low and df.map[i] < clus_1_hig:
            df.loc[i,'clust'] = 1
        elif df.map[i] >= clus_2_low and df.map[i] < clus_2_hig:
            df.loc[i,'clust'] = 2
        elif df.map[i] >= clus_3_low and df.map[i] < clus_3_hig:
            df.loc[i,'clust'] = 3
        elif df.map[i] >= clus_4_low and df.map[i] < clus_4_hig:
            df.loc[i,'clust'] = 4

    return (df)

if __name__ == "__main__":

    fdir = "data/g1/processed"
    df = pd.read_csv(os.path.join(fdir, "g1_isotope_screened_mapai.csv"))
    df = compare_within_hydroclimate(df)

    df.to_csv(os.path.join(fdir, "g1_isotope_screened_mapai_clustered.csv"),
              index=False)
