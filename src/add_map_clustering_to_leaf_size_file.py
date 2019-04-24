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

def compare_within_hydroclimate(df, leaf=False):
    """
    Use K-Mean clustering to figure out the unique classes to compare across
    """

    df_eucs = df[df['Genus species'].str.contains('Eucalyptus') &\
                 (df['Growth form'] == "Tree") &\
                 (df['woody_non-woody'] == "woody")]

    df_eucs = df_eucs[~np.isnan(df_eucs['Leaf size (cm2)'])]
    df_eucs = df_eucs[~np.isnan(df_eucs['MAP'])]

    df1 = df_eucs[['MAP','Leaf size (cm2)']]
    data = df1.as_matrix()

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
    #sys.exit()

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

        if df.MAP[i] >= clus_1_low and df.MAP[i] < clus_1_hig:
            df.loc[i,'clust'] = 1
        elif df.MAP[i] >= clus_2_low and df.MAP[i] < clus_2_hig:
            df.loc[i,'clust'] = 2
        elif df.MAP[i] >= clus_3_low and df.MAP[i] < clus_3_hig:
            df.loc[i,'clust'] = 3
        elif df.MAP[i] >= clus_4_low and df.MAP[i] < clus_4_hig:
            df.loc[i,'clust'] = 4

    return (df)

if __name__ == "__main__":

    fdir = "data/leaf_size"
    fn = "aal4760-Wright-SM_Data_Set_S1.csv"
    df = pd.read_csv(os.path.join(fdir, fn))
    df = compare_within_hydroclimate(df)
    df.to_csv(os.path.join(fdir, "leaf_size_clustered.csv"),
              index=False)
