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

def cluster_heights(df):

    data = df.height.values.reshape(-1,1) # need 2d array

    # create kmeans object
    MYSEED = 5
    kmeans = KMeans(n_clusters=5, random_state=MYSEED)


    # fit kmeans object to data
    kmeans.fit(data)

    # print location of clusters learned by kmeans object
    clusters = sorted(np.array([i[0] for i in kmeans.cluster_centers_]))
    print(clusters)

    return (clusters)

if __name__ == "__main__":

    fdir = "data/height/"
    fn = "euc_heights.csv"
    df = pd.read_csv(os.path.join(fdir, fn))

    df_euc = df[df['species'].str.contains('Eucalyptus')]
    df_other = df[~ df['species'].str.contains('Eucalyptus')]

    euc_clust = cluster_heights(df_euc)
    other_clust = cluster_heights(df_other)

    fig = plt.figure(figsize=(9,6))
    fig.subplots_adjust(hspace=0.1)
    fig.subplots_adjust(wspace=0.05)
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['font.size'] = 14
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14

    ax = fig.add_subplot(111)

    #ax.hist(df_euc.height.values, bins=20, density=True,
    #        ls='-', lw=1, facecolor="royalblue", edgecolor="lightgrey")

    sns.distplot(df_euc.height.values, ax=ax, rug=False, norm_hist=True,
                 kde_kws={"label": "KDE"})

    for i in euc_clust:
        ax.axvline(x=i, ls="--", color="grey")

    ax.set_xlabel('Height (m)')
    ax.set_ylabel('Probability density')
    plt.show()

    odir = "plots"
    fig.savefig(os.path.join(odir, "Euc_height_histogram.pdf"),
                bbox_inches='tight', pad_inches=0.1)
