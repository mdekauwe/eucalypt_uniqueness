#!/usr/bin/env python

"""
Compare g1 derived from isotope:

- within Australia
- between species in evergreen broadleaf class

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

def compare_within_Australia(df):

    # Cut off g1 at 14 for visual purposes
    df = df[df.g1 <= 14]

    # extracted from
    # http//www.naturalearthdata.com/download/110m/cultural/ne_110m_admin_0_countries.zip
    # https://gist.github.com/graydon/11198540
    df_aus = df[(df.latitude >= -43.6345972634) &
                (df.latitude <= -10.6681857235) &
                (df.longitude >= 113.338953078) &
                (df.longitude <= 153.569469029)]

    #print(len(df_aus), len(df))

    df_euc = df_aus[df_aus['orig_spp'].str.contains('Eucalyptus')]
    df_other = df_aus[~ df_aus['orig_spp'].str.contains('Eucalyptus')]

    #print(len(df_euc), len(df_other), len(df_euc) + len(df_other))

    data = [df_euc.g1.values, df_other.g1.values]

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

    ax.boxplot(data, whiskerprops=dict(color="black"), notch=True)

    # jitter & show all points
    for i in range(2):
        y = data[i]
        x = np.random.normal(i+1, 0.04, size=len(y))
        ax.plot(x, y, 'k.', alpha=0.1)

    ax.set_ylabel("$g_1$ (kPa$^{0.5}$)")
    plt.xticks([1, 2], ['Eucalyptus species', 'Other'])
    ax.set_ylim(-1.5, 14)

    plt.show()

    odir = "plots"
    fig.savefig(os.path.join(odir, "g1_isotope_boxplot_within_australia.pdf"),
                bbox_inches='tight', pad_inches=0.1)

if __name__ == "__main__":

    fdir = "data/g1"
    df = pd.read_csv(os.path.join(fdir, "g1_isotope_screened.csv"))

    compare_within_Australia(df)
    #compare_within_ebt(df)
