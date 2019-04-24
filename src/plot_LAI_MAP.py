#!/usr/bin/env python

"""
Plot LAI vs MAP using LAI compilation from Iio et al.

Iio, A., and A. Ito. 2014. A Global Database of Field-observed Leaf Area Index
in Woody Plant Species, 1932-2011. ORNL DAAC, Oak Ridge, Tennessee, USA.
https://doi.org/10.3334/ORNLDAAC/1231

That's all folks.
"""
__author__ = "Martin De Kauwe"
__version__ = "1.0 (24.04.2019)"
__email__ = "mdekauwe@gmail.com"

import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
from matplotlib import cm
from pygam import LinearGAM


def plot_LAI_MAP(df):

    # Clean the df
    df = df.drop(columns=['Potentially_erroneous_data', 'Reference_number',\
                          'Method'])
    df.rename(columns={'MAT_(Literature value)':'MAT',
                       'MAP_(Literature value)':'MAP'}, inplace=True)

    df_eucs = df[(df['PFT'] == "EB") & \
                 (df['MAP'] < 3000.0) & \
                 (df['Vegetation_status'] == "Natural") &\
                 (df['Dominant_species'].str.contains('Eucalyptus'))]

    df_ebf = df[(df['PFT'] == "EB") &\
                (df['MAP'] < 3000.0) & \
                (df['Vegetation_status'] == "Natural") &\
                (~ df['Dominant_species'].str.contains('Eucalyptus'))]

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

    colours = sns.color_palette("Set2", 8)




    ax.plot(df_ebf.MAP, df_ebf.Total_LAI, color=colours[0], ls=" ", marker="o")
    nsplines = 7
    x = df_ebf.MAP.values
    y = df_ebf.Total_LAI.values
    gam = LinearGAM(n_splines=nsplines).fit(x, y)
    x_pred = np.linspace(min(df_ebf.MAP), max(x), num=100)
    y_pred = gam.predict(x_pred)
    y_int = gam.confidence_intervals(x_pred, width=.95)
    ax.plot(x_pred, y_pred, color=colours[0], ls='-', lw=2.0, zorder=10)
    ax.fill_between(x_pred, y_int[:, 0], y_int[:, 1], alpha=0.2,
                    facecolor=colours[0], zorder=10)

    ax.plot(df_eucs.MAP, df_eucs.Total_LAI, color=colours[1], ls=" ",
             marker="o")
    nsplines = 4
    x = df_eucs.MAP.values
    y = df_eucs.Total_LAI.values
    gam = LinearGAM(n_splines=nsplines).fit(x, y)
    x_pred = np.linspace(min(df_ebf.MAP), max(x), num=100)
    y_pred = gam.predict(x_pred)
    y_int = gam.confidence_intervals(x_pred, width=.95)
    ax.plot(x_pred, y_pred, color=colours[1], ls='-', lw=2.0, zorder=10)
    ax.fill_between(x_pred, y_int[:, 0], y_int[:, 1], alpha=0.2,
                    facecolor=colours[1], zorder=10)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_xlabel("Mean annual precipitation (mm)")
    ax.set_ylabel("Leaf area index (m$^{2}$ m$^{-2}$)")

    plt.show()

    odir = "plots"
    fig.savefig(os.path.join(odir, "LAI_vs_MAP_Eucs_vs_EBF.pdf"),
                bbox_inches='tight', pad_inches=0.1)

    plt.show()


def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)


if __name__ == "__main__":

    fdir = "data/lai/"
    fn = "geb12133-sup-0001-as1.csv"
    df = pd.read_csv(os.path.join(fdir, fn), encoding='latin-1',
                     skiprows=[0,1,2,3,4,5])

    plot_LAI_MAP(df)
