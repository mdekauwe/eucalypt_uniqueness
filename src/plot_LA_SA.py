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
from pygam import GammaGAM


def plot_LA_SA_vs_height_species(df):

    df_eucs = df[(df['pft'] == "EA") & \
                 (df['species'].str.contains('Eucalyptus')) |\
                 (df['species'].str.contains('Corymbia'))]

    # drop SaldanaAcosta2009 data.
    df_ebf = df[(df['pft'] == "EA") & \
                (~df['species'].str.contains('Eucalyptus')) &\
                (~df['species'].str.contains('Corymbia')) &\
                (df['studyName'] != "SaldanaAcosta2009")]

    fig = plt.figure(figsize=(9,6))
    fig.subplots_adjust(hspace=0.1)
    fig.subplots_adjust(wspace=0.05)
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['font.size'] = 14
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14

    ax = fig.add_subplot(111)

    # ax is the log, log scale which produces nice labels and ticks
    ax.set(yscale="log", xscale="log")

    # ax2 is the axes where the values are plottet to
    #ax2 = ax.twinx()

    #colours = sns.color_palette("Set2", 8)

    x = df_ebf["h.t"]
    y = df_ebf["a.lf"] / df_ebf["a.ssba"]
    yy = df_ebf["a.lf"] / df_ebf["a.ssbh"]
    yyy = df_ebf["a.lf"] / df_ebf["a.ssbc"]

    ax.plot(x, y, color="black", ls=" ", marker=".",
            alpha=0.1)
    ax.plot(x, yy, color="black", ls=" ", marker="*",
            alpha=0.1)
    ax.plot(x, yyy, color="black", ls=" ", marker="^",
            alpha=0.1)

    df_eucs = df_eucs[['species','h.t', 'a.lf', 'a.ssba', 'a.ssbh', 'a.ssbc']]
    #df_eucs = df_eucs.dropna()
    n_species = np.unique(df_eucs.species)
    n_colours = len(n_species)
    colours = sns.color_palette('hls', n_colors=n_colours)
    import itertools
    marker = itertools.cycle(('^', 'v', '.', '*'))
    for i, spp in enumerate(n_species):

        df_spp = df_eucs[df_eucs.species == spp]
        name = str(np.unique(df_spp.species)[0]).replace("Eucalyptus", "E.")

        x = df_spp["h.t"]
        y = df_spp["a.lf"] / df_spp["a.ssba"]
        yy = df_spp["a.lf"] / df_spp["a.ssbh"]
        yyy = df_spp["a.lf"] / df_spp["a.ssbc"]

        ax.plot(x, y, ls=" ", marker = ".", alpha=1, label=name,
                color=colours[i])
        ax.plot(x, yy, ls=" ", marker = "^", alpha=1, label=name,
                color=colours[i])
        ax.plot(x, yyy, ls=" ", marker = "*", alpha=1, label=name,
                color=colours[i])

    ax.legend(numpoints=1, loc="best", frameon=False, ncol=2)



    ax.set_xlabel("Plant height (m)")
    ax.set_ylabel("LA:SA")

    # set the limits of the log log axis to 10 to the power of the label of ax2
    #ax.set_ylim(10e-6, 10e2)
    #ax.set_xlim(10e-2, 10e1)


    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


    plt.show()

    odir = "plots"
    ofname = os.path.join(odir, "LASA_vs_height_Eucs_vs_EBF_species.pdf")
    fig.savefig(ofname, bbox_inches='tight', pad_inches=0.1)


if __name__ == "__main__":

    fdir = "data/baad"
    fn = "baad_data.csv"
    df = pd.read_csv(os.path.join(fdir, fn))

    plot_LA_SA_vs_height_species(df)
