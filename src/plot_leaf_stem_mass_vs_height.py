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



def plot_leaf_stem_vs_height(df):

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
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14

    ax = fig.add_subplot(111)

    # ax is the log, log scale which produces nice labels and ticks
    ax.set(yscale="log", xscale="log")

    # ax2 is the axes where the values are plottet to
    #ax2 = ax.twinx()

    colours = sns.color_palette("Set2", 8)

    x = df_ebf["h.t"].values
    y = df_ebf["m.lf"].values / df_ebf["m.st"].values
    y = y[~np.isnan(x)]
    x = x[~np.isnan(x)]
    x = x[~np.isnan(y)]
    y = y[~np.isnan(y)]


    ax.plot(x, y, color=colours[0], ls=" ", marker=".",
            alpha=0.3)
    gam = GammaGAM(n_splines=10).fit(x, y)
    x_pred = np.linspace(min(x), max(x), num=100)
    y_pred = gam.predict(x_pred)
    y_int = gam.confidence_intervals(x_pred, width=.95)
    ax.plot(x_pred, y_pred, color=colours[0], ls='-', lw=3.0, zorder=10,
            label="Global EBF")
    ax.fill_between(x_pred, y_int[:, 0], y_int[:, 1], alpha=0.2,
                    facecolor=colours[0], zorder=10)

    x = df_eucs["h.t"].values
    y = df_eucs["m.lf"].values / df_eucs["m.st"].values
    y = y[~np.isnan(x)]
    x = x[~np.isnan(x)]
    x = x[~np.isnan(y)]
    y = y[~np.isnan(y)]

    ax.plot(x, y, color=colours[1],ls=" ", marker=".",
            alpha=0.5)

    gam = GammaGAM(n_splines=7).fit(x, y)
    x_pred = np.linspace(min(x), max(x), num=100)
    y_pred = gam.predict(x_pred)
    y_int = gam.confidence_intervals(x_pred, width=.95)
    ax.plot(x_pred, y_pred, color=colours[1], ls='-', lw=3.0, zorder=10,
            label="Eucalypts")
    ax.fill_between(x_pred, y_int[:, 0], y_int[:, 1], alpha=0.2,
                    facecolor=colours[1], zorder=10)


    ax.legend(numpoints=1, loc="best", frameon=False)



    ax.set_xlabel("Plant height (m)")
    ax.set_ylabel("Leaf / stem biomass (kg kg$^{-1}$)")

    # set the limits of the log log axis to 10 to the power of the label of ax2
    ax.set_ylim(10e-4, 10e1)
    #ax.set_xlim(10e-2, 10e1)


    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


    plt.show()

    odir = "plots"
    fig.savefig(os.path.join(odir, "leaf_stem_vs_height_Eucs_vs_EBF.pdf"),
                bbox_inches='tight', pad_inches=0.1)

    plt.show()


def plot_leaf_stem_vs_height(df):

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
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14

    ax = fig.add_subplot(111)

    # ax is the log, log scale which produces nice labels and ticks
    ax.set(yscale="log", xscale="log")

    # ax2 is the axes where the values are plottet to
    #ax2 = ax.twinx()

    colours = sns.color_palette("Set2", 8)

    x = df_ebf["h.t"].values
    y = df_ebf["m.lf"].values / df_ebf["m.st"].values
    y = y[~np.isnan(x)]
    x = x[~np.isnan(x)]
    x = x[~np.isnan(y)]
    y = y[~np.isnan(y)]


    ax.plot(x, y, color=colours[0], ls=" ", marker=".",
            alpha=0.3)
    gam = GammaGAM(n_splines=10).fit(x, y)
    x_pred = np.linspace(min(x), max(x), num=100)
    y_pred = gam.predict(x_pred)
    y_int = gam.confidence_intervals(x_pred, width=.95)
    ax.plot(x_pred, y_pred, color=colours[0], ls='-', lw=3.0, zorder=10,
            label="Global EBF")
    ax.fill_between(x_pred, y_int[:, 0], y_int[:, 1], alpha=0.2,
                    facecolor=colours[0], zorder=10)

    x = df_eucs["h.t"].values
    y = df_eucs["m.lf"].values / df_eucs["m.st"].values
    y = y[~np.isnan(x)]
    x = x[~np.isnan(x)]
    x = x[~np.isnan(y)]
    y = y[~np.isnan(y)]

    ax.plot(x, y, color=colours[1],ls=" ", marker=".",
            alpha=0.5)

    gam = GammaGAM(n_splines=7).fit(x, y)
    x_pred = np.linspace(min(x), max(x), num=100)
    y_pred = gam.predict(x_pred)
    y_int = gam.confidence_intervals(x_pred, width=.95)
    ax.plot(x_pred, y_pred, color=colours[1], ls='-', lw=3.0, zorder=10,
            label="Eucalypts")
    ax.fill_between(x_pred, y_int[:, 0], y_int[:, 1], alpha=0.2,
                    facecolor=colours[1], zorder=10)


    ax.legend(numpoints=1, loc="best", frameon=False)



    ax.set_xlabel("Plant height (m)")
    ax.set_ylabel("Leaf / stem biomass (kg kg$^{-1}$)")

    # set the limits of the log log axis to 10 to the power of the label of ax2
    ax.set_ylim(10e-4, 10e1)
    #ax.set_xlim(10e-2, 10e1)


    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


    plt.show()

    odir = "plots"
    fig.savefig(os.path.join(odir, "leaf_stem_vs_height_Eucs_vs_EBF.pdf"),
                bbox_inches='tight', pad_inches=0.1)

    #plt.show()


def plot_leaf_stem_vs_height_contributors(df):

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
    y = df_ebf["m.lf"] / df_ebf["m.st"]
    ax.plot(x, y, color="black", ls=" ", marker=".",
            alpha=0.1)

    df_eucs = df_eucs[['species','h.t', 'm.lf', 'm.st']]
    df_eucs = df_eucs.dropna()
    n_species = np.unique(df_eucs.species)
    n_colours = len(n_species)
    colours = sns.color_palette('hls', n_colors=n_colours)
    import itertools
    marker = itertools.cycle(('^', 'v', '.', '*'))
    for i, spp in enumerate(n_species):

        df_spp = df_eucs[df_eucs.species == spp]
        name = str(np.unique(df_spp.species)[0]).replace("Eucalyptus", "E.")

        x = df_spp["h.t"]
        y = df_spp["m.lf"] / df_spp["m.st"]

        ax.plot(x, y, ls=" ", marker = next(marker), alpha=1, label=name,
                color=colours[i])

    ax.legend(numpoints=1, loc="best", frameon=False, ncol=3)



    ax.set_xlabel("Plant height (m)")
    ax.set_ylabel("Leaf / stem biomass (kg kg$^{-1}$)")

    # set the limits of the log log axis to 10 to the power of the label of ax2
    ax.set_ylim(10e-4, 10e1)
    #ax.set_xlim(10e-2, 10e1)


    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


    #plt.show()

    odir = "plots"
    fig.savefig(os.path.join(odir, "leaf_stem_vs_height_Eucs_vs_EBF_species.pdf"),
                bbox_inches='tight', pad_inches=0.1)



if __name__ == "__main__":

    fdir = "data/baad"
    fn = "baad_data.csv"
    df = pd.read_csv(os.path.join(fdir, fn))

    plot_leaf_stem_vs_height(df)
    plot_leaf_stem_vs_height_contributors(df)
