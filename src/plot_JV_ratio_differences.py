#!/usr/bin/env python

"""
Compare g1 derived from leaf gas exchange:

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
import scipy.stats

def stars(p):
   if p < 0.0001:
       return "****"
   elif (p < 0.001):
       return "***"
   elif (p < 0.01):
       return "**"
   elif (p < 0.05):
       return "*"
   else:
       return "ns"


def compare_within_ebt(df):

    df_eucs = df[df['Species'].str.contains('Eucalyptus')]
    df_other = df[~ df['Species'].str.contains('Eucalyptus')]
    df_other = df_other[(df_other['PFT'] == "EBF") | (df_other['PFT'] == "TRF")]

    JV_eucs = df_eucs['Jmax'].values / df_eucs['Vcmax'].values
    JV_other = df_other['Jmax'].values / df_other['Vcmax'].values
    data = [JV_eucs, JV_other]

    # non-parametric alternative to the t-test, Mann-Whitney U-test
    z, p = scipy.stats.mannwhitneyu(data[0], data[1])
    p_value = p * 2 # two tailed

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

    ax.boxplot(data, whiskerprops=dict(color="black"), notch=True, whis=1.5)

    y_max = 3
    y_min = 0

    ax.annotate("", xy=(1, y_max), xycoords='data',
                xytext=(2, y_max), textcoords='data',
                arrowprops=dict(arrowstyle="-", ec='#aaaaaa',
                connectionstyle="bar,fraction=0.2"))
    ax.text(1.5, y_max + abs(y_max - y_min)*0.15, stars(p_value),
            horizontalalignment='center',
            verticalalignment='center')

    # jitter & show all points
    for i in range(2):
        y = data[i]
        x = np.random.normal(i+1, 0.04, size=len(y))
        ax.plot(x, y, 'k.', alpha=0.1)

    ax.set_ylabel("$V_{\mathrm{cmax}}$ (\u03BCmol m$^{-2}$ s$^{-1}$)")
    plt.xticks([1, 2], ['Eucalyptus species\n(n=%d)' % len(data[0]),\
                        'EBF\n(n=%d)' % len(data[1])])

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.show()

    odir = "plots"
    fig.savefig(os.path.join(odir, "jv_ratio_boxplot_within_EBT.pdf"),
                bbox_inches='tight', pad_inches=0.1)

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)


if __name__ == "__main__":

    fdir = "data/vcmax"
    fn = "Vcmax_one_point.csv"
    df = pd.read_csv(os.path.join(fdir, fn))

    compare_within_ebt(df)
    #compare_within_hydroclimate(df)
