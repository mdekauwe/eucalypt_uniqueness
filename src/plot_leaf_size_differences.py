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

    df_eucs = df[df['Genus species'].str.contains('Eucalyptus') &\
                 (df['Growth form'] == "Tree") &\
                 (df['woody_non-woody'] == "woody")]
    df_other = df[(~ df['Genus species'].str.contains('Eucalyptus')) &\
                  (df['Growth form'] == "Tree") &\
                  (df['woody_non-woody'] == "woody") &\
                  (df['DecidEver (woody only)'] == "E")]
    df_eucs = df_eucs.reset_index()
    df_other = df_other.reset_index()

    # cm2 to m2
    leaf_size_eucs = df_eucs['leaf_size'].values #* 0.0001
    leaf_size_other = df_other['leaf_size'].values #* 0.0001
    leaf_size_eucs = leaf_size_eucs[~np.isnan(leaf_size_eucs)]
    leaf_size_other = leaf_size_other[~np.isnan(leaf_size_other)]

    data = [leaf_size_eucs, leaf_size_other]

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

    ax.boxplot(data, whiskerprops=dict(color="black"), notch=True, whis=1.5,
               showfliers=False)

    y_max = 140
    y_min = 0.0

    ax.annotate("", xy=(1, y_max), xycoords='data',
                xytext=(2, y_max), textcoords='data',
                arrowprops=dict(arrowstyle="-", ec='#aaaaaa',
                connectionstyle="bar,fraction=0.2"))
    ax.text(1.5, y_max + abs(y_max - y_min)*0.2, stars(p_value),
            horizontalalignment='center',
            verticalalignment='center')

    # jitter & show all points
    for i in range(2):
        y = data[i]
        x = np.random.normal(i+1.1, 0.04, size=len(y))
        ax.plot(x, y, 'k.', alpha=0.1)

    ax.set_ylabel("Leaf area (cm$^{2}$)")
    plt.xticks([1, 2], ['Eucalyptus species\n(n=%d)' % len(data[0]),\
                        'EBF\n(n=%d)' % len(data[1])])

    ax.set_ylim(0, 150)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    #plt.show()

    odir = "plots"
    fig.savefig(os.path.join(odir, "leaf_size_boxplot_within_EBT.pdf"),
                bbox_inches='tight', pad_inches=0.1)


def compare_within_hydroclimate(df):

    df_eucs = df[df['Genus species'].str.contains('Eucalyptus') &\
                 (df['Growth form'] == "Tree") &\
                 (df['woody_non-woody'] == "woody")]
    df_other = df[(~ df['Genus species'].str.contains('Eucalyptus')) &\
                  (df['Growth form'] == "Tree") &\
                  (df['woody_non-woody'] == "woody") &\
                  (df['DecidEver (woody only)'] == "E")]


    df_eucs = df_eucs[~np.isnan(df_eucs['leaf_size'])]
    df_eucs = df_eucs[~np.isnan(df_eucs['MAP'])]
    df_other = df_other[~np.isnan(df_other['leaf_size'])]
    df_other = df_other[~np.isnan(df_other['MAP'])]

    euc_clus_1 = df_eucs[(df_eucs.clust == 1) & \
                        (df_eucs.clust == 1)].leaf_size.values
    euc_clus_2 = df_eucs[(df_eucs.clust == 2) & \
                        (df_eucs.clust == 2)].leaf_size.values
    euc_clus_3 = df_eucs[(df_eucs.clust == 3) & \
                        (df_eucs.clust == 3)].leaf_size.values
    euc_clus_4 = df_eucs[(df_eucs.clust == 4) & \
                        (df_eucs.clust == 4)].leaf_size.values

    other_clus_1 = df_other[(df_other.clust == 1) & \
                            (df_other.clust == 1)].leaf_size.values
    other_clus_2 = df_other[(df_other.clust == 2) & \
                            (df_other.clust == 2)].leaf_size.values
    other_clus_3 = df_other[(df_other.clust == 3) & \
                            (df_other.clust == 3)].leaf_size.values
    other_clus_4 = df_other[(df_other.clust == 4) & \
                            (df_other.clust == 4)].leaf_size.values

    data_a = [euc_clus_1, euc_clus_2, euc_clus_3, euc_clus_4]
    data_b = [other_clus_1, other_clus_2, other_clus_3, other_clus_4]

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

    # $ python src/add_map_clustering_to_file.py
    (clus_1_low, clus_2_low,
     clus_3_low, clus_4_low) = (260.0,527.0,961.0,1700.0)
    (clus_1_hig, clus_2_hig,
     clus_3_hig, clus_4_hig) = (479.0,915.0,1575.0,2351.0)

    ticks = ['%d-%d mm' % (round(clus_1_low,1),round(clus_1_hig,1)),\
             '%d-%d mm' % (round(clus_2_low,1),round(clus_2_hig,1)),\
             '%d-%d mm' % (round(clus_3_low,1),round(clus_3_hig,1)),\
             '%d-%d mm' % (round(clus_4_low,1),round(clus_4_hig,1))]


    xloc_a = np.array(range(len(data_a)))*2.0-0.4
    xloc_b = np.array(range(len(data_b)))*2.0+0.4


    bpl = ax.boxplot(data_a, positions=xloc_a, sym='', widths=0.6,
                     whiskerprops=dict(color="black"), notch=True, whis=1.5,
                     showfliers=False)
    bpr = ax.boxplot(data_b, positions=xloc_b, sym='', widths=0.6,
                     whiskerprops=dict(color="black"), notch=True, whis=1.5,
                     showfliers=False)
    set_box_color(bpl, '#D7191C') # colors are from http://colorbrewer2.org/
    set_box_color(bpr, '#2C7BB6')

    # jitter & show all points
    for i in range(4):
        y = data_a[i]
        x = np.random.normal(xloc_a[i]+0.1, 0.04, size=len(y))
        ax.plot(x, y, 'k.', alpha=0.1)

        y = data_b[i]
        x = np.random.normal(xloc_b[i]+0.1, 0.04, size=len(y))
        ax.plot(x, y, 'k.', alpha=0.1)

    # draw temporary red and blue lines and use them to create a legend
    ax.plot([], c='#D7191C', label='Eucalyptus species')
    ax.plot([], c='#2C7BB6', label='EBF')
    plt.xticks(range(0, len(ticks) * 2, 2), ticks)
    ax.set_xlim(-2, len(ticks)*2)
    ax.set_ylim(0, 9)
    ax.legend(numpoints=1, loc=(0.05,0.7), frameon=False)
    ax.set_ylabel("Leaf area (cm$^{2}$)")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # non-parametric alternative to the t-test, Mann-Whitney U-test
    z, p = scipy.stats.mannwhitneyu(data_a[0], data_b[0])
    p_value = p * 2 # two tailed

    y_max = 150
    y_min = 0.0

    ax.annotate("", xy=(xloc_a[0], y_max), xycoords='data',
                xytext=(xloc_b[0], y_max), textcoords='data',
                arrowprops=dict(arrowstyle="-", ec='#aaaaaa',
                connectionstyle="bar,fraction=0.2"))
    ax.text(xloc_a[0]+0.4, y_max+8, stars(p_value),
            horizontalalignment='center',
            verticalalignment='center')

    # non-parametric alternative to the t-test, Mann-Whitney U-test
    z, p = scipy.stats.mannwhitneyu(data_a[1], data_b[1])
    p_value = p * 2 # two tailed

    y_max = 150
    y_min = 0

    ax.annotate("", xy=(xloc_a[1], y_max), xycoords='data',
                xytext=(xloc_b[1], y_max), textcoords='data',
                arrowprops=dict(arrowstyle="-", ec='#aaaaaa',
                connectionstyle="bar,fraction=0.2"))
    ax.text(xloc_a[1]+0.4, y_max+8, stars(p_value),
            horizontalalignment='center',
            verticalalignment='center')

    # non-parametric alternative to the t-test, Mann-Whitney U-test
    z, p = scipy.stats.mannwhitneyu(data_a[2], data_b[2])
    p_value = p * 2 # two tailed

    y_max = 150
    y_min = 0

    ax.annotate("", xy=(xloc_a[2], y_max), xycoords='data',
                xytext=(xloc_b[2], y_max), textcoords='data',
                arrowprops=dict(arrowstyle="-", ec='#aaaaaa',
                connectionstyle="bar,fraction=0.2"))
    ax.text(xloc_a[2]+0.4, y_max+8, stars(p_value),
            horizontalalignment='center',
            verticalalignment='center')

    # non-parametric alternative to the t-test, Mann-Whitney U-test
    z, p = scipy.stats.mannwhitneyu(data_a[3], data_b[3])
    p_value = p * 2 # two tailed

    y_max = 150
    y_min = 0

    ax.annotate("", xy=(xloc_a[3], y_max), xycoords='data',
                xytext=(xloc_b[3], y_max), textcoords='data',
                arrowprops=dict(arrowstyle="-", ec='#aaaaaa',
                connectionstyle="bar,fraction=0.2"))
    ax.text(xloc_a[3]+0.4, y_max+8, stars(p_value),
            horizontalalignment='center',
            verticalalignment='center')
    ax.set_ylim(0, 1.5)

    #plt.show()
    ax.set_ylim(0, 150)

    odir = "plots"
    fig.savefig(os.path.join(odir, "leaf_size_boxplot_hydroclimate.pdf"),
                bbox_inches='tight', pad_inches=0.1)

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)


if __name__ == "__main__":

    fdir = "data/leaf_size"
    fn = "leaf_size_clustered.csv"
    df = pd.read_csv(os.path.join(fdir, fn))
    df.rename(columns={'Leaf size (cm2)':'leaf_size'}, inplace=True)

    compare_within_ebt(df)
    compare_within_hydroclimate(df)
