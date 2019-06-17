#!/usr/bin/env python

"""
Reorganise the Austraits database a bit...

Given a list of desired traits, this will reorganise the required traits as
cols, using NaN where we don't have them. I will need to refine this a little...

That's all folks.
"""
__author__ = "Martin De Kauwe"
__version__ = "1.0 (17.07.2019)"
__email__ = "mdekauwe@gmail.com"

import os
import sys
import glob
import pandas as pd
import numpy as np

def reorganise_database(df, df_out, traits):

    # drop empty rows...
    df = df[df.species_name != '']


    id = 0
    for index, row in df.iterrows():

        d = {}
        d["species_name"] = row.species_name
        d["id"] = id

        for trait in traits:
            d[trait] = get_trait(row, trait)

        df_out = df_out.append(d, ignore_index=True)

        id += 1


    return (df_out)


def get_trait(row, trait_name):
    if row.trait_name == trait_name:
        value = row.value
    else:
        value = "NaN"

    return value

if __name__ == "__main__":

    fdir = "data/austraits/"
    fn = "austraits_eucalypt.csv"
    df = pd.read_csv(os.path.join(fdir, fn))

    traits = ["leaf_length", "leaf_width", "plant_height", "leaf_area", \
              "leaf_dark_respiration_per_area", \
              "leaf_dark_respiration_per_dry_mass", "leaf_N_per_dry_mass", \
              "leaf_P_per_dry_mass", "photosynthetic_rate_per_area", \
              "photosynthetic_rate_per_dry_mass", \
              "sapwood_specific_conductivity", "specific_leaf_area",\
              "wood_density", "leaf_dry_mass", "leaf_delta13C"]
    out_cols = ['species_name','id'] + traits
    df_out = pd.DataFrame(columns=out_cols)

    df_out = reorganise_database(df, df_out, traits)

    ofname = os.path.join(fdir, "austraits_eucalypt_reoranised.csv")
    df_out.to_csv(ofname, index=False)
