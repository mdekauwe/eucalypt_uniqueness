#!/usr/bin/env python

"""
Explore the austraits db a bit

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

def main(df):

    # drop empty rows...
    df = df[df.species_name != '']

    # weird issue of mixed casting string & float, fix that
    df['species_name'] = df['species_name'].astype(str)

    print(np.unique(df.species_name))


if __name__ == "__main__":

    fdir = "data/austraits/"
    fn = "austraits_eucalypt_reoranised.csv"
    df = pd.read_csv(os.path.join(fdir, fn))

    main(df)
