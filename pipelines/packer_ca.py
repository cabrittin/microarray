"""
@name:                         
@description:                  


@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05              
"""

import sys
import os
import argparse
from inspect import getmembers,isfunction
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from tqdm import tqdm
import mplcursors
import time

from pycsvparser import read
from sctool import util,scmod
import sctool.pp as pp
import toolbox.matrix_properties as mp


CONFIG = 'config.ini'
DATASET = 'packer2019'

def pca_embedding(sc):
    ## Before removing cells first compute per batch background
    qc_flag = "passed_initial_QC_or_later_whitelisted"
    scmod.select_cell_flag(sc,qc_flag,True)
    sc.cells['size_factor'] = pp.query.size_factor(sc)
    pp.scale.cells_by_vector(sc,label='size_factor') 
    pp.log1p(sc) 
    #split_into_batch(sc) 


    #pp.flag.hvg_batch(sc,method='mean_variance',num_hvg=1000,keep_model=keep_model)


def old_pca_embedding(sc):
    count_filter(sc,verbose=True)
    pct_mitochondria_filter(sc,verbose=True)
    num_genes_in_cell_filter(sc,verbose=True)
    split_into_batch(sc) 
    normalize_to_median(sc)
    hvg_batch(sc,keep_model=False)
    remove_cell_cycle(sc,verbose=True)
    sc.X = pp.scale.standardize(sc)
    sc.X = pp.scale.clip(sc,10) 
    pp.embedding.pca(sc,gene_flag='merge_hvg',set_loadings=True)
    util.to_pickle(sc,sc.cfg['pickle']['embedding'])
    print(f"Pickle dump to {sc.cfg['pickle']['embedding']}")



if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('mode',
                        action = 'store',
                        choices = [t for (t,o) in getmembers(sys.modules[__name__]) if isfunction(o)],
                        help = 'Function call')
    
    parser.add_argument('--from_pickle',
                dest = 'from_pickle',
                action = 'store',
                default = None,
                required = False,
                help = 'Path to pickle file. If provided, SC will be loaded from pickle')
    
    parser.add_argument('--load_light',
                dest = 'load_light',
                action = 'store_true',
                default = False,
                required = False,
                help = 'If True, only load the meta data')
    
    parser.add_argument('--no_sc_load',
                dest = 'no_sc_load',
                action = 'store_true',
                default = False,
                required = False,
                help = 'If true, SC object not loaded. For functions that do not require SC loading')
    
    parser.add_argument('-c','--config',
                dest = 'config',
                action = 'store',
                default = CONFIG,
                required = False,
                help = 'Config file')
    
    args = parser.parse_args()
    
    if args.no_sc_load:
        sc = None
    else:
        cfg = util.checkout(args.config,DATASET)
        if args.from_pickle is not None:
            sc = util.from_pickle(args.from_pickle)
            sc.loaded_from = args.from_pickle
        else:
            sc = util.load_sc(cfg,load_light=args.load_light)
        
    eval(args.mode + '(sc)')


