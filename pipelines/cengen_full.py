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
DATASET = 'cengenfull'

def reduce_to_pioneers(sc):
    print(f"# cells before neuron removal: {len(sc.cells)}") 
    cells = read.into_list(sc.cfg['mat']['ref_nodes'])
    flag = scmod.flag_cells_isin(sc,sc.cell_key,cells,'neurons')
    scmod.select_cell_flag(sc,flag) 
    print(f"# cells after neuron removal: {len(sc.cells)}") 

def normalize_to_median(sc):
    med_libsize = pp.query.median_cell_count(sc)
    Xn = pp.scale.normalize_per_cell(sc,counts_per_cell_after=med_libsize,copy=True)
    sc.set_count_matrix(Xn)

def run_hvg_pioneers(sc):
    reduce_to_pioneers(sc)
    print(sc.X.shape)
    normalize_to_median(sc)
    hvg_score = np.zeros(sc.X.shape[1])

    pp.flag.hvg(sc,method='mean_variance',num_hvg=3000,label='hvg',keep_model=True,hvg_score=hvg_score) 
    pp.plot.hvg_mean_var(sc,label='hvg')
    
    sc.genes['hvg_score'] = hvg_score
    sc.genes.to_csv(cfg['files']['hvg_pioneer_score'],index=False)

    plt.show() 



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

