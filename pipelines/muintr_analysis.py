"""
@name: muintr_analysis.py                         
@description:                  
    Functions for Mu_int_r analysis

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

from pycsvparser import read,write
from sctool import util,scmod
import sctool.pp as pp
import toolbox.matrix_properties as mp


CONFIG = 'config.ini'
DATASET = 'packer2019'
QC_FLAG = "passed_initial_QC_or_later_whitelisted"


def total_gene_counts(sc):
    scmod.select_cell_flag(sc,QC_FLAG,True)
    sc.genes['total'] = mp.axis_sum(sc.X,axis=0)
    fig,ax = plt.subplots(1,1,figsize=(10,10))
    pp.plot.plot_ecdf(sc.genes['total'].to_numpy(),ax=ax,xlabel='total_gene_umi',
                    plot_params=sc.cfg['plot_params'])
    #ax.set_xlim([0,100])
    ax.set_xlim([0,10000])
    plt.show()

def display_cells(sc):
    print(sc.cells.columns)  
    print(sc.cells['cell.type'].unique())  
    print(sc.cells['cell.subtype'].unique())  
    print(sc.cells['lineage'].unique())  

def print_lineage_ids(sc):
    L = sc.cells['lineage'].unique()
    
    lout = sorted([[l,0,0] for l in L if type(l) == str])
    fout = 'data/packer2019/mat/lineage_ids.csv'

    write.from_list(fout,lout)
    



if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('mode',
                        action = 'store',
                        choices = [t for (t,o) in getmembers(sys.modules[__name__]) if isfunction(o)],
                        help = 'Function call')
    
    parser.add_argument('--load_light',
                dest = 'load_light',
                action = 'store_true',
                default = False,
                required = False,
                help = 'If True, only load the meta data')
    
    parser.add_argument('-c','--config',
                dest = 'config',
                action = 'store',
                default = CONFIG,
                required = False,
                help = 'Config file')
    
    args = parser.parse_args()
    
    cfg = util.checkout(args.config,DATASET)
    sc = util.load_sc(cfg,load_light=args.load_light)
    eval(args.mode + '(sc)')

