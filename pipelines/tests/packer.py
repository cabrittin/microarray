"""
@name:                         
@description:                  


@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05              
"""

import sys
from configparser import ConfigParser,ExtendedInterpolation
import argparse
from inspect import getmembers,isfunction
import matplotlib.pyplot as plt

from sctool import util,query,qc,explore
import pipelines.packer as pipe

CONFIG = 'config.ini'
DATASET = 'packer2019'
CFG = util.checkout(CONFIG,DATASET)


def cell_columns(args):
    sc = util.load_sc(CFG,load_light=True)
    print(sc.cell_columns())

def gene_columns(args):
    sc = util.load_sc(CFG,load_light=True)
    print(sc.gene_columns())

def reduce_to_neurons(args):
    sc = util.load_sc(CFG,load_light=True)
    pipe.reduce_to_neurons(sc,verbose=True)

def remove_cell_cycle(args):
    sc = util.load_sc(CFG,load_light=True)
    pipe.remove_cell_cycle(sc,verbose=True)

def count_filter(args):
    sc = util.load_sc(CFG,load_light=False)
    pipe.count_filter(sc,verbose=True)

def split_into_batch(args):
    sc = util.load_sc(CFG,load_light=True)
    pipe.split_into_batch(sc)

def default(args):
    sc = util.load_sc(CFG,load_light=False)
    pipe.default(sc)

def hvg_batch(args):
    sc = util.load_sc(CFG,load_light=False)
    pipe.split_into_batch(sc) 
    pipe.count_filter(sc,verbose=False)
    pipe.normalize_to_median(sc)
    pipe.hvg_batch(sc)
    pipe.hvg_all(sc)
    explore.hvg_mean_var(sc,label='merge_hvg') 
    plt.show()
    print(qc.hvg_batch_vs_all(sc))
    qc.plot_hvg_batch_vs_all(sc)

def hvg_all(args):
    sc = util.load_sc(CFG,load_light=False)
    pipe.count_filter(sc,verbose=False)
    pipe.normalize_to_median(sc)
    pipe.hvg_all(sc)
    explore.hvg_mean_var(sc) 
    plt.show()
 
def normalize_to_median(sc):
    sc = util.load_sc(CFG,load_light=False)
    pipe.normalize_to_median(sc)
    print(sc.X.sum(1))

def pca_embedding(args):
    sc = util.load_sc(CFG,load_light=False)
    pipe.pca_embedding(sc)
    explore.scree_plot(sc)
    plt.show()

def pca_embedding_neurons(args):
    sc = util.load_sc(CFG,load_light=False)
    pipe.pca_embedding_neurons(sc)
    explore.scree_plot(sc)
    plt.show()


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('mode',
                        action = 'store',
                        choices = [t for (t,o) in getmembers(sys.modules[__name__]) if isfunction(o)],
                        help = 'Function call')
    
    args = parser.parse_args()
    eval(args.mode + '(args)')


