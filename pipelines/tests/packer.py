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

from sctool import util,scale,plot,query,qc
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

def count_filter(args):
    sc = util.load_sc(CFG,load_light=False)
    pipe.count_filter(sc,verbose=True)

def split_into_batch(args):
    sc = util.load_sc(CFG,load_light=True)
    pipe.split_into_batch(sc)

def default(args):
    sc = util.load_sc(CFG,load_light=False)
    pipe.default(sc)

def batch_hvg(args):
    sc = util.load_sc(CFG,load_light=False)
    pipe.split_into_batch(sc) 
    pipe.count_filter(sc,verbose=False)
    pipe.hvg_batch(sc)
    pipe.hvg_all(sc)
    print(qc.hvg_batch_vs_all(sc))
    qc.plot_hvg_batch_vs_all(sc)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('mode',
                        action = 'store',
                        choices = [t for (t,o) in getmembers(sys.modules[__name__]) if isfunction(o)],
                        help = 'Function call')
    
    args = parser.parse_args()
    eval(args.mode + '(args)')


