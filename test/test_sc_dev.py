"""
@name: test_sc_dev.py                 
@description:                  
    Test sc class functions for development

@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05              
"""

import sys
from configparser import ConfigParser,ExtendedInterpolation
import argparse
from inspect import getmembers,isfunction
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sctool import util,scale,plot,query,qc
from sctool.sc import SingleCell as SC
from sctool.feature_selection import fit_loess

CONFIG = 'config.ini'
TORDER = [
        '< 100', 
        '100-130',
        '130-170', 
        '170-210', 
        '210-270', 
        '270-330', 
        '330-390', 
        '390-450', 
        '450-510', 
        '510-580', 
        '580-650', 
        '> 650' 
        ]

def checkout(args):
    cfg = util.checkout(args.config,args.dataset)
    print(cfg)
 
def load(args):
    cfg = util.checkout(args.config,args.dataset)
    sc = util.load_sc(cfg)
    print(sc.X.shape)

def cell_columns(args):
    cfg = util.checkout(args.config,args.dataset)
    sc = SC(cfg,load_light=True)
    print(sc.cell_columns())

def gene_columns(args):
    cfg = util.checkout(args.config,args.dataset)
    sc = util.load_sc(cfg,load_light=True)
    print(sc.gene_columns())

def get_unique_cell_meta(args):
    cfg = util.checkout(args.config,args.dataset)
    sc = SC(cfg,load_light=True)
    print(sc.get_unique_cell_meta('embryo.time.bin'))

def get_unique_cells(args):
    cfg = util.checkout(args.config,args.dataset)
    sc = SC(cfg,load_light=True)
    print(sc.get_unique_cells())

def load_gene_list(args):
    cfg = util.checkout(args.config,args.dataset)
    sc = util.load_sc(cfg,load_light=True)
    sc.load_gene_list('mitochondria')
    print(sc.gene_list)

def qc_ecdf_cell_total_counts(args):
    cfg = util.checkout(args.config,args.dataset)
    sc = util.load_sc(cfg,load_light=False)
    qc.ecdf_cell_total_counts(sc)

def qc_cell_total_counts(args):
    thresh = 6.5
    cfg = util.checkout(args.config,args.dataset)
    sc = util.load_sc(cfg,load_light=False)
    query.qc_cell_total_count(sc,thresh=thresh)
    fig,ax = plt.subplots(1,1,figsize=(5,5))
    qc.ecdf_cell_total_counts(sc,ax=ax,show=False)
    ax.axvline(x=thresh,color='r',linestyle='--') 
    plt.show()

def qc_mitochondria_total_counts(args):
    thresh = 2
    cfg = util.checkout(args.config,args.dataset)
    sc = util.load_sc(cfg,load_light=False)
    sc.load_gene_list('mitochondria')
    query.qc_cell_total_count(sc,genes=sc.gene_list,thresh=thresh,label='qc_mt_count')
    fig,ax = plt.subplots(1,1,figsize=(5,5))
    qc.ecdf_cell_total_counts(sc,genes=sc.gene_list,ax=ax,show=False)
    ax.axvline(x=thresh,color='r',linestyle='--') 
    plt.show()

def plot_mitochondria_qc(args):
    cfg = util.checkout(args.config,args.dataset)
    sc = util.load_sc(cfg,load_light=False)
    sc.load_gene_list('mitochondria')
    #sc.X.data = np.log(sc.X.data) 
    query.qc_cell_total_count(sc,thresh=6.5)
    query.qc_cell_total_count(sc,genes=sc.gene_list,thresh=2,label='qc_mt_count')
    print(sc.X.shape)
    sc.filter_cells_eq('qc_total_count',1)
    print(sc.X.shape)
    sc.filter_cells_eq('qc_mt_count',1)
    print(sc.X.shape)
    x = query.cell_total_counts(sc)
    y = query.cell_total_counts(sc,genes=sc.gene_list)
    #x = x - y 
    print(x.shape,y.shape)
    x = np.log(x+1)
    y = np.log(y+1)
    print(x.min())
    idx, model = fit_loess(sc,x,y,return_model=True,axis=1) 
    num_hvg = 1000 
    hvg_0 = idx[:num_hvg]
    hvg_1 = idx[num_hvg:]
    #print(x.min())
    _x = np.linspace(x.min(),x.max(),100)
    _y = model.predict(_x).values
    fig,ax = plt.subplots(1,1,figsize=(10,10)) 
    ax.plot(_x,_y,c='k')
    ax.scatter(x,y,s=5,c='#9f9f9f',alpha=0.5)
    #ax.scatter(x[hvg_0],y[hvg_0],s=5,c='r',label='Low MT')
    plt.show()

def query_gene_counts(args):
    cfg = util.checkout(args.config,args.dataset)
    sc = SC(cfg,load_light=False)
    scale.seurat_log_scale(sc)
    print(sc.X.shape) 
    X = query.gene_counts(sc,['nhr-25','nhr-23'])
    print(X.shape)

def query_label_gene_counts(args):
    cfg = util.checkout(args.config,args.dataset)
    sc = SC(cfg,load_light=False)
    scale.seurat_log_scale(sc)
    print(sc.X.shape) 
    df = query.label_gene_counts(sc,['nhr-25','nhr-23'],'embryo.time.bin',std_scale=True)
    print(df)

def plot_label_gene_counts(args):
    cfg = util.checkout(args.config,args.dataset)
    sc = SC(cfg,load_light=False)
    scale.seurat_log_scale(sc)
    #gtest = 'par-3'
    gtest = 'egl-5'
    df = query.label_gene_counts(sc,['nhr-25','nhr-23',gtest],'embryo.time.bin',std_scale=True)
    print(df)
    fig,ax = plt.subplots(3,1,figsize=(20,10))
    thresh = 2

    gene = gtest
    df1 = df[df[gene] > thresh]
    sns.violinplot(ax=ax[0],data=df1,x='embryo.time.bin',y=gene,order=TORDER)
    ax[0].set_ylim([0,10])
    
    gene = 'nhr-25' 
    df1 = df[df[gene] > thresh]
    sns.violinplot(ax=ax[1],data=df1,x='embryo.time.bin',y=gene,order=TORDER)
    ax[1].set_ylim([0,10])
    
    gene = 'nhr-23' 
    df1 = df[df[gene] > thresh]
    sns.violinplot(ax=ax[2],data=df1,x='embryo.time.bin',y=gene,order=TORDER)
    ax[2].set_ylim([0,10])
    
    plt.show()

def plot_label_gene_counts_2(args):
    cfg = util.checkout(args.config,args.dataset)
    sc = SC(cfg,load_light=False)
    scale.seurat_log_scale(sc)
    #gtest = 'par-3'
    gtest = 'egl-5'
    df = query.label_gene_counts(sc,['nhr-25','nhr-23',gtest],'embryo.time.bin',std_scale=True)
    
    def foo(_df,_gene,_label,thresh=2,**kwargs):
        return _df[_df[_gene] > thresh]

    plot.gene_by_label(df,'nhr-25','embryo.time.bin',order=TORDER,callback=foo)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('mode',
                        action = 'store',
                        choices = [t for (t,o) in getmembers(sys.modules[__name__]) if isfunction(o)],
                        help = 'Function call')
    
    parser.add_argument('dataset',
                action = 'store',
                help = 'Dataset alias in config file')

    parser.add_argument('-c','--config',
                dest = 'config',
                action = 'store',
                default = CONFIG,
                required = False,
                help = 'Config file')

    args = parser.parse_args()
    eval(args.mode + '(args)')

