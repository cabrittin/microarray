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
from scipy import stats
from scipy import sparse

from sctool import util,scale,plot,query,qc
from sctool.feature_selection import hvg_seurat3
from sctool.sc import SingleCell as SC
import toolbox.matrix_properties as mp

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

def qc_ecdf_gene_elements(args):
    cfg = util.checkout(args.config,args.dataset)
    sc = util.load_sc(cfg,load_light=False)
    gnz = np.log2(mp.axis_elements(sc.X,axis=0))
    qc.plot_ecdf(gnz,xlabel='# cells with gene')
    plt.show()

def qc_cell_total_counts(args):
    thresh = 6.5
    cfg = util.checkout(args.config,args.dataset)
    sc = util.load_sc(cfg,load_light=False)
    query.qc_cell_total_count(sc,thresh=thresh)
    fig,ax = plt.subplots(1,1,figsize=(5,5))
    qc.ecdf_cell_total_counts(sc,ax=ax,show=False)
    ax.axvline(x=thresh,color='r',linestyle='--') 
    plt.savefig('data/packer2019/qc_total_counts.svg')
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
    plt.savefig('data/packer2019/qc_mt_counts.svg')
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
    [slope, intercept, r, p, std_err] = query.qc_residual_filter(sc,x,y,thresh=-2,label='qc_mt')
    _x = np.linspace(x.min(),x.max(),100)
    _y = slope * _x + intercept
    sc.cells['total_count'] = x
    sc.cells['mt_count'] = y

    fig,ax = plt.subplots(1,1,figsize=(5,5)) 
    cdict = {0:'r',1:'#9f9f9f'} 
    sns.scatterplot(sc.cells,ax=ax,x='total_count',y='mt_count',hue='qc_mt',palette=cdict,s=10)
    ax.plot(_x,_y,c='k')
    plt.savefig('data/packer2019/qc_mt.png')
    plt.show() 

def plot_poisson(args):
    cfg = util.checkout(args.config,args.dataset)
    sc = util.load_sc(cfg,load_light=False)
    sc.load_gene_list('mitochondria')
    query.qc_cell_total_count(sc,thresh=6.5)
    query.qc_cell_total_count(sc,genes=sc.gene_list,thresh=2,label='qc_mt_count')
    print(sc.X.shape)
    sc.filter_cells_eq('qc_total_count',1)
    print(sc.X.shape)
    sc.filter_cells_eq('qc_mt_count',1)
    print(sc.X.shape)
    x = query.cell_total_counts(sc)
    y = query.cell_total_counts(sc,genes=sc.gene_list)
    x = np.log(x+1)
    y = np.log(y+1)
    query.qc_residual_filter(sc,x,y,thresh=-2,label='qc_mt_resid')
    sc.cells['total_count'] = x
    sc.cells['mt_count'] = y
    sc.filter_cells_eq('qc_mt_resid',1)
    print(sc.X.shape)
    query.gene_mean_filter(sc,0.1,label='qc_gene_mean')
    query.gene_zero_count_filter(sc,0.1*sc.X.shape[0],label='qc_gene_zero_count')
    sc.filter_genes_bool('qc_gene_mean',1)
    sc.filter_genes_bool('qc_gene_zero_count',1)
    print(sc.X.shape)

    mu = mp.axis_mean(sc.X,axis=0,skip_zeros=False)
    num_z = sc.X.shape[0] - mp.axis_elements(sc.X,axis=0)
    
    mean,var = mp.axis_mean_var(sc.X,axis=0)
    cv2 = np.divide(var,np.power(mean,2))
    x,y = np.log2(mean),np.log2(cv2)
    thresh = 1 
    slope, intercept, r, p, std_err = stats.linregress(x, y)
    resid = y - (slope* x + intercept)
    rstd = np.std(resid)
    rnorm = np.divide(resid - np.mean(resid), np.std(resid))
    resid[rnorm < thresh] = 0
    resid[rnorm >= thresh] = 1
    yes = np.where(resid == 1)[0]
    no = np.where(resid==0)[0]
    _x = np.linspace(x.min(),x.max(),100)
    _y = slope * _x + intercept + thresh*rstd

    print(mu.shape,num_z.shape)
    print(len(yes))
    fig,ax = plt.subplots(1,1,figsize=(5,5))
    ax.scatter(np.log2(mu),num_z,s=5)
    ax.set_xlabel('log2(mean)',fontsize=10)
    ax.set_ylabel('# 0 counts',fontsize=10)
    plt.savefig('data/packer2019/poisson_1.svg')

    fig,ax = plt.subplots(1,1,figsize=(5,5))
    ax.scatter(x[yes],y[yes],s=5,color='k')
    ax.scatter(x[no],y[no],s=5,color='#9f9f9f')
    ax.plot(_x,_y,'r') 
    ax.set_xlabel('log2(mean)',fontsize=10)
    ax.set_ylabel('log2(CV2)',fontsize=10)
    plt.savefig('data/packer2019/poisson_2.svg')


    plt.show()

def quick_test(args):
    z = np.array([[1,1,0],[0,1000,0],[0,0,0]])
    A = sparse.csr_matrix(z)
    nz = mp.axis_elements(A,axis=0)
    zz = A.shape[0] - nz

    print(nz)
    print(zz)

def gene_mean_var(args):
    cfg = util.checkout(args.config,args.dataset)
    sc = util.load_sc(cfg,load_light=False)
    sc.load_gene_list('mitochondria')
    query.qc_cell_total_count(sc,thresh=6.5)
    query.qc_cell_total_count(sc,genes=sc.gene_list,thresh=2,label='qc_mt_count')
    print(sc.X.shape)
    sc.filter_cells_eq('qc_total_count',1)
    print(sc.X.shape)
    sc.filter_cells_eq('qc_mt_count',1)
   

    num_hvg = 2000 
    eps = 1e-5
    idx, model = hvg_seurat3(sc,return_model=True) 
    hvg_0 = idx[:num_hvg]
    hvg_1 = idx[num_hvg:]
    
    mean,var = mp.axis_mean_var(sc.X,axis=0,skip_zeros=False) 
    x = np.log10(mean[var>0])
    y = np.log10(var[var>0])
    _x = np.linspace(x.min(),x.max(),100)
    _y = model.predict(_x).values
    fig,ax = plt.subplots(1,1,figsize=(5,5))
    x,y = np.log10(mean[hvg_0]),np.log10(var[hvg_0])
    ax.scatter(x,y,s=5,c='r',label='High variable genes')
    x,y = np.log10(mean[hvg_1]+eps),np.log10(var[hvg_1]+eps)
    ax.scatter(x,y,s=5,c='#9f9f9f',alpha=0.5)
    ax.plot(_x,_y,c='k')
    ax.legend(loc='upper left',fontsize=8)
    xlabel= 'gene mean'
    ylabel = 'gene var'
    ax.set_xlabel(xlabel,fontsize=6)
    ax.set_ylabel(ylabel,fontsize=6)
    ax.tick_params(axis='x',labelsize=4)
    ax.tick_params(axis='y',labelsize=4)
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

