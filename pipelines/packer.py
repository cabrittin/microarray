"""
@name: filters.packer.py                        
@description:                  
    Some convenicence filters for packer dataset

@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05              
"""

import sys
import os
import argparse
from inspect import getmembers,isfunction
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import mplcursors

import phenograph

from pycsvparser import read
from sctool import util,query,qc,scale,explore,scmod,hvg,embedding,plot
from toolbox.stats.basic import ecdf

CONFIG = 'config.ini'
DATASET = 'packer2019'

def reduce_to_neurons(sc,verbose=False):
    if verbose: print(f"# cells before neuron removal: {len(sc.cells)}") 
    cells = read.into_list(sc.cfg['mat']['neurons'])
    flag = scmod.flag_cells_isin(sc,sc.cell_key,cells,'neurons')
    scmod.select_cell_flag(sc,flag) 
    #scmod.filter_cells_isin(sc,sc.cell_key,cells)
    if verbose: print(f"# cells after neuron removal: {len(sc.cells)}") 

def remove_cell_cycle(sc,verbose=False):
    if verbose: print(f"# genes before CC removeal: {len(sc.genes)}")
    genes = read.into_list(sc.cfg['gene_list']['cell_cycle'])
    flag = scmod.flag_genes_isin(sc,sc.gene_key,genes,'cell_cycle')
    scmod.select_gene_flag(sc,flag,val=0) 
    if verbose: print(f"# genes after CC removeal: {len(sc.genes)}")

def split_into_batch(sc):
    scmod.split_cells_by_key(sc,sc.cfg['keys']['meta_key_batch'],'batches')

def normalize_to_median(sc):
    med_libsize = query.median_cell_count(sc)
    Xn = scale.normalize_per_cell(sc,counts_per_cell_after=med_libsize,copy=True)
    sc.set_count_matrix(Xn)

def hvg_batch(sc):
    hvg.flag_hvg_batch(sc,method='mean_variance',num_hvg=1000)
    hvg.merge_hvg_batch(sc)

def hvg_all(sc):
    hvg.flag_hvg(sc,method='mean_variance',num_hvg=1000) 

def count_filter(sc,verbose=False):
    if verbose: print('Matrix shape before count filter: ',sc.X.shape) 
    thresh = sc.cfg.getint('filters','min_cells_with_gene') 
    query.minimum_cells_with_gene(sc,thresh,label='total_cells')
    thresh = sc.cfg.getint('filters','min_genes_in_cells') 
    query.minimum_genes_in_cell(sc,thresh,label='total_genes') 
    scmod.select_gene_flag(sc,'total_cells',1)
    scmod.select_cell_flag(sc, 'total_genes',1)
    if verbose: print('Matrix shape after count filter: ',sc.X.shape) 

def set_mitochondria(sc):
    sc.load_gene_list('mitochondria')
    sc.cells['total_count'] = query.cell_total_counts(sc)
    sc.cells['mt_count'] = query.cell_total_counts(sc,genes=sc.gene_list)
    query.qc_residual_filter(sc,sc.cells['total_count'].tonumpy(),
            sc.cells['mt_count'].tonumpy(),thresh=-2,label='qc_mt')

def pca_embedding(sc):
    split_into_batch(sc) 
    count_filter(sc,verbose=True)
    normalize_to_median(sc)
    hvg_batch(sc)
    remove_cell_cycle(sc,verbose=True)
    scale.log1p(sc)
    embedding.pca(sc,gene_flag='merge_hvg',set_loadings=True)
    util.to_pickle(sc,sc.cfg['pickle']['embedding'])
    print(f"Pickle dump to {sc.cfg['pickle']['embedding']}")

def pca_embedding_neurons(sc):
    reduce_to_neurons(sc,verbose=True) 
    split_into_batch(sc) 
    count_filter(sc,verbose=True)
    normalize_to_median(sc)
    hvg_batch(sc)
    remove_cell_cycle(sc,verbose=True)
    scale.log1p(sc)
    embedding.pca(sc,gene_flag='merge_hvg',set_loadings=True)
    util.to_pickle(sc,sc.cfg['pickle']['embedding_neurons'])
    print(f"Pickle dump to {sc.cfg['pickle']['embedding_neurons']}")

def build_umap_embedding(sc):
    emb,red = plot.build_umap(sc.pca.components,
            n_components=sc.cfg.getint("umap","n_components"),
            n_neighbors=sc.cfg.getint("umap","n_neighbors"),
            min_dist=sc.cfg.getfloat("umap","min_dist"))
    sc.umap_embedding = emb 

def multi_umap_builds(sc):
    n_components = 2
    #neighbors = [25,50,75,100]
    neighbors = [5,15,25,35]
    min_dist = [15,20,25,30]
    for k in neighbors:
        for d in min_dist:
            print(f"(# neighbors, min_dist) = ({k},{d})")
            emb,red = plot.build_umap(sc.pca.components,
                    n_components=n_components,
                    n_neighbors=k,
                    min_dist = d/100.
                    )
            fout = f'data/packer2019/umap/embedding_{k}_{d}.npy'
            np.save(fout,emb)

def plot_umap_builds(sc):
    #neighbors = [25,50,75,100]
    neighbors = [5,15,25,35]
    min_dist = [15,20,25,30]
    fig,_ax = plt.subplots(4,4,figsize=(20,20))
    ax = _ax.flatten() 
    idx = 0
    for k in neighbors:
        for d in min_dist:
            fin = f'data/packer2019/umap/embedding_{k}_{d}.npy'
            U = np.load(fin)
            ax[idx].scatter(U[:,0],U[:,1],s=5,c='#9f9f9f')
            ax[idx].set_xticks([])
            ax[idx].set_yticks([])
            ax[idx].text(0.05,0.9,f'n_neigh={k}, min_dist={d/100.}',transform=ax[idx].transAxes,fontsize=8)
            idx += 1
    plt.show()
 
def multi_res_clustering(sc):
    k_vals = np.arange(25, 101, 25)
    for k in tqdm(k_vals,desc='k:'):
        communities, graph, Q = phenograph.cluster(sc.pca.components,k=k)
        sc.cells[f'pheno_k_{k}'] = pd.Categorical(communities)
    try:
        util.to_pickle(sc,sc.loaded_from)
        print(f"Pickle dump to {sc.loaded_from}")
    except:
        print('Need a loaded from file pointer')

def plot_multi_res_clusters(sc):
    k,d = 5,20
    umap = f'data/packer2019/umap/embedding_{k}_{d}.npy'
    U = np.load(umap)
    fig,ax = plt.subplots(1,4,figsize=(40,10))
    for (idx,k) in enumerate([25,50,75,100]):
        label = f'pheno_k_{k}'
        cmap = sc.cells[label].tolist()
        ax[idx].scatter(U[:,0],U[:,1],s=5,c=cmap,cmap='tab20')
        ax[idx].set_xticks([])
        ax[idx].set_yticks([])
        ax[idx].text(0.05,0.9,f'clusters_k={k}',transform=ax[idx].transAxes,fontsize=12)
        #print(sorted(sc.cells['pheno_k_100'].unique().tolist()))
    plt.show()

def plot_clusters(sc):
    k,d,cls = 5,20,50
    umap = f'data/packer2019/umap/embedding_{k}_{d}.npy'
    cls = f'pheno_k_{cls}'
    cmap = sc.cells[cls].tolist()
    U = np.load(umap)
    fig,ax = plt.subplots(1,1,figsize=(20,20))
    ax.scatter(U[:,0],U[:,1],s=5,c=cmap,cmap='tab20')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.05,0.9,f'clusters_k={cls}',transform=ax.transAxes,fontsize=12)
    cursor = mplcursors.cursor(hover=True)
    cursor.connect(
        "add", lambda sel: sel.annotation.set_text(
            f"Cell: {sc.cells[sc.cell_key][sel.target.index]}, Cluster: {sc.cells[cls][sel.target.index]}")
    )
    plt.show()

def evaluate_clusters(sc):
    from collections import Counter
    k,d,cls = 5,20,25
    umap = f'data/packer2019/umap/embedding_{k}_{d}.npy'
    cls = f'pheno_k_{cls}'
    cls = sc.cells[cls].tolist()
    pid = sc.cells[sc.cell_key].tolist()
    print(len(set(pid))) 
    print(len(set(cls))) 
    N = len(sc.cells)
    C = {}
    for (idx,i) in enumerate(cls):
        if i not in C: C[i] = []
        C[i].append(pid[idx])
     
    Z = []
    for (k,v) in C.items():
        d = Counter(v)
        _v = set(v)
        l = float(len(v))
        #print(l)
        num = [d[u] for u in _v]
        print(l,len(v),min(num))#,num)
        Z = Z + num
    print(len(Z))  
    x,y = ecdf(Z,reverse=True)
    fig,ax = plt.subplots(1,1,figsize=(10,10))
    ax.plot(x,y)
    ax.set_xlim([0,10])

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
            sc = util.load_sc(cfg,load_light=False)
        
    eval(args.mode + '(sc)')


