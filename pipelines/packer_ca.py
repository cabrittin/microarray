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
QC_FLAG = "passed_initial_QC_or_later_whitelisted"


def total_gene_counts(sc):
    scmod.select_cell_flag(sc,QC_FLAG,True)
    sc.genes['total'] = mp.axis_sum(sc.X,axis=0)
    fig,ax = plt.subplots(1,1,figsize=(10,10))
    pp.plot.plot_ecdf(sc.genes['total'].to_numpy(),ax=ax,xlabel='total_gene_umi',
                    plot_params=sc.cfg['plot_params'])
    ax.set_xlim([0,100])
    plt.show()


def batch_background(sc):
    scmod.split_cells_by_key(sc,sc.cfg['keys']['meta_key_batch'],'batches')
    sc.cells['total_umi'] = pp.query.cell_total_counts(sc) 
    pp.query.batch_background(sc,thresh=50) 

def batch_norm(sc):
    scmod.split_cells_by_key(sc,sc.cfg['keys']['meta_key_batch'],'batches')
    batch_scale = np.zeros(sc.X.shape[0])
    for b in tqdm(sc.batches,desc='Rescale batches by size factor:'):
        idx = sc.cells.index[sc.cells[b] == 1].tolist()
        size_factor = pp.query.size_factor(sc,cells=idx)
        batch_scale[idx] = size_factor
    pp.scale.cells_by_vector(sc,x=batch_med)

    for b in tqdm(sc.batches,desc='Rescale batches by TPM:'):
        idx = sc.cells.index[sc.cells[b] == 1].tolist()
        size_factor = pp.query.tpm(sc,cells=idx)
        batch_scale[idx] = size_factor

def filter_genes_by_min_total(sc):
    sc.genes['total'] = mp.axis_sum(sc.X,axis=0)
    sc.genes['min_total'] = (sc.genes['total'] >= 20)
    scmod.select_gene_flag(sc,'min_total')

def normalize_to_median(sc):
    med_libsize = pp.query.median_cell_count(sc)
    Xn = pp.scale.normalize_per_cell(sc,counts_per_cell_after=med_libsize,copy=True)
    sc.set_count_matrix(Xn)

def scale_by_size_factor(sc):
    sc.cells['size_factor'] = pp.query.size_factor(sc)
    pp.scale.cells_by_vector(sc,label='size_factor') 

def scale_by_tpm(sc):
    scale = 1e6
    sc.genes['tpm_scale'] = pp.query.mean_gene_count(sc)
    print(sc.genes['tpm_scale'].min(),sc.genes['tpm_scale'].max())
    gsum = sc.genes['tpm_scale'].sum()
    print(gsum)
    sc.genes['tpm_scale'] = sc.genes['tpm_scale'] * float(scale) / gsum
    #print(sc.genes['tpm_scale'].sum())
    print(sc.genes['tpm_scale'].min(),sc.genes['tpm_scale'].max())
    pp.scale.genes_by_vector(sc,label='tpm_scale')
    mean = mp.axis_mean(sc.X,axis=0)
    print(min(mean),max(mean))


def run_hvg(sc):
    scmod.select_cell_flag(sc,QC_FLAG,True)
    #batch_norm(sc) 
    reduce_to_neurons(sc,verbose=True) 
    scale_by_size_factor(sc)
    scmod.split_cells_by_key(sc,sc.cfg['keys']['meta_key_batch'],'batches')
    pp.flag.hvg_batch(sc,method='mean_variance',num_hvg=1000,keep_model=True)
    #pp.scale.log1p(sc)
    #filter_genes_by_min_total(sc)
    #scale_by_tpm(sc) 
    pp.flag.hvg(sc,method='mean_variance',num_hvg=1000,keep_model=True) 
    pp.plot.hvg_mean_var(sc,label='merge_hvg') 
    pp.plot.hvg_batch_vs_all(sc)
    plt.show() 

def remove_cell_cycle(sc,verbose=False):
    if verbose: print(f"# genes before CC removeal: {len(sc.genes)}")
    genes = [g[1] for g in read.into_list(sc.cfg['gene_list']['go_cell_cycle'],multi_dim=True)]
    flag = scmod.flag_genes_isin(sc,sc.gene_key,genes,'cell_cycle')
    scmod.select_gene_flag(sc,flag,val=0) 
    if verbose: print(f"# genes after CC removeal: {len(sc.genes)}")

def reduce_to_neurons(sc,verbose=False):
    if verbose: print(f"# cells before neuron removal: {len(sc.cells)}") 
    cells = read.into_list(sc.cfg['mat']['neurons'])
    flag = scmod.flag_cells_isin(sc,sc.cell_key,cells,'neurons')
    scmod.select_cell_flag(sc,flag) 
    #scmod.filter_cells_isin(sc,sc.cell_key,cells)
    if verbose: print(f"# cells after neuron removal: {len(sc.cells)}") 



def pca_embedding(sc):
    scmod.select_cell_flag(sc,QC_FLAG,True)
    reduce_to_neurons(sc,verbose=True) 
    scale_by_size_factor(sc)
    scmod.split_cells_by_key(sc,sc.cfg['keys']['meta_key_batch'],'batches')
    pp.flag.hvg_batch(sc,method='mean_variance',num_hvg=1000,keep_model=False)
    remove_cell_cycle(sc,verbose=True)
    #sc.X = pp.scale.log1p(sc) 
    sc.X = pp.scale.standardize(sc)
    sc.X = pp.scale.clip(sc,10) 
    pp.embedding.pca(sc,gene_flag='merge_hvg',set_loadings=True)
    util.to_pickle(sc,sc.cfg['pickle']['embedding_neurons'])
    print(f"Pickle dump to {sc.cfg['pickle']['embedding_neurons']}")

def pca_plots(sc):
    pp.plot.scree_plot(sc)
    plt.tight_layout()
    plt.savefig("data/packer2019/plots/pp_scree_neurons.png")
    
    pp.plot.pca_loadings(sc)
    plt.tight_layout()
    plt.savefig("data/packer2019/plots/pp_pca_loadings.png")

    plt.show() 

def build_umap_embedding(sc):
    emb,red = pp.plot.build_umap(sc.pca.components,
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
            emb,red = pp.plot.build_umap(sc.pca.components,
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
    fout = 'data/packer2019/plots/embeddings.png'
    plt.savefig(fout)
    plt.show()
 
def multi_res_clustering(sc):
    k_vals = [5,15,25,35,45]
    pp.clustering.phenograph(sc,k_vals,label='pheno_k')
    try:
        util.to_pickle(sc,sc.loaded_from)
        print(f"Pickle dump to {sc.loaded_from}")
    except:
        print('Need a loaded from file pointer')

def plot_multi_res_clusters(sc):
    k,d = 15,25
    k_vals = [5,15,25,35]
    umap = f'data/packer2019/umap/embedding_{k}_{d}.npy'
    U = np.load(umap)
    fig,ax = plt.subplots(1,4,figsize=(40,10))
    for (idx,k) in enumerate(k_vals):
        label = f'pheno_k{k}'
        cmap = sc.cells[label].tolist()
        ax[idx].scatter(U[:,0],U[:,1],s=5,c=cmap,cmap='tab20')
        ax[idx].set_xticks([])
        ax[idx].set_yticks([])
        ax[idx].text(0.05,0.9,f'clusters_k={k}',transform=ax[idx].transAxes,fontsize=12)
        #print(sorted(sc.cells['pheno_k_100'].unique().tolist()))
    
    fout = 'data/packer2019/plots/multi_pheno_clusters.png'
    plt.savefig(fout,dpi=300)
    
    pp.plot.multi_res_clusters(sc)
    fout = 'data/packer2019/plots/multi_pheno_clusters_ari.png'
    plt.savefig(fout)
    
    plt.show()

def plot_clusters(sc):
    print(sc.X.shape)
    k,d,cls = 15,25,25
    umap = f'data/packer2019/umap/embedding_{k}_{d}.npy'
    cls = f'pheno_k{cls}'
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

    fout = 'data/packer2019/plots/pheno_clusters.png'
    plt.savefig(fout,dpi=300)
    plt.show()

def assess_clusters(sc):
    from collections import defaultdict
    import random

    k,d,cls = 15,25,25
    umap = f'data/packer2019/umap/embedding_{k}_{d}.npy'
    cls = f'pheno_k{cls}'
    cid = sc.cells[cls].tolist()
    cmap = sc.cells[sc.cell_key].tolist()
    vals,counts = np.unique(cmap,return_counts=True)
    counts = counts / counts.sum()
    #cls_count[:,1] /= cls_count.shape(0)
    cstore = [] 
    for (i,v) in enumerate(vals):
        for j in range(int(counts[i]*1000)):
            cstore.append(v)
    vstore = defaultdict(list)
    for i in range(len(cmap)): vstore[cmap[i]].append(i) 
    
    score = np.zeros(100)
    tot = 5000
    for l in range(len(score)):
        for k in range(tot):
            c = random.choice(cstore)
            [i,j] = random.sample(vstore[c],2)
            score[l] += int(cid[i] == cid[j])
            if cid[i] != cid[j]:
                print(c,cid[i],cid[j])
    score = score/float(tot)
    print(np.mean(score),np.std(score)/10.)


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

