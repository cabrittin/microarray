"""
@name: hvg.py
@description:
    Functions for identifying the highly varying genes

@author: Christopher Brittin
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05
"""

import toolbox.matrix_properties as mp
import numpy as np
from skmisc.loess import loess
from tqdm import tqdm

import toolbox.matrix_properties as mp

def flag_hvg(sc,method='mean_variance',num_hvg=1000,label='hvg',keep_model=False):
    hvg,model = globals()[method](sc.X,num_hvg)
    sc.genes[label] = hvg
    if keep_model: sc.hvg_model = model
     
def flag_hvg_batch(sc,method='mean_variance',num_hvg=1000,label='hvg',meta_key='batch_hvg',keep_model=False):
    sc.meta[meta_key] = []
    if keep_model: sc.hvg_batch_model = {} 
    sc.X = sc.X.tocsr()
    for b in tqdm(sc.batches,desc='Batches processed:'):
        idx = sc.cells.index[sc.cells[b] == 1].tolist()
        hvg,model = globals()[method](sc.X[idx,:],num_hvg)
        bkey = b + '_hvg'
        sc.meta[meta_key].append(bkey)
        sc.genes[bkey] =  hvg
        if keep_model: sc.hvg_batch_model['b'] = model
    sc.X = sc.X.tocoo()

def merge_hvg_batch(sc,label='merge_hvg'):
    hvg = sc.genes[sc.meta['batch_hvg']].to_numpy()
    hvg = hvg.sum(1)
    hvg[hvg > 0] = 1
    sc.genes[label] = hvg

def run_hvg(func):
    def inner(X,num_hvg,**kwargs):
        idx,model = func(X,**kwargs)
        hvg = np.zeros(X.shape[1],dtype=int)
        hvg[idx[:num_hvg]] = 1
        return hvg,model

    return inner

@run_hvg
def mean_variance(X,**kwargs):
    """
    Identifies highly variable genes using the Seurat3
    methodology. For details, see https://doi.org/10.1101/460147

    Parameters:
    -----------
    X: cells by gene array
    
    Returns:
    ----------
    rank_hvg: A ranked (descending order) of genes by variance
    
    model: If return_model, returns the loess model

    """
    mean,var = mp.axis_mean_var(X,axis=0,skip_zeros=False) 
    not_const = var > 0
    estimate_var = np.zeros(X.shape[1], dtype=np.float64)
    x = np.log10(mean[not_const])
    y = np.log10(var[not_const])
    
    ## Fit loess
    model = loess(x, y, span=0.3, degree=2)
    model.fit()
    estimate_var[not_const] = model.outputs.fitted_values
    reg_std = np.sqrt(10 ** estimate_var)
    
    ## Clip values
    N = X.shape[0] 
    clip_val = reg_std * np.sqrt(N) + mean
    batch_counts = mp.axis_clip_value(X,clip_val,axis=0)

    ## Variance of standardized values
    norm_gene_var = mp.var_of_user_standardized_values(X,reg_std,axis=0) 
    idx = np.argsort(norm_gene_var)[::-1]

    return idx, model




