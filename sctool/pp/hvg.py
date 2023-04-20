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

def merge_batch(sc,label='merge_hvg'):
    hvg = sc.genes[sc.meta['batch_hvg']].to_numpy()
    hvg = hvg.sum(1)
    hvg[hvg > 0] = 1
    sc.genes[label] = hvg

def batch_vs_all_matrix(sc):
    bkeys = sc.meta['batch_hvg'] + ['hvg']
    N = len(bkeys) 
    Z = np.zeros((len(bkeys),len(bkeys)))
    for i in range(N):
        x = sc.genes[[bkeys[i]]].to_numpy() 
        for j in range(i,N):
            y = sc.genes[[bkeys[j]]].to_numpy()
            Z[i,j] = cossim(x,y)
    return Z

def cossim(x,y):
    x = x.reshape(len(x))
    y = y.reshape(len(y))
    return np.dot(x,y) / float(np.sqrt(x.sum()) * np.sqrt(y.sum()))

def merge_batch_vs_all(sc):
    hall = sc.genes[['hvg']].to_numpy()
    hall = hall.reshape(len(hall))
    hbatch = sc.genes[sc.meta['batch_hvg']].to_numpy()
    hbatch = hbatch.sum(1)
    hbatch[hbatch > 0] = 1
    return cossim(hall,hbatch)


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




