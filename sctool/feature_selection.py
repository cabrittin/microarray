"""
@name: feature_selection.py
@description:
    Functions for feature (i.e. gene selection).

@author: Christopher Brittin
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05
"""

import toolbox.matrix_properties as mp
import numpy as np
from skmisc.loess import loess

def hvg_seurat3(X,return_model=False,num_genes=0):
    """
    Identifies highly variable genes using the Seurat3
    methodology. For details, see https://doi.org/10.1101/460147

    Parameters:
    -----------
    X: cells by gene array
    
    return_model: bool (optional, default: False)
        If true, return the loess model use to fit the data
   
    num_genes: int (optional, default all)
        If greater than 1, only returns the largest num_genes. 

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

    if num_genes > 0: idx = idx[:num_genes]
    if return_model:
        return idx, model
    else:
        return idx


