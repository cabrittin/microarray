"""
@name: scale.py                      
@description:                  
    Performs various transformations, scalings and normalizations

@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05              
"""

import toolbox.scale as tsc
import numpy as np
import scipy.sparse as sp

def to_target(sc,target):
    sc.X = tsc.sum_to_target(sc.X,target,axis=1)

def seurat_log_scale(sc,scale=10000):
    """
    Performs the 'LogNormalization' in Seurat.
    Note that this is a scaling NOT a normalization.
    Count matrix is scaled by the sum of each row,
    multiplied by a scaling factor (defautl 1e5),
    and then log-transformed
    """
    sc.X = tsc.sum_to_target(sc.X,scale,axis=1)
    #sc.log_scale = True
    sc.X.data = np.log(sc.X.data + 1)

def round_averaged_counts(sc):
    sc.X.data = np.around(sc.X.data)
    sc.X.eliminate_zeros()

def standardize(X,axis=0):
    std = X.std(axis=axis)
    if axis == 1: std = std.reshape(-1,1)
    return np.true_divide(X - X.mean(axis=axis),std)

def normalize_per_cell(sc,counts_per_cell_after=10000,copy=True):
    return tsc.sum_to_target(sc.X,counts_per_cell_after,axis=1)

def log1p(sc):
    if sp.issparse(sc.X):
        sc.X.data = np.log(sc.X.data)
    else:
        sc.X = np.log(sc.X + 1)
