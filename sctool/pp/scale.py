"""
@name: scale.py                      
@description:                  
    Performs various transformations, scalings and normalizations

@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05              
"""

import toolbox.scale as tsc
import toolbox.matrix_properties as mp
import numpy as np
import scipy.sparse as sp

def to_target(sc,target):
    sc.X = tsc.sum_to_target(sc.X,target,axis=1)

def cells_by_vector(sc,x=None,label=None):
    # Check if coo, if coo make sure to conver back to coo
    to_coo = (sc.X.getformat() == 'coo')

    if x is None:
        x = sc.cells[label].to_numpy()
    r,c = sc.X.nonzero()
    rD_sp = sp.csr_matrix(((1.0/x)[r], (r,c)), shape=(sc.X.shape))
    sc.X = sc.X.multiply(rD_sp)
    if to_coo: sc.X = sc.X.tocoo()


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

def log_normalization(sc,scale=1000):
    sc.X = tsc.sum_to_target(sc.X,scale,axis=1)
    sc.X.data = np.log(sc.X.data + 1)


def round_averaged_counts(sc):
    sc.X.data = np.around(sc.X.data)
    sc.X.eliminate_zeros()

def standardize(sc,axis=0):
    """
    Note is sc.X is sparse then standardizing with result in dense array
    """
    if sp.issparse(sc.X): sc.X = sc.X.toarray()
    std = sc.X.std(axis=axis)
    std[std==0] = 1e-5
    if axis == 1: std = std.reshape(-1,1)
    return np.true_divide(sc.X - sc.X.mean(axis=axis),std)

def clip(sc,clip_val):
    if sp.issparse(sc.X): return mp.axis_clip_value(sc.X,clip_val)
    
    X = sc.X.copy()
    X[X>clip_val] = clip_val
    return X

def normalize_per_cell(sc,counts_per_cell_after=1000000,copy=True):
    return tsc.sum_to_target(sc.X,counts_per_cell_after,axis=1)

def log1p(sc):
    if sp.issparse(sc.X):
        sc.X.data = np.log(sc.X.data)
    else:
        sc.X = np.log(sc.X + 1)
