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
from collections import namedtuple

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


@run_hvg
def poisson_dispersion(X,**kwargs):
    from scipy import stats
    _x,_y = _poisson_dispersion(X)
    yr = _y>1
    x = _x[yr]
    y = _y[yr]
    slope, intercept, r, p, std_err = stats.linregress(x, y=y)
    resid = _y - (slope*_x + intercept)
    rstd = np.std(resid)
    rnorm = np.divide(resid - np.mean(resid), np.std(resid))
    
    """ This is a hack to maintain consistency with the loess class""" 
    M = namedtuple('Model','predict inputs')
    I = namedtuple('inputs','x') 
    def predict(newdata):
        P = namedtuple("P","values") 
        return P((slope * newdata)+intercept)
    
    model = M(predict,I(_x))
    idx = np.argsort(rnorm)[::-1]
    return idx, model

def _poisson_dispersion(X):
    eps = 1e-5
    mean,var = mp.axis_mean_var(X,axis=0,skip_zeros=False) 
    cv2 = np.divide(var,np.power(mean+eps,2))
    cv2 = np.log2(cv2+eps) 
    mean = np.log2(mean+eps)
    return mean,cv2
 

@run_hvg
def poisson_zero_count(X,**kwargs):
    x,y = _poisson_zero_count(X)
    #xx = np.zeros((len(x),2))
    #xx[:,0] = x
    #xx[:,1] = y
    
    """
    import matplotlib.pyplot as plt
    from sklearn.neighbors import NearestNeighbors

    nbrs = NearestNeighbors(n_neighbors = 3)
    nbrs.fit(xx)
    distances,index = nbrs.kneighbors(xx)
    fig,ax = plt.subplots(1,1,figsize=(10,10))
    ax.hist(distances.mean(axis=1),bins=100)
    """
    
    """ This is a hack to maintain consistency with the loess class""" 
    M = namedtuple('Model','predict inputs')
    I = namedtuple('inputs','x') 
    def predict(newdata):
        P = namedtuple("P","values") 
        mu = 2**newdata
        n = X.shape[0]
        _y = n*np.exp(-mu) + (mu / np.sqrt(n))
        return P(_y)
    
    model = M(predict,I(x))

    ## Fit loess
    model1 = loess(x, y, span=0.3, degree=2)
    model1.fit()
    estimate_nz = model1.outputs.fitted_values
    reg_std = np.sqrt(10 ** estimate_nz)
    
    ## Variance of standardized values
    mean = mp.axis_mean(X,axis=0) 
    norm_nz = np.exp(-mean+reg_std)

    idx = np.argsort(norm_nz)[::-1]
    
    return idx, model

def _poisson_zero_count(X):
    eps = 1e-5 
    mean = mp.axis_mean(X,axis=0,skip_zeros=False)
    mean = np.log2(mean + eps) 
    num_z = X.shape[0] - mp.axis_elements(X,axis=0)
    
    return mean,num_z

@run_hvg
def poisson_zero_count_vs(X,**kwargs):
    _x,_y = _poisson_zero_count_vs(X)
    inlier = _x <= 2
    x = _x[inlier]
    y = _y[inlier]

    ## Fit loess
    model = loess(x, y, span=0.3, degree=2)
    model.fit()
    estimate_var = model.outputs.fitted_values
    reg_std = np.sqrt(10 ** estimate_var)
 
    ## Variance of standardized values
    #mean = mp.axis_mean(X,axis=0) 
    #norm_nz = np.exp(-mean+reg_std)

    idx = np.argsort(y)[::-1]
    
    return idx, model

def _poisson_zero_count_vs(_X):
    X = _X.copy()
    X.data = np.sqrt(X.data)
    eps = 1e-5 
    mean = mp.axis_mean(X,axis=0,skip_zeros=False)
    #mean = np.log2(mean + eps) 
    num_z = X.shape[0] - mp.axis_elements(X,axis=0)
    return mean,num_z

@run_hvg
def poisson_zero_count_zi(X,**kwargs):
    x,y = _poisson_zero_count_zi(X)
    
    ## Fit loess
    model = loess(x, y, span=0.5, degree=2)
    model.fit()
    estimate_obs = model.outputs.fitted_values
    reg_std = np.sqrt(1-estimate_obs)
 
    ## Variance of standardized values
    #mean = mp.axis_mean(X,axis=0) 
    #norm_nz = np.exp(-mean+reg_std)
    diff = (y-estimate_obs)/reg_std
    idx = np.argsort(diff)[::-1]
    
    return idx, model

def _poisson_zero_count_zi(X):
    exp = np.exp(-mp.axis_mean(X,axis=0,skip_zeros=False))
    #mean = np.log2(mean + eps) 
    obs = 1 - (mp.axis_elements(X,axis=0)/float(X.shape[0]))
    return exp,obs

