"""
@name: query.py                      
@description:                  
    Functions for querying data from SingleCell object
    Generally no assumptions are made about the datatype of count martrix (i.e., any scaling is already assumed to be complete)

@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05              
"""

import scipy.sparse as sp
import numpy as np
import pandas as pd
from scipy import stats
from collections import namedtuple
import sklearn.decomposition as skd
from scipy.spatial.distance import pdist

import toolbox.matrix_properties as mp
#from sctool.feature_selection import hvg_seurat3

def matrix_rows(X,idx):
    if sp.issparse(X): 
        return X.tocsr()[idx,:].tocoo()
    else:
        return X[idx,:]

def matrix_cols(X,jdx):
    if sp.issparse(X): 
        return X.tocsr()[:,jdx].tocoo()
    else:
        return X[:,jdx]

def gene_counts(sc,genes,key=None):
    """
    Returns count data for provided genes
    
    Args:
    -----
    sc: SingleCell object 
    genes: str,list
        Name of genes
    key: str, optional (default=None)
        Name of gene meta key. If None, then the default key will be used

    """
    jdx = sc.get_gene_index(genes,key=key)
    X = matrix_cols(sc.X,jdx)
    return X

def run_query(func):
    def inner(sc,genes=None,cells=None,**kwargs):
        """
        Decorator function for making queries on the count matrix rows (cells)

        Function provides a convenient wrapper to allow users to specify a subset of genes.

        Args:
        -----
        sc: SingleCell object 
        genes: str,list
            Name of genes
        """
        X = sc.X.copy()

        if genes is not None:
            jdx = sc.get_gene_index(genes)
            X = matrix_cols(X,jdx)
        
        if cells is not None:
            X = matrix_rows(X,cells)

        return func(X,**kwargs)
    
    return inner


@run_query
def cell_total_counts(X,**kwargs):
    """
    Returns total counts for cells.
    """ 
    return mp.axis_sum(X,axis=1)

@run_query
def cell_num_genes(X,**kwargs):
    """
    Returns number of genes expressed in cell across provided genes.
    If genes is None, then counts across all cells are taken
    """ 
    return mp.axis_elements(X,axis=1)

@run_query
def size_factor(X,**kwargs):
    """
    Computes the size factor used in the Monocle package
    
    Size factor is defined as the cell’s total UMI count divided by the 
    geometric mean of all cells’ total UMI counts

    """
    csum = mp.axis_sum(X,axis=1)
    return csum / np.exp(np.mean(np.log(csum)))

@run_query
def median_cell_count(X,**kwargs):
    """
    Computes the median cell count
    """
    return np.median(mp.axis_sum(X,axis=1))

@run_query
def mean_gene_count(X,**kwargs):
    """
    Computes the mean gene count
    """
    return mp.axis_mean(X,axis=0)


def minimum_cells_with_gene(sc,thresh,label='total_cells'):
    x = mp.axis_elements(sc.X,axis=0)
    qc = np.zeros(len(x),dtype=int)
    qc[x>=thresh] = 1
    sc.genes[label] = qc

def minimum_genes_in_cell(sc,thresh,label='total_cells'):
    x = mp.axis_elements(sc.X,axis=1)
    qc = np.zeros(len(x),dtype=int)
    qc[x>=thresh] = 1
    sc.cells[label] = qc

def batch_background(sc,xlabel='total_umi',thresh=50):
    for b in sorted(sc.batches):
        print(b)
        idx = sc.cells.index[sc.cells[b] == 1].to_numpy()
        X = cell_total_counts(sc,cells=idx)
        idx = idx[X<thresh]
        print(X.shape,idx.shape,X.min(),X.max())


def qc_residual_filter(sc,x,y,thresh=-2,label='qc_residual_filter'):
    slope, intercept, r, p, std_err = stats.linregress(x, y)
    resid = y - (slope* x + intercept)
    rnorm = np.divide(resid - np.mean(resid), np.std(resid))
    if thresh < 0: 
        resid[rnorm < thresh] = 0
        resid[rnorm >= thresh] = 1
    else:
        resid[rnorm > thresh] = 0
        resid[rnorm <= thresh] = 1
    
    resid = resid.astype(int)
    sc.cells[label] = resid
    RF = namedtuple("Residual", "slope intercept r p std_err")
    sc.residual_filter = RF(slope,intercept,r,p,std_err) 


def gene_mean_filter(sc,thresh,label='qc_gene_mean'):
    mu = mp.axis_mean(sc.X,axis=0,skip_zeros=False)
    mu[mu < thresh] = 0
    mu[mu > 0 ] = 1
    sc.genes[label] = mu

def gene_zero_count_filter(sc,thresh,label='qc_zero_count'):
    mu = mp.axis_elements(sc.X,axis=0)
    mu[mu < thresh] = 0
    mu[mu > 0 ] = 1
    sc.genes[label] = mu
    

def label_gene_counts(sc,genes,labels,key=None,std_scale=False):
    """
    Returns pandas dataframe with cell meta colomns and genes for provided genes
    
    Args:
    -----
    sc: SingleCell object 
    genes: str,list
        Name of genes
    labels: str,list
        Cell meta columns to use as labels
    key: str, optional (default=None)
        Name of gene meta key. If None, then the default key will be used
    std_scale: bool, optional (default = False)
        If True, gene counts will be standardized across cells 
    """
    if type(genes) is not list: genes = [genes]
    if type(labels) is not list: labels = [labels]
    if key is None: key = sc.gene_key
    
    X = gene_counts(sc,genes,key=key)
    if sp.issparse(X): X = X.todense()
    if std_scale: X = scale.standardize(X,axis=0)
    #print(X.mean(0),X.std(0))
    
    df1 = sc.cells[labels]
    df2 = pd.DataFrame(X,columns=genes)
    return pd.concat([df1,df2],axis='columns')

def pca(sc,gene_flag=None,n_components=50,set_loadings=False,**kwargs):
    """
    gene_flag: subselect genes based on conditional flag, must alread be set in genes dataframe
    """
    X = sc.X.toarray() 
    sc.pca = skd.PCA(n_components=n_components,**kwargs)
    if gene_flag is not None:
        jdx = sc.genes.index[sc.genes[gene_flag] == 1].tolist()
        sc.pca.components = sc.pca.fit_transform(X[:,jdx])
    else:
        sc.pca.components = sc.pca.fit_transform(X)
    
    if set_loadings:
        sc.pca.loadings = sc.pca.components_.T*np.sqrt(sc.pca.explained_variance_)

def similarity_matrix(sc,metric='jaccard',pca=True,**kwargs):
    if pca:
        return pdist(sc.pca.components,metric,**kwargs)
    else:
        return pdist(sc.X,metric,**kwargs)

