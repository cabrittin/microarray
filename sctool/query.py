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

import toolbox.matrix_properties as mp
from sctool import scale

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

def cell_total_counts(sc,genes=None):
    """
    Returns total counts for cell across provided genes.
    If genes is None, then counts across all cells are taken
    
    Args:
    -----
    sc: SingleCell object 
    genes: str,list
        Name of genes
    """
    if genes is not None:
        jdx = sc.get_gene_index(genes)
        X = matrix_cols(sc.X,jdx)
        csum = mp.axis_counts(X,axis=1)
    else:
        csum = mp.axis_counts(sc.X,axis=1)

    return csum

def qc_cell_total_count(sc,genes=None,thresh=0,label='qc_total_count'):
    """
    Adds columns to cell meta if qc cell total count thresh is met
    """
    x = np.log(cell_total_counts(sc,genes=genes)+1)
    x[x<thresh] = 0
    x[x > 0] = 1
    x = x.astype(int)
    sc.cells[label] = x

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


