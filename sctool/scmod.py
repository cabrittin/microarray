"""
@name: sctool.scmod.py                     
@description:                  
    Module for modify the SingleCell class

@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05              
"""

import scipy.sparse as sp
import numpy as np

def split_cells_by_key(sc,key,label='batches'):
    assert not hasattr(sc,label), "SingleCell instance already has {label} attribute, consider using another attribute label" 
    #bkey = sc.cfg['keys']['meta_key_batch']
    batches = list(set(sc.cells[key].tolist()))
    for b in batches:
        bflag = np.zeros(len(sc.cells),dtype=int)
        idx = sc.cells.index[sc.cells[key] == b].tolist()
        bflag[idx] = 1
        sc.cells[b] = bflag
    setattr(sc,label,batches)

def filter_matrices_rows(sc,idx):
    if sp.issparse(sc.X): 
        sc.X = sc.X.tocsr()[idx,:].tocoo()
    else:
        sc.X = sc.X[idx,:]

def filter_matrices_cols(sc,jdx):
    if sp.issparse(sc.X): 
        sc.X = sc.X.tocsr()[:,jdx].tocoo()
    else:
        sc.X = sc.X[:,jdx]


def select_cell_flag(sc,flag,val=1):
    """
    Keep cells (matrix rows) from the data.

    Parameters
    ----------
    idx: list
        Cell indicies to keep. Indicies can be obtained from 
        dataframe query. 
    
    Example
    -------
    >> sc = SingleCell(cfg)
    >> idx = sc.cells.index[sc.cells["attribute"] == "value"].tolist()
    >> sc.select_cell(idx)

    """
    idx = sc.cells.index[sc.cells[flag]==val].tolist()
    sc.cells = sc.cells.iloc[idx]#.reset_index()
    sc.cells.index = range(len(sc.cells.index))
    if not sc.load_light: filter_matrices_rows(sc,idx) 

def select_gene_flag(sc,flag,val=1):
    """
    Select genes (matrix columns) from the data.

    Parameters
    ----------
    jdx: list
        Gene indicies to keep. Indicies can be obtained from 
        dataframe query.
    
    Example
    -------
    >> sc = SingleCell(cfg)
    >> jdx = sc.genes.index[sc.genes["name"].isin(['gene1','gene2','gene3'])].tolist()
    >> sc.select_genes(jdx)

    """
    idx = sc.genes.index[sc.genes[flag]==val].tolist()
    sc.genes = sc.genes.iloc[idx]#.reset_index()
    sc.genes.index = range(len(sc.genes.index))
    if not sc.load_light: filter_matrices_cols(sc,idx) 

def flag_cells_isin(sc,attr,values,flag):
    """
    Flag cells (rows) with meta attibute in some values list.

    Parameters
    ----------
    sc : SingleCell class 
    attr: string
        Cells dataframe column name
    values: list
        Cells with attr in list values will be kept 
    
    """
    jdx = sc.cells.index[sc.cells[attr].isin(values)].tolist()
    #select_cells(sc,jdx)
    _flag = np.zeros(len(sc.cells)) 
    _flag[jdx] = 1 
    sc.cells[flag] = _flag
    return flag
    
def flag_genes_isin(sc,attr,values,flag):
    """
    Flag genes (cols) with meta attibute in some values list.

    Parameters
    ----------
    sc : SingleCell class 
    attr: string
        Cells dataframe column name
    values: list
        Cells with attr in list values will be kept 
    
    """
    jdx = sc.genes.index[sc.genes[attr].isin(values)].tolist()
    #select_cells(sc,jdx)
    _flag = np.zeros(len(sc.genes)) 
    _flag[jdx] = 1 
    sc.genes[flag] = _flag
    return flag
 

