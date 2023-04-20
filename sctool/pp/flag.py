"""
@name: sctool.pp.flag.py                       
@description:                  
    Functions for flagging quality control

@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05              
"""

import numpy as np
from tqdm import tqdm

import toolbox.matrix_properties as mp
from sctool.pp import hvg as hvg_


def minimum_cells_with_gene(sc,thresh,label='pp_cells_with_gene'):
    x = mp.axis_elements(sc.X,axis=0)
    pp = np.zeros(len(x),dtype=int)
    pp[x>=thresh] = 1
    sc.genes[label] = pp
    return label

def minimum_genes_in_cell(sc,thresh,label='pp_genes_in_cell'):
    x = mp.axis_elements(sc.X,axis=1)
    pp = np.zeros(len(x),dtype=int)
    pp[x>=thresh] = 1
    sc.cells[label] = pp
    return label

def meta_cells_lte(sc,meta_key,thresh,label='pp_cell_meta'):
    x = sc.cells[meta_key].to_numpy()
    pp = np.zeros(len(x),dtype=int)
    pp[x<=thresh] = 1
    sc.cells[label] = pp
    return label

def meta_genes_lte(sc,meta_key,thresh,label='pp_gene_meta'):
    x = sc.genes[meta_key].to_numpy()
    pp = np.zeros(len(x),dtype=int)
    pp[x<=thresh] = 1
    sc.genes[label] = pp
    return label

def hvg(sc,method='mean_variance',num_hvg=1000,label='hvg',keep_model=False):
    _hvg,model = getattr(hvg_,method)(sc.X,num_hvg)
    sc.genes[label] = _hvg
    if keep_model: sc.hvg_model = model
 
def hvg_batch(sc,method='mean_variance',num_hvg=1000,label='hvg',meta_key='batch_hvg',keep_model=False):
    sc.meta[meta_key] = []
    if keep_model: sc.hvg_batch_model = {} 
    sc.X = sc.X.tocsr()
    for b in tqdm(sc.batches,desc='Batches processed:'):
        idx = sc.cells.index[sc.cells[b] == 1].tolist()
        _hvg,model = getattr(hvg_,method)(sc.X[idx,:],num_hvg)
        bkey = b + '_hvg'
        sc.meta[meta_key].append(bkey)
        sc.genes[bkey] =  _hvg
        if keep_model: sc.hvg_batch_model['b'] = model
    sc.X = sc.X.tocoo()
    hvg_.merge_batch(sc)


