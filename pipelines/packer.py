"""
@name: filters.packer.py                        
@description:                  
    Some convenicence filters for packer dataset

@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05              
"""

import matplotlib.pyplot as plt
from pycsvparser import read
from sctool import query,qc,scale,batch,explore
from sctool.feature_selection import hvg_seurat3

def reduce_to_neurons(sc,verbose=False):
    if verbose: print(f"# cells before neuron removal: {len(sc.cells)}") 
    cells = read.into_list(sc.cfg['mat']['neurons'])
    sc.filter_cells_isin('cell.subtype',cells)
    if verbose: print(f"# cells after neuron removal: {len(sc.cells)}") 

def split_into_batch(sc):
    batch.split(sc)

def default(sc):
    count_filter(sc)
    query.qc_mean_var_hvg(sc)
    explore.hvg_mean_var(sc) 
    plt.show()
    med_libsize = query.median_cell_count(sc)
    Xn = scale.normalize_per_cell(sc,counts_per_cell_after=med_libsize,copy=True)
    sc.set_count_matrix(Xn)
    scale.log1p(sc)
    print("Run pca")
    query.pca(sc,gene_flag='qc_hvg')

def hvg_batch(sc):
    query.qc_mean_var_hvg(sc,batch=True)

def hvg_all(sc):
    query.qc_mean_var_hvg(sc,batch=False)



def count_filter(sc,verbose=False):
    if verbose: print('Matrix shape before count filter: ',sc.X.shape) 
    thresh = sc.cfg.getint('filters','min_cells_with_gene') 
    query.minimum_cells_with_gene(sc,thresh,label='total_cells')
    thresh = sc.cfg.getint('filters','min_genes_in_cells') 
    query.minimum_genes_in_cell(sc,thresh,label='total_genes') 
    sc.filter_genes_bool('total_cells',1)
    sc.filter_cells_bool('total_genes',1)
    if verbose: print('Matrix shape after count filter: ',sc.X.shape) 

def set_mitochondria(sc):
    sc.load_gene_list('mitochondria')
    sc.cells['total_count'] = query.cell_total_counts(sc)
    sc.cells['mt_count'] = query.cell_total_counts(sc,genes=sc.gene_list)
    query.qc_residual_filter(sc,sc.cells['total_count'].tonumpy(),
            sc.cells['mt_count'].tonumpy(),thresh=-2,label='qc_mt')


"""
query.qc_residual_filter(sc,x,y,thresh=-2,label='qc_mt')
sc.filter_cells_eq('qc_mt',1)
print('Shape after filter: ',sc.X.shape) 
"""
