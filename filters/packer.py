"""
@name: filters.packer.py                        
@description:                  
    Some convenicence filters for packer dataset

@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05              
"""

from sctool import query,qc
from sctool.feature_selection import hvg_seurat3

def default(sc):
    print('Shape before filter: ',sc.X.shape) 
    sc.load_gene_list('mitochondria')
    sc.cells['total_count'] = query.cell_total_counts(sc)
    sc.cells['mt_count'] = query.cell_total_counts(sc,genes=sc.gene_list)
    x = query.cell_total_counts(sc)
    y = query.cell_total_counts(sc,genes=sc.gene_list)
    sc.cells['total_count'] = x
    sc.cells['mt_count'] = y
    query.qc_residual_filter(sc,x,y,thresh=-2,label='qc_mt')
    sc.filter_cells_eq('qc_mt',1)
    print('Shape after filter: ',sc.X.shape) 


