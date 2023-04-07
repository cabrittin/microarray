"""
@name: csv_to_sc_obj.py                        
@description:                  
    Formats csv table to a format that can be read into a single cell object
    Outputs:
        1. cell meta file
        2. gene meta file
        3. a binarizied (.npz) 'count' matrix

@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05              
"""

import sys
from configparser import ConfigParser,ExtendedInterpolation
import argparse
from scipy.sparse import coo_array,save_npz
import pandas as pd

if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('fin',
                        action = 'store',
                        help = 'Function call')

    parser.add_argument('--gene_by_cell',
                    action = 'store_true',
                    default=False,
                    required=False,
                    help=('By default, table is assumed to be cell by gene. '
                        'If flag, table will be treated as gene by cell'))
    
    parser.add_argument('--col_pad',
            action = 'store',
            default=1,
            type=int,
            required=False,
            help = 'Number of left columns which label rows')
    

    params = parser.parse_args()
    
    
    df = pd.read_csv(params.fin)

    columns = df.columns.tolist()


    if params.gene_by_cell:
        genes = df.iloc[:,:params.col_pad]
        cells = [c for c in columns[params.col_pad:]]
        cells = pd.DataFrame(cells,columns=['cell_id'])
    else:
        cells = df.iloc[:,:params.col_pad]
        genes = [g for g in columns[params.col_pad:]]
        genes = pd.DataFrame(genes,columns=['gene_id'])
    
    counts = df.iloc[:,params.col_pad:].to_numpy()
    counts = coo_array(counts)
    
    cell_out = params.fin.replace('.csv','_cell_meta.csv')
    gene_out = params.fin.replace('.csv','_gene_meta.csv')
    count_out = params.fin.replace('.csv','_count_matrix_meta.npz')
    
    cells.to_csv(cell_out,index=False)
    genes.to_csv(gene_out,index=False)
    save_npz(count_out,counts)

    print(f'Cells meta --> {cell_out}')
    print(f'Genes meta --> {gene_out}')
    print(f'Count matrix --> {count_out}')
