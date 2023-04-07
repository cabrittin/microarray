"""
@name:                         
@description:                  


@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05              
"""

import sys
import argparse

from pycsvparser import read,write


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('seurat_rows',
                        action = 'store',
                        help = 'Seurat rows')

    parser.add_argument('wb',
                        action = 'store',
                        help = 'WB file')
   
    parser.add_argument('fout',
                        action = 'store',
                        help = 'Output file')

    params = parser.parse_args()

    #A = [a[1].replace('"','') for a in read.into_list(params.seurat_rows,multi_dim=True,delimiter=' ',skip_header=1)]
    A = read.into_list(params.seurat_rows)
    
    B = read.into_list(params.wb,delimiter="\t",multi_dim=True,skip_header=4)
    
    C = {}
    for r in B:
        if r[1] != 'Caenorhabditis elegans': continue
        C[r[2]] = [r[3],r[4]]
    
    D = [['id','short_id','locus']]
    for k in A:
        try:
            D.append([k] + C[k])
        except:
            D.append([k,'',''])
    
    write.from_list(params.fout,D)
    
    
