"""
@name: count_matrix_to_binary.py                       
@description:                  
    Converts matrix market format to a sparse binary cell (rows) by gene (cols) array

@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05              
"""

import sys
import argparse

import numpy as np
import scipy.io
import scipy.sparse as sp

if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)


    parser.add_argument('mtx',
                        action = 'store',
                        help = 'mtx file')

    params = parser.parse_args()
    
    A = scipy.io.mmread(params.mtx)

    fout = params.mtx.replace('.mtx','.npz')
    
    print("Matrix shape: ", A.shape)

    scipy.sparse.save_npz(fout, A)
    print(f"Written to {fout}")
