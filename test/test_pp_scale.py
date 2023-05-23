"""
@name:                         
@description:                  


@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05              
"""

import pytest

import scipy.sparse as sp
import numpy as np

import sctool.pp as pp


class SC:
    def __init__(self):
        """ 
        Test matrix taken from 

        https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.html#scipy.sparse.coo_array
        
        # Constructing a array using ijv format
        >> row  = np.array([0, 3, 1, 0])
        >> col  = np.array([0, 3, 1, 2])
        >> data = np.array([4, 5, 7, 9])
        >> coo_array((data, (row, col)), shape=(4, 4)).toarray()
            array([[4, 0, 9, 0],
                   [0, 7, 0, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 5]])
        """
        row  = np.array([0,0,1,3])
        col  = np.array([0,2,1,3])
        data = np.array([4, 9, 7, 5])
        self.X = sp.coo_matrix((data, (row, col)), shape=(4, 4))


def test_cells_by_vector():
    sc = SC()
    vec = np.array([13,7,1,5])
    pp.scale.cells_by_vector(sc,x=vec)
    assert(np.array_equal(sc.X.data,np.array([4./13,9./13,1.,1.])))

def test_genes_by_vector():
    sc = SC()
    vec = np.array([8.,7.,9.,20.])
    pp.scale.genes_by_vector(sc,x=vec)
    assert(np.array_equal(sc.X.data,np.array([0.5,1.,1.,0.25])))

