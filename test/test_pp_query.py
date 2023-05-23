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
        row  = np.array([0,0,1,3,2])
        col  = np.array([0,2,1,3,2])
        data = np.array([4, 9, 7, 5,1])
        self.X = sp.coo_matrix((data, (row, col)), shape=(4, 4))

def test_total_cell_counts():
    from scipy.stats import gmean
    sc = SC()
    exp = np.array([13,7,1,5])
    assert(np.array_equal(exp,pp.query.cell_total_counts(sc))) 
    
    exp = np.array([13,7])
    assert(np.array_equal(exp,pp.query.cell_total_counts(sc,cells=[0,1]))) 



def test_size_factor():
    from scipy.stats import gmean
    sc = SC()
    sm = np.array([13,7,1,5])
    gm = gmean(sm)
    exp = sm / gm 
    t = pp.query.size_factor(sc)
    assert(np.array_equal(exp,pp.query.size_factor(sc))) 

    
