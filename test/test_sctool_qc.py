"""
@name: test_sctool_qc.py                       
@description:                  
    Unit tests for sctool.qc.py

@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05              
"""

import pytest
import numpy as np

from sctool import util,qc
import pipelines.packer as pipe

CONFIG = 'config.ini'
DATASET = 'packer2019'
CFG = util.checkout(CONFIG,DATASET)


def test_hvg_vs_all():
    sc = util.load_sc(CFG,load_light=True)
    N = len(sc.genes)
    g = np.zeros(N)
    g[:1000] = 1
    sc.genes['qc_hvg'] = g
    sc.meta['batch_hvg'] = ['b1_hvg','b2_hvg']
    
    g = np.zeros(N)
    g[:500] = 1
    sc.genes['b1_hvg'] = g
    
    g = np.zeros(N)
    g[600:1100] = 1
    sc.genes['b2_hvg'] = g
    
    assert(qc.hvg_batch_vs_all(sc) == 0.9)
   
    g = np.zeros(N)
    g[400:900] = 1
    sc.genes['b2_hvg'] = g
    exp = 900. / (np.sqrt(1000)*np.sqrt(900)) 
    assert(qc.hvg_batch_vs_all(sc) == exp )


