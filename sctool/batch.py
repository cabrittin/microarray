"""
@name: sctool.batch.py                     
@description:                  
    Module for splitting SingleCell class into batches

@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05              
"""

import numpy as np

def split(sc,label='batches'):
    assert not hasattr(sc,label), "SingleCell instance already has {label} attribute, consider using another attribute label" 
    bkey = sc.cfg['keys']['meta_key_batch']
    batches = list(set(sc.cells[bkey].tolist()))
    for b in batches:
        bflag = np.zeros(len(sc.cells),dtype=int)
        idx = sc.cells.index[sc.cells[bkey] == b].tolist()
        bflag[idx] = 1
        sc.cells[b] = bflag
    setattr(sc,label,batches)
