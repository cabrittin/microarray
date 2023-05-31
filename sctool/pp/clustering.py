"""
@name: sctool.pp.clustering.py                       
@description:                  
    Module for doing clustering

@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05              
"""

import numpy as np
from tqdm import tqdm
import pandas as pd

def multi_res_clustering(func):
    def inner(sc,kvals,rvals,label='comm_k_',**kwargs):
        sc.meta['cluster_res'] = []
        for k in kvals:
            for r in tqdm(rvals,desc=f"KNN: {k} resoutions completed"): 
                communities = func(sc,k,resolution_parameter=r,**kwargs)
                sc.cells[f'{label}_k{k}_r{r}'] = pd.Categorical(communities)
                sc.meta['cluster_res'].append(f'{label}{k}_{r}')
        return None
    return inner

def multi_res_compare(sc,**kwargs):
    from sklearn.metrics.cluster import adjusted_rand_score
    keys = sc.meta['cluster_res']
    N = len(keys)
    m = len(sc.cells)
    Z = np.zeros((N,N))
    for i in range(N):
        x = sc.cells[[keys[i]]].to_numpy().reshape(m)
        for j in range(i,N):
            y = sc.cells[[keys[j]]].to_numpy().reshape(m)
            Z[i,j] = adjusted_rand_score(x,y)
    return Z

@multi_res_clustering
def phenograph(sc,k,**kwargs):
    import phenograph as _phenograph
    communities, graph, Q = _phenograph.cluster(sc.pca.components,k=k,**kwargs)
    return communities
