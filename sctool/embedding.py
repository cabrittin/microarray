"""
@name: sctool.embedding.py                     
@description:                  
    Functions for low dimensional embedding

@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05              
"""

import sklearn.decomposition as skd
import numpy as np


def pca(sc,gene_flag=None,n_components=50,set_loadings=False,**kwargs):
    """
    gene_flag: subselect genes based on conditional flag, must alread be set in genes dataframe
    """
    X = sc.X.toarray() 
    sc.pca = skd.PCA(n_components=n_components,**kwargs)
    if gene_flag is not None:
        jdx = sc.genes.index[sc.genes[gene_flag] == 1].tolist()
        sc.pca.components = sc.pca.fit_transform(X[:,jdx])
    else:
        sc.pca.components = sc.pca.fit_transform(X)
    
    if set_loadings:
        sc.pca.loadings = sc.pca.components_.T*np.sqrt(sc.pca.explained_variance_)


