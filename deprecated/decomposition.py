"""
@name: decomposition.py                        
@description:                  
    Perform various decompositions on single cell object.
    The decomposition with be stored as SingleCell(object).D

@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05              
"""

import numpy as np
import sklearn.decomposition as skd
from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA
from sklearn.metrics import DistanceMetric
from scipy.sparse.linalg import svds
from sklearn.utils.extmath import svd_flip

def _sk_decomp(X,decomp,**kwargs):
    D = getattr(skd,decomp)(**kwargs)
    D.components = D.fit_transform(X)
    return D

def pca(sc,X=None,**kwargs): 
    if X is not None: 
        sc.D = _sk_decomp(X,'PCA',**kwargs)
    else:
        sc.D = _sk_decomp(sc.X,'PCA',**kwargs)

def factor_analysis(sc,**kwargs):
    sc.D = _sk_decomp(sc.X,'FactorAnalysis',**kwargs)

def nmds(sc,**kwargs):
    #similarities = euclidean_distances(sc.X)
    #dist = DistanceMetric.get_metric('euclidean')
    dist = DistanceMetric.get_metric('manhattan')
    #dist = DistanceMetric.get_metric('braycurtis')
    X = dist.pairwise(sc.X)
    #X = X / np.max(X)
    #X = 1 - X
    n_components = 2 
    seed = np.random.RandomState(seed=3)
    mds = manifold.MDS(
        n_components=n_components,
        max_iter=3000,
        eps=1e-9,
        random_state=seed,
        dissimilarity="precomputed",
        n_jobs=4,
    )
    pos = mds.fit(X).embedding_
    #pos = mds.fit_transform(X)
    
    nmds = manifold.MDS(
    n_components=n_components,
    metric=False,
    max_iter=3000,
    eps=1e-12,
    dissimilarity="precomputed",
    random_state=seed,
    n_jobs=4,
    n_init=0,
    )

    npos = nmds.fit_transform(X, init=pos)
    
    pos *= np.sqrt((sc.X ** 2).sum()) / np.sqrt((pos ** 2).sum()) 
    npos *= np.sqrt((sc.X ** 2).sum()) / np.sqrt((npos ** 2).sum())

    nmds.components = npos
    mds.components = pos
    #sc.D = nmds
    sc.D = mds

def spectral(sc,**kwargs):
    n_neighbors = 5
    sc.D = manifold.SpectralEmbedding(n_components=4, n_neighbors=n_neighbors)
    sc.D.components = sc.D.fit_transform(sc.X)

def tsne(sc,X = None,**kwargs):
    #dist = DistanceMetric.get_metric('manhattan')
    #sc.X = dist.pairwise(sc.X)
    if X is not None:
        sc.D = manifold.TSNE(**kwargs)
        sc.D.components = sc.D.fit_transform(X)
    else: 
        sc.D = manifold.TSNE(**kwargs)
        sc.D.components = sc.D.fit_transform(sc.X)

def lle(sc,X = None,**kwargs):
    #dist = DistanceMetric.get_metric('manhattan')
    #sc.X = dist.pairwise(sc.X)
    if X is not None:
        sc.D = manifold.LocallyLinearEmbedding(**kwargs)
        sc.D.components = sc.D.fit_transform(X)
    else: 
        sc.D = manifold.TSNE(**kwargs)
        sc.D.components = sc.D.fit_transform(sc.X)


def pcoa(sc):
    dist = DistanceMetric.get_metric('manhattan')
    X = dist.pairwise(sc.X)
    #X = X / np.max(X)
    #X = 1 - X
    n = X.shape[0]
    C = np.eye(n) - np.ones((n,n)) / float(n)
    B = -0.5*np.dot(C,np.dot(X,C))
    
    solver='arpack'
    random_init = np.random.rand(np.min(sc.X.shape))
    npcs = 3

    u, s, v = svds(sc.X, solver=solver, k=npcs, v0=random_init)
    u, v = svd_flip(u, v)
    idx = np.argsort(-s)
    sc.components = (u * s)[:, idx]
    
    

