"""
@name:
@description:


@author: Christopher Brittin
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from tqdm import tqdm
from sklearn.covariance import OAS
#from sklearn.naive_bayes import GaussianNB as CLF
from sklearn.naive_bayes import MultinomialNB as CLF
#from sklearn.naive_bayes import ComplementNB as CLF
#from sklearn.naive_bayes import BernoulliNB as CLF
#from sklearn.linear_model import LogisticRegression as CLF
#from sklearn.neighbors import KNeighborsClassifier as CLF
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,balanced_accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import matthews_corrcoef
from scipy.linalg import svd
from sklearn.utils.extmath import svd_flip
from scipy.spatial import distance

from cebraindev.cetorch.cedata import CeData
from toolbox import plots
from toolbox.stats.basic import ecdf
#from toolbox.ml.dr import classic_mds as mds
from toolbox.ml.dr import test_mds as mds

class dirlist:
    def __init__(self,size):
        self.l = -1*np.ones(size,dtype=int)

    def __call__(self,j,i):
        assert j > i, print("j must be greater than i") 
        while self.l[i] != -1:
            i = self.l[i]
        self.l[j] = i

    def set_index(self):
        self.index = np.where(self.l<0)[0]

def shrink_covariance(data):
    model = OAS()
    model.fit(data)
    return model.covariance_
    

def plot_shrinkage(params):
    D = CeData(params.data)
    Y = D.target.astype(int)
    cov = np.cov(D.input.T)

    cov_ = shrink_covariance(D.input)

    data = [cov,cov_]
    fig,ax = plt.subplots(1,2,figsize=(10,5))
    for (i,c) in enumerate(data):
        print(c.shape,c.min(),c.max())
        ax[i].imshow(c,vmin=-0.2,vmax=0.2,cmap='PRGn')
        ax[i].set_xticks([])
        ax[i].set_yticks([])
       
    iu = np.triu_indices(cov.shape[0],k=1)
    fig,ax = plt.subplots(1,2,figsize=(10,5))
    for (i,c) in enumerate(data):
        x,y = ecdf(abs(c[iu]),reverse=True)
        ax[i].plot(x,y)
        
    plt.show()

def filter_cov(alpha,cov,beta=None):
    n = cov.shape[0]
    dl = dirlist(n)
    for i in range(n):
        for j in range(i+1,n):
            if abs(cov[i,j]) >= alpha: dl(j,i)
    dl.set_index() 
    return dl

def run_classifier(params):
    alpha = 0.25
    D = CeData(params.data)
    cov = shrink_covariance(D.input)
    
    dl = filter_cov(alpha,cov)
    X = D.input[:,dl.index]
    #X = D.input 
    print(X.shape)
    Y = D.target
    print(Y)
    cval = KFold(n_splits=5, shuffle=True, random_state=42)
    accuracy = []
    for idx,(train, test) in enumerate(cval.split(X)):
        clf = CLF(alpha=1)
        clf.fit(X[train,:], Y[train])
        #score = accuracy_score(Y[test,i],clf.predict(X[test,:])) 
        #score = balanced_accuracy_score(Y[test,i],clf.predict(X[test,:])) 
        #score = f1_score(Y[test,i],clf.predict(X[test,:])) 
        #score = cohen_kappa_score(Y[test],clf.predict(X[test,:])) 
        score = matthews_corrcoef(Y[test],clf.predict(X[test,:])) 
        accuracy.append(score)
    print(f"Average accuracy for class: ", np.array(accuracy).mean())
    
    clf = CLF(alpha=1)
    clf.fit(X,Y)
    print(clf.feature_log_prob_.shape,np.exp(np.max(clf.feature_log_prob_,axis=1)))

def pca(X,shrink_cov=False) -> np.ndarray:
    #Center data
    mu = np.mean(X,axis=0)
    X -= mu
    if shrink_cov:
        print('here')
        cov = shrink_covariance(X.T)
    else:
        cov = np.cov(X)
    
    n = cov.shape[0]
    print(n)
    U,S,V = svd(X,full_matrices=False)
    U, V = svd_flip(U, V)
    eigs = S*S / np.sqrt(n-1)
    return eigs, (U * S)/np.sqrt(n-1)


def test_dist(params):
    D = CeData(params.data)
    Z = D.input
    N = Z.shape[0]
    print(N)

    Z = distance.cdist(Z,Z,'braycurtis')
    X,eigs = mds(Z)   
    
    scaled_eigs = eigs / np.sqrt(N) 
    fig,ax = plt.subplots(1,1,figsize=(10,10))
    ax.hist(scaled_eigs,bins=50,density=True)
    ax.set_ylabel('Density')
    ax.set_xlabel('Scaled eigenvalues')
    #ax.set_xlim([0,1])
    
    fig,ax = plt.subplots(1,1,figsize=(10,10))
    ax.scatter(X[:,1],X[:,2],s=20,c=D.get_labels())
    plt.show()

 


def test_rmt(params):
    D = CeData(params.data)
    N = D.input.shape[0]
    X = D.input

    shrink_cov=True
    mu = np.mean(X,axis=0)
    X -= mu
    if shrink_cov:
        print('here')
        cov = shrink_covariance(X.T)
    else:
        cov = np.cov(X)
    
    eigs,_ = np.linalg.eigh(cov)
    scaled_eigs = eigs / np.sqrt(N)

    fig,ax = plt.subplots(1,1,figsize=(10,10))
    ax.hist(scaled_eigs,bins=50,density=True)
    ax.set_ylabel('Density')
    ax.set_xlabel('Scaled eigenvalues')
    plt.show() 


def compare_eigs(params):
    D = CeData(params.data)
    N = D.input.shape[0]
    
    shrink_cov=False
    eigs,evec =pca(D.input,shrink_cov=shrink_cov)
    scaled_eigs = eigs / np.sqrt(N) 
    
    print(eigs)
    fig,ax = plt.subplots(1,1,figsize=(10,10))
    ax.hist(scaled_eigs,bins=50,density=True)
    ax.set_ylabel('Density')
    ax.set_xlabel('Scaled eigenvalues')
    #ax.set_xlim([0,1])
    
    
    fig,ax = plt.subplots(1,1,figsize=(10,10))
    ax.scatter(evec[:,0],evec[:,1],s=20,c=D.get_labels(),cmap='Set1')
    plt.show()

def run_pca(params):
    from sklearn.decomposition import PCA
    D = CeData(params.data)
    N = D.input.shape[0]

    pca = PCA()
    pca.fit(D.input)
    X = pca.transform(D.input)

    scaled_eigs = (pca.singular_values_**2) / np.sqrt(N-1) / np.sqrt(N)
    fig,ax = plt.subplots(1,1,figsize=(10,10))
    ax.hist(scaled_eigs,bins=50,density=True)
    ax.set_ylabel('Density')
    ax.set_xlabel('Scaled eigenvalues')
    #ax.set_xlim([0,1])
    
    fig,ax = plt.subplots(1,1,figsize=(10,10))
    ax.scatter(X[:,2],X[:,3],s=20,c=D.get_labels(),cmap='Set1')
    
    plt.show()
 
def run_sparse_pca(params):
    from sklearn.decomposition import SparsePCA as PCA
    D = CeData(params.data)
    N = D.input.shape[0]

    pca = PCA(n_components=2,n_jobs=5)
    pca.fit(D.input)
    X = pca.transform(D.input) 
    print(X.shape)
    fig,ax = plt.subplots(1,1,figsize=(10,10))
    ax.scatter(X[:,0],X[:,1],s=20,c=D.get_labels(),cmap='Set1')
    
    plt.show()
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument('data',
                        action = 'store',
                        help = 'Path to npz datafile')
    
    parser.add_argument('mode',
                        action = 'store',
                        help = 'Mode to run')

    params = parser.parse_args()
    
    eval(params.mode + '(params)')
