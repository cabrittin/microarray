"""
@name: pls_test.py
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

from sklearn.cross_decomposition import PLSRegression as PLS
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,balanced_accuracy_score

from cebraindev.cetorch.cedata import CeData
from toolbox import plots

def parameterize_theta(Y):
    rnorm = np.linalg.norm(Y,axis=1)
    Z = Y / rnorm[:,None]
    x,y = (Z[:,0],Z[:,1])
    #r = np.sqrt(x**2 + y**2)
    #theta = np.arccos(x / r)
    theta = np.arccos(x)
    theta[y<0] = -theta[y<0]
    return theta 

def shift_theta(theta,shift=0):
    theta += shift
    mod = np.where(theta > np.pi)
    theta[mod] = theta[mod] - 2*np.pi

def encode_theta(theta,bounds=[-0.5*np.pi,0,0.5*np.pi]):
    cur_id = len(bounds) + 1
    enc_theta = np.ones(theta.shape,dtype=int)*cur_id
    bounds = sorted(bounds,reverse=True)
    for (j,b) in enumerate(bounds): 
        cur_id -= 1
        enc_theta[np.where(theta < b)] = cur_id
    #plot_unit_circle(theta,s=20,c=enc_theta,cmap='Set1')
    return enc_theta

def encode_onehot(labels):
    udx = sorted(np.unique(labels))
    one_hot = np.zeros((len(labels),len(udx)),dtype=int)
    for (i,u) in enumerate(udx):
        one_hot[np.where(labels==u)[0],i] = 1
    return one_hot

def load_data(D):
    theta_s = parameterize_theta(D.target)
    shift_theta(theta_s,0.35*np.pi)
    label_s = encode_theta(theta_s)
    ohs = encode_onehot(label_s)
    #plot_unit_circle(theta_s,s=20,c=label_s,cmap='Set1')    
    
    theta_l = parameterize_theta(D.lineage)
    label_l = encode_theta(theta_l)
    ohl = encode_onehot(label_l)
    
    labels = list(zip(label_l,label_s))
    return labels,ohs,ohl

def run_pls(params):
    D = CeData(params.data,lineage=True)
    labels,Y,ohl = load_data(D)
    
    X = D.input

    pls = PLS(n_components=2,scale=True)
    
    labels = D.get_labels()
    fig,_ax = plt.subplots(2,2,figsize=(10,10))
    for (i,ax) in enumerate(_ax.flatten()):
        pls.fit(X,Y[:,i])
        Xc, Yc = pls.transform(D.input, Y[:,i])
        ax.scatter(Xc[:,0],Xc[:,1],s=20,c=Y[:,i],cmap='tab10')
        ax.set_xlabel('latent_1',fontsize=8)
        ax.set_ylabel('latent_2',fontsize=8)
    
    accuracy = []
    cval = KFold(n_splits=3, shuffle=True, random_state=7)
    for i in range(Y.shape[1]): 
        accuracy = []
        for train, test in cval.split(X):
            y_pred = pls_da(X[train,:], Y[train,i], X[test,:])
            #accuracy.append(accuracy_score(Y[test,i], y_pred))
            #accuracy.append(f1_score(Y[test,i], y_pred))
            #accuracy.append(precision_score(Y[test,i], y_pred))
            #accuracy.append(recall_score(Y[test,i], y_pred))
            accuracy.append(balanced_accuracy_score(Y[test,i], y_pred))

        print(f"Average accuracy on 5 splits group {i}: ", np.array(accuracy).mean())
    
    plt.show()

def pls_da(X_train,y_train, X_test):
    # Define the PLS object for binary classification
    plsda = PLS(n_components=2)
    # Fit the training set
    plsda.fit(X_train, y_train)
    # Binary prediction on the test set, done with thresholding
    binary_prediction = (plsda.predict(X_test)[:,0] > 0.5).astype('uint8')
    return binary_prediction

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
