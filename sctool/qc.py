"""
@name: explore.py                
@description:                  
   Convenience functions for basic QC analysis  

@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05              
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skmisc.loess import loess

from sctool import query
from toolbox.stats.basic import ecdf


def ecdf_cell_total_counts(sc,genes=None,ax=None,show=True,**kwargs):
    x = query.cell_total_counts(sc,genes=genes)
    x = np.log(x)
    plot_ecdf(x,ax=ax,xlabel='log(Total UMI count)',**kwargs) 
    

def plot_ecdf(data,ax=None,xlabel=None,reverse=False,plot_params=None,**kwargs):
    ylabel = ['ECDF','1-ECDF'][int(reverse)]
    if ax is None: fig,ax = plt.subplots(1,1,figsize=(5,5))
    x,y = ecdf(data,reverse=reverse)
    ax.plot(x,y,**kwargs)
    __plot_labels__(ax,xlabel,ylabel,plot_params)
   

def plot_loss_fit(x,y,ax=None,xlabel=None,ylabel=None,**kwargs):
    if ax is None: fig,ax = plt.subplots(1,1,figsize=(5,5))


def cossim(x,y):
    x = x.reshape(len(x))
    y = y.reshape(len(y))
    return np.dot(x,y) / float(np.sqrt(x.sum()) * np.sqrt(y.sum()))

def hvg_batch_vs_all(sc):
    hall = sc.genes[['hvg']].to_numpy()
    hall = hall.reshape(len(hall))
    hbatch = sc.genes[sc.meta['batch_hvg']].to_numpy()
    hbatch = hbatch.sum(1)
    hbatch[hbatch > 0] = 1
    return cossim(hall,hbatch)

def hvg_batch_vs_all_matrix(sc):
    bkeys = sc.meta['batch_hvg'] + ['hvg']
    N = len(bkeys) 
    Z = np.zeros((len(bkeys),len(bkeys)))
    for i in range(N):
        x = sc.genes[[bkeys[i]]].to_numpy() 
        for j in range(i,N):
            y = sc.genes[[bkeys[j]]].to_numpy()
            Z[i,j] = cossim(x,y)
    return Z

def plot_hvg_batch_vs_all(sc):
    tot_sim = hvg_batch_vs_all(sc)
    Z = hvg_batch_vs_all_matrix(sc)
    bkeys = sc.meta['batch_hvg'] + ['hvg']
    

    fig,ax = plt.subplots(1,1,figsize=(10,10))
    sns.heatmap(Z,xticklabels=bkeys,yticklabels=bkeys)
    ax.set_title('Merged similarity: %1.2f' %tot_sim)
    plt.show()

def __plot_labels__(ax,xlabel,ylabel,params):
    ax.set_xlabel(xlabel,fontsize=12)
    ax.set_ylabel(ylabel,fontsize=12)
    ax.tick_params(axis='x',labelsize=10)
    ax.tick_params(axis='y',labelsize=10)

