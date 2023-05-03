"""
@name: sctool.pp.plot                      
@description:                  
    Module for making quality control plots

@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05              
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import mplcursors

import toolbox.matrix_properties as mp
from toolbox.stats.basic import ecdf
from sctool.pp import hvg,clustering

def pp_ecdf(func):
    def inner(sc,ax=None,reverse=False,log_scale=True,**kwargs):
        xlabel, data = func(sc,**kwargs)

        if log_scale:
            data = np.log2(data+1)
            xlabel = f"log2({xlabel})"
        
        ax = plot_ecdf(data,ax=ax,xlabel=xlabel,
                reverse=reverse,plot_params=sc.cfg['plot_params'],**kwargs)

        return ax
    return inner

@pp_ecdf
def num_cells_with_gene(sc,**kwargs):
    return 'num_cells_with_gene', mp.axis_elements(sc.X,axis=0) 

@pp_ecdf
def num_genes_in_cell(sc,**kwargs):
    return 'num_genes_in_cell', mp.axis_elements(sc.X,axis=1) 

@pp_ecdf
def total_gene_counts(sc,**kwargs):
    return 'total_gene_counts', mp.axis_sum(sc.X,axis=0) 

@pp_ecdf
def total_counts_per_cell(sc,**kwargs):
    return 'total_counts_per_cell', mp.axis_sum(sc.X,axis=1) 

@pp_ecdf
def meta_gene_value(sc,label=None,**kwargs):
    return label, sc.genes[label].to_numpy()

@pp_ecdf
def meta_cell_value(sc,label=None,**kwargs):
    return label, sc.cells[label].to_numpy()


def pp_hvg(func):
    def inner(sc,label='hvg',ax=None,**kwargs):
        if ax is None: fig,ax = plt.subplots(1,1,figsize=(10,10))
        xlabel,ylabel,x,y = func(sc.X,**kwargs) 
        sc.genes[xlabel] = x
        sc.genes[ylabel] = y
        _x = np.linspace(sc.hvg_model.inputs.x.min(),sc.hvg_model.inputs.x.max(),100)
        _y = sc.hvg_model.predict(_x).values
        cdict = {1:'r',0:'#9f9f9f'} 
        sns.scatterplot(sc.genes,x=xlabel,y=ylabel,hue=label,palette=cdict,ax=ax,s=5)
        ax.plot(_x,_y,c='k')
        __plot_labels__(ax,xlabel,ylabel,sc.cfg['plot_params'])
        
    return inner

@pp_hvg
def hvg_mean_var(X,**kwargs):
    xlabel,ylabel = 'log10(mean)','log10(var)'
    eps = 1e-5 
    mean,var = mp.axis_mean_var(X,axis=0,skip_zeros=False) 
    return xlabel,ylabel,np.log10(mean+eps),np.log10(var+eps)

@pp_hvg
def hvg_poisson_dispersion(X,**kwargs):
    from sctool.pp.hvg import _poisson_dispersion 
    xlabel,ylabel = 'log2(mean)','log2(cv2)'
    x,y = _poisson_dispersion(X)
    return xlabel,ylabel,x,y

@pp_hvg
def hvg_poisson_zero_count(X,**kwargs):
    from sctool.pp.hvg import _poisson_zero_count 
    xlabel,ylabel = 'log2(mean)','# zero UMI'
    x,y = _poisson_zero_count(X) 
    return xlabel,ylabel,x,y



def hvg_batch_vs_all(sc,ax=None):
    if ax is None: fig,ax = plt.subplots(1,1,figsize=(10,10))
    tot_sim = hvg.merge_batch_vs_all(sc)
    Z = hvg.batch_vs_all_matrix(sc)
    bkeys = sc.meta['batch_hvg'] + ['hvg']
    sns.heatmap(Z,xticklabels=bkeys,yticklabels=bkeys)
    ax.set_title('Merged similarity: %1.2f' %tot_sim)


def scree_plot(sc,ax=None,**kwarg):
    x = np.arange(sc.pca.n_components_) + 1
    if ax is None: fig,ax = plt.subplots(1,1,figsize=(5,5)) 
    ax.plot(x, sc.pca.explained_variance_ratio_, 'o-',markersize=4,c='k')
    __plot_labels__(ax,'Principle component','Variance explained',sc.cfg['plot_params'])

def pca_loadings(sc,ax=None,dim=[0,1],**kwargs):
    x = np.arange(sc.pca.n_components_) + 1
    if ax is None: fig,ax = plt.subplots(1,1,figsize=(10,10)) 
    ax.scatter(sc.pca.loadings[:,dim[0]],sc.pca.loadings[:,dim[1]],c='#9f9f9f',s=5)
    __plot_labels__(ax,f'PC {dim[0]} loading',f'PC {dim[1]} loading',sc.cfg['plot_params'])
    cursor = mplcursors.cursor(hover=True)
    cursor.connect(
        "add", lambda sel: sel.annotation.set_text(
            f"Gene: {sc.genes[sc.gene_key][sel.target.index]}")
    )

def multi_res_clusters(sc,ax=None):
    if ax is None: fig,ax = plt.subplots(1,1,figsize=(10,10))
    Z = clustering.multi_res_compare(sc)
    bkeys = sc.meta['cluster_res']
    sns.heatmap(Z,xticklabels=bkeys,yticklabels=bkeys)


def plot_ecdf(data,ax=None,xlabel=None,reverse=False,plot_params=None,callback=None,**kwargs):
    ylabel = ['ECDF','1-ECDF'][int(reverse)]
    if ax is None: fig,ax = plt.subplots(1,1,figsize=(10,10))
    x,y = ecdf(data,reverse=reverse)
    ax.plot(x,y,**kwargs)
    ax.set_ylim([0,1])
    __plot_labels__(ax,xlabel,ylabel,plot_params)
    if callback is not None: callback(ax,x,y,**kwargs)

    return ax
    
def __plot_labels__(ax,xlabel,ylabel,params):
    try:
        ax.set_xlabel(xlabel,fontsize=params['x_label_size'])
        ax.set_ylabel(ylabel,fontsize=params['y_label_size'])
        ax.tick_params(axis='x',labelsize=params['x_tick_size'])
        ax.tick_params(axis='y',labelsize=params['y_tick_size'])
    except:
        ax.set_xlabel(xlabel,fontsize=6)
        ax.set_ylabel(ylabel,fontsize=6)
        ax.tick_params(axis='x',labelsize=4)
        ax.tick_params(axis='y',labelsize=4)

def build_umap(X,**kwargs): 
    import umap.umap_ as umap
    reducer = umap.UMAP(**kwargs)
    reducer.fit(X)
    embedding = reducer.transform(X)
    return embedding,reducer

