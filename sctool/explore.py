"""
@name: explore.py
@description:
    Convenient functions for running basic stats on a count matrix
    

@author: Christopher Brittin
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05
"""

import numpy as np
import inspect
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns
import mplcursors

#from toolbox.plots import plot_multi_pages
import toolbox.matrix_properties as mp
from toolbox.stats.basic import ecdf
from toolbox.scale import standardize
#from sctool.feature_selection import hvg_seurat3
from sctool import query
#import sctool.decomposition as decomp



def gene_subset_vs_all_count(sc,gene_list,ax=None,log_scale=True,**kwargs):
    if ax is None: fig,ax = plt.subplots(1,1,figsize=(10,10)) 
    x = query.cell_total_counts(sc)
    y = query.cell_total_counts(sc,genes=gene_list)
    xlabel,ylabel = 'all_gene_count','sub_gene_count'  
    if log_scale:
        x = np.log(x+1)
        y = np.log(y+1)
        xlabel,ylabel = 'log(all_gene_count)','log(sub_gene_count)'  
    
    ax.scatter(x,y,s=5,c='#9f9f9f')
    __plot_labels__(ax,xlabel,ylabel,sc.cfg['plot_params'])

def residual_filter(sc,x,y,hue,ax=None,**kwargs):
    if ax is None: fig,ax = plt.subplots(1,1,figsize=(10,10)) 
    _x = np.linspace(sc.cells[x].min(),sc.cells[x].max(),100)
    _y = sc.residual_filter.slope * _x + sc.residual_filter.intercept
    cdict = {0:'r',1:'#9f9f9f'} 
    sns.scatterplot(sc.cells,ax=ax,x=x,y=y,hue=hue,palette=cdict,s=10)
    ax.plot(_x,_y,c='k')

def cumulative_variance(sc,ax=None,**kwarg):
    x = np.arange(sc.pca.n_components_) + 1
    if ax is None: fig,ax = plt.subplots(1,1,figsize=(5,5)) 
    ax.plot(x, np.cumsum(sc.pca.explained_variance_ratio_), 'o-',markersize=4,c='k')
    __plot_labels__(ax,'Principle component','Cumulative variance explained',sc.cfg['plot_params'])

def hover_gene(sc,idx):
    return sc.genes[sc.gene.key][idx]

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



""" Deprecated below here """

def gene_mean(sc,ax=None,**kwargs):
    if not sc.log_scale: sc.X.data = np.log(sc.X.data)
    data = mp.axis_mean(sc.X,axis=0,skip_zeros=sc.params.nz)
    xlabel = f"log({inspect.stack()[0][3]})"  
    if sc.params.nz: xlabel = xlabel.replace(')','_nz)')
    plot_ecdf(data,ax=ax,xlabel=xlabel,
            reverse=True,plot_params=sc.plot_params,**kwargs)

def gene_dispersion(sc,ax=None,**kwargs):
    #if not sc.log_scale: sc.X.data = np.log(sc.X.data)
    mu,var = mp.axis_mean_var(sc.X,axis=0,skip_zeros=sc.params.nz) 
    data = np.log2(np.sqrt(var) / mu)
    xlabel = ['log2(dispersion)','log2(dispersion_nz)'][int(sc.params.nz)] 
    plot_ecdf(data,ax=ax,xlabel=xlabel,
            reverse=True,plot_params=sc.plot_params,**kwargs)


def gene_cell_count_vs_dispersion(sc,ax=None,**kwargs):
    ylabel = ['log2(dispersion)','log2(dispersion_nz)'][int(sc.params.nz)]
    sc.X.data = np.log(sc.X.data) 
    mu,var = mp.axis_mean_var(sc.X,axis=0,skip_zeros=sc.params.nz) 
    disp = np.log2(np.sqrt(var) / mu)
    elements = mp.axis_elements(sc.X,axis=0)
    if ax is None: fig,ax = plt.subplots(1,1,figsize=(5,5))
    ax.scatter(elements,disp,s=5)
    xlabel = '# cells with gene count'
    __plot_labels__(ax,xlabel,ylabel,sc.plot_params)


def cell_count(sc,**kwargs):
    @plot_multi_pages()
    def run_plot(sc,index=None,ax=None,**kwargs):
        if kwargs['nz']:
            nz = np.where(sc.X[index,:] > 0)[0]
            x = sc.X[index,nz]
        else:
            x = sc.X[index,:]
        label = sc.cells['cell_id'][index]
        plot_cell_ecdf(ax,x,label,reverse=kwargs['reverse'])
    
    sc.X = np.log(sc.X.toarray() + 1)
    reverse = sc.params.nz
    ylabel = ['ECDF','1-ECDF'][reverse] 
    xlabel = ['log(gene_counts)','log(gene_counts_nz)'][sc.params.nz] 
    run_plot(sc,ylabel=ylabel,xlabel=xlabel,fout=sc.params.fout,reverse=reverse,nz=sc.params.nz)
    sc.fig_saved = True

def cell_genes_in_other_cells(sc,**kwargs):
    @plot_multi_pages()
    def run_plot(sc,index=None,ax=None,**kwargs):
        nz = np.where(sc.X[index,:] > 0)[0]
        x = kwargs['gene_num_cells'][nz] 
        label = sc.cells['cell_id'][index]
        plot_cell_ecdf(ax,x,label,reverse=kwargs['reverse'])
    
    gene_num_cells = mp.axis_elements(sc.X,axis=0)
    sc.X = np.log(sc.X.toarray() + 1)
    reverse = False
    ylabel = ['ECDF','1-ECDF'][reverse] 
    xlabel = '#_cells_expressing_gene' 
    run_plot(sc,ylabel=ylabel,xlabel=xlabel,fout=sc.params.fout,gene_num_cells=gene_num_cells,reverse=reverse)
    sc.fig_saved = True

def cell_zscore(sc,**kwargs):
    @plot_multi_pages()
    def run_plot(sc,index=None,ax=None,**kwargs):
        nz = np.where(sc.X[index,:] > 0)[0]
        x = sc.X[index,nz]
        zscore = (x - kwargs['mu'][nz]) / kwargs['std'][nz]

        label = sc.cells['cell_id'][index]
        plot_cell_ecdf(ax,zscore,label,reverse=kwargs['reverse'])
    
    sc.X.data = np.log2(sc.X.data)
    mu,var = mp.axis_mean_var(sc.X,axis=0)
    var = np.sqrt(var)
    sc.X = sc.X.toarray()
    reverse = True
    ylabel = ['ECDF','1-ECDF'][reverse] 
    xlabel = 'z-score' 
    run_plot(sc,sc.X.shape[0],ylabel=ylabel,xlabel=xlabel,fout=sc.params.fout,mu=mu,std=var,reverse=reverse)
    sc.fig_saved = True

def cell_elements_vs_counts_density(sc,**kwargs):
    _cell_elements_vs_counts(sc,plot_cell_density)

def cell_elements_vs_counts_scatter(sc,**kwargs):
    _cell_elements_vs_counts(sc,plot_cell_scatter)

def _cell_elements_vs_counts(sc,callback):
    @plot_multi_pages()
    def run_plot(sc,index=None,ax=None,**kwargs):
        nz = np.where(sc.X[index,:] > 0)[0]
        y = sc.X[index,nz]
        label = sc.cells['cell_id'][index]
        kwargs['callback'](ax,kwargs['x'][nz],y,label)
    
    elements = mp.axis_elements(sc.X,axis=0)
    sc.X = np.log(sc.X.toarray() + 1)
    ylabel = 'gene_counts'
    xlabel = '#_cells_with_gene'
    run_plot(sc,sc.X.shape[0],ylabel=ylabel,xlabel=xlabel,fout=sc.params.fout,x=elements,callback=callback)
    sc.fig_saved = True

def plot_cell_scatter(ax,x,y,label):
    ax.scatter(x,y,s=2)
    label_pos = (0.5,0.8)
    ax.text(label_pos[0],label_pos[1],f'{label}',transform=ax.transAxes,fontsize=8,color='w')

def plot_cell_density(ax,x,y,label):
    nbins_x = 80
    nbins_y = 10 
    data = np.vstack((x,y))
    k = scipy.stats.gaussian_kde(data)
    xi, yi = np.mgrid[x.min():x.max():nbins_x*1j, y.min():y.max():nbins_y*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    ax.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.plasma)
    label_pos = (0.05,0.8)
    ax.text(label_pos[0],label_pos[1],f'{label}',transform=ax.transAxes,fontsize=8,color='w')



def plot_cell_ecdf(ax,data,label,reverse=False):
    x,y = ecdf(data,reverse=reverse)
    ax.plot(x,y)
    label_pos = [(0.05,0.9),(0.85,0.9)][int(reverse)]
    count_pos = [(0.7,0.05),(0.7,0.8)][int(reverse)]
    ax.text(label_pos[0],label_pos[1],f'{label}',transform=ax.transAxes,fontsize=8)
    ax.text(count_pos[0],count_pos[1],f'$n=${len(x)}',transform=ax.transAxes,fontsize=8)

def plot_ecdf(data,ax=None,xlabel=None,reverse=False,plot_params=None):
    ylabel = ['ECDF','1-ECDF'][int(reverse)]
    if ax is None: fig,ax = plt.subplots(1,1,figsize=(10,10))
    x,y = ecdf(data,reverse=reverse)
    ax.plot(x,y)
    ax.set_ylim([0,1])
    __plot_labels__(ax,xlabel,ylabel,plot_params)
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

