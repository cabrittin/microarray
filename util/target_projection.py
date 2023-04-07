"""
@name:  target_projection.py                      
@description:                  
    Constructs transcriptomic space defined by target cells
    and then projects non-target cells into this space

@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05              
"""

import os
import sys
from configparser import ConfigParser,ExtendedInterpolation
import argparse
from inspect import getmembers,isfunction
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import pickle
from collections import defaultdict
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import scipy.cluster.hierarchy as sch

from sctool.sc_extension import SCAggregator as SingleCell
import sctool.decomposition as decomp
from sctool import scale
from pycsvparser import read
import toolbox.matrix_properties as mp
from sctool import explore
from toolbox.plots import viz_scatter,add_hover
from toolbox.stats.basic import ecdf
from toolbox.scale import standardize,minmax
from toolbox.diffusion import Diffusion2D
from toolbox.plots import plot_multi_pages
from pycsvparser import write


def load_sc(params):
    sc = SingleCell(params.fin)
    sc.params = params
    
    if sc.params.filter_level > -1:
        sc.filter_cells_gte('filter_level',params.filter_level)
        print('filter',sc.X.shape)
    scale.round_averaged_counts(sc)
    return sc 

def _hover_cells(df,idx):
    return f"Cells: {df['cell_id'][idx]}"

def hover_cells(cells,idx):
    return f"Cells: {cells[idx]}"

def mean_weighted_expression(sc,target_index=None):
    _A = sc.get_aggregator(edge_attr='id') 
    A = sc.get_aggregator(edge_attr='wnorm')
    A[_A < 16 ] = 0
    asum = A.sum(1)
    asum[asum==0] = 1
    A = A/ asum[:,None]
    E18 = sc.local_mean_aggregate(A)
    
    """
    A = sc.get_aggregator(edge_attr='wnorm') 
    A[_A > 6 ] = 0
    if target_index is not None:
        A = np.multiply(A,M)
    asum = A.sum(1)
    asum[asum==0] = 1
    A = A/ asum[:,None]
    E1 = sc.local_mean_aggregate(A)
    E = E18 - E1 
    """ 
    return E18

def gene_covariance(params):
    sc = load_sc(params)
    genes = read.into_list(params.genes)
    sc.filter_genes_isin(sc.gene_key,genes)
    print(sc.X.shape) 
    sc.load_aggregator()
    sc.X = np.log10(sc.X.toarray() + 1)
    K = sc.X.T
    print(K.shape)
    #K = np.cov(X)
    #K = (K - K.mean()) / K.std()
    K = minmax(K,axis=0)
    #d = sch.distance.pdist(K,metric='cityblock')
    #d = sch.distance.pdist(K,metric='braycurtis')
    d = sch.distance.pdist(K,metric='cosine')
    d = 1 - d
    y = sch.linkage(K, method='ward',optimal_ordering = True)
    ind = sch.fcluster(y, 6, 'distance')
    print(np.unique(ind))
    genes = [[g,ind[i]] for (i,g) in enumerate(sc.genes[sc.gene_key].tolist())]
    write.from_list('data/pioneer_embedding/clr_gene_groups.csv',genes)
    
    d = sch.distance.squareform(d)
    #d = 1 - (d / d.max())
    #d = (d - d.mean()) / d.std()
    print(d.shape)
    im = sns.clustermap(d,row_linkage=y,col_linkage=y,dendrogram_ratio=(.2, .2),
        #cbar_pos=(0.75,0.95,0.2,0.03),xticklabels=ind,yticklabels=[],figsize=(25,5),
        cbar_pos=(0.25,0.9,0.2,0.03),xticklabels=[],yticklabels=[],figsize=(5,5),
        #cmap='PRGn',#vmin=-3,vmax=3,
        #cmap='PRGn',#vmin=-3,vmax=3,
        cmap='viridis',vmin=0,vmax=1,
        cbar_kws={"orientation": "horizontal","ticks":[0,0.5,1.0],'shrink':0.5} )
    im.ax_row_dendrogram.set_visible(False)
    im.ax_heatmap.tick_params(axis='x',labelsize=6,rotation=0)
    plt.savefig('data/pioneer_embedding/clr_clusters.png',dpi=300)
    plt.show()


def build_target_space(params):
    sc = SingleCell(params.fin)
    sc.params = params
    
    if sc.params.filter_level > -1:
        sc.filter_cells_gte('filter_level',params.filter_level)
        print('filter',sc.X.shape)
    scale.round_averaged_counts(sc)
    
    genes = read.into_list(params.genes)
    sc.filter_genes_isin(sc.gene_key,genes)
    
    sc.load_aggregator()
    sc.index_cells()
    idx_cell = dict([(v,k) for (k,v) in sc.cell_idx.items()])
    print(sc.X.shape) 
    print(sc.cells) 
    cells = sc.cells[sc.cell_key].tolist()
    tcells = read.into_list(params.target_cells)
    ntcells = list(set(cells) - set(tcells))
    tdx = sc.cells.index[sc.cells[sc.cell_key].isin(tcells)].tolist()
    ntdx = sc.cells.index[sc.cells[sc.cell_key].isin(ntcells)].tolist()
    #print(tdx) 
    ord_tcells = [idx_cell[i] for i in tdx]
    ord_ntcells = [idx_cell[i] for i in ntdx]
    sc.X = np.log10(sc.X.toarray() + 1)
    E = mean_weighted_expression(sc,target_index=tdx) 

    Xt = sc.X[tdx,:]
    Xnt = E[ntdx,:]
    #print(ntdx) 
    #sc.filter_cells_isin(sc.cell_key,tcells)
    
    #decomp.pca(sc,X=Xt,n_components=2)
    decomp.lle(sc,X=Xt,n_components=2)
    _Xnt = sc.D.transform(Xnt)
    np.random.seed(42)
    _Xnt = _Xnt + np.random.uniform(low=0.0,high=0.05,size=_Xnt.shape)
    
    ## EC group colormap
    #cmaplist = ['#9301E7','#E7A401','#5E7FF1','#FC0000','#1FFF00','#9b9b9b']
    cmaplist = ['#a9c9ff','#fff3a0','#ff9a9a','#f8b3ff','#e4e4e4','#bdffb5','#111111']
    t_cmap = mpl.colors.LinearSegmentedColormap.from_list(
                        'Custom cmap', cmaplist, len(cmaplist))
    t_bounds = np.linspace(0, len(cmaplist), len(cmaplist)+1)
    t_norm = mpl.colors.BoundaryNorm(t_bounds, t_cmap.N)    
    ec = sc.cells['ec_group'].to_numpy()
    t_col = [t_cmap(j) for j in ec[tdx]]
 
    ## Spatial Domain colormap
    cmaplist = ['#9301E7','#E7A401','#5E7FF1','#FC0000','#1FFF00','#9b9b9b']
    nt_cmap = mpl.colors.LinearSegmentedColormap.from_list(
                        'Custom cmap', cmaplist, len(cmaplist))
    nt_bounds = np.linspace(0, len(cmaplist), len(cmaplist)+1)
    nt_norm = mpl.colors.BoundaryNorm(nt_bounds, nt_cmap.N)    
    sd = sc.cells['spatial_domain'].to_numpy()
    nt_col = [nt_cmap(j) for j in sd[ntdx]]
    

    msize_range = 80
    msize_min = 10
    y = sc.cells['eig_centrality'].to_numpy()
    #y = sc.cells['pioneer_contact'].to_numpy()
    msize = y/np.max(y)*msize_range + msize_min
    nt_msize = msize[ntdx]
    t_msize = msize[tdx]
    
    if params.fout is not None:
        t_num,nt_num = len(tdx), len(ntdx)
        saved_data = {
                'C':np.zeros((t_num+nt_num,2)),
                'X':np.zeros(sc.X.shape),
                'genes':sc.genes[sc.gene_key].tolist()
                }
        saved_data['C'][:t_num,:] = sc.D.components
        saved_data['C'][t_num:,:] = _Xnt
        saved_data['X'][:t_num,:] = sc.X[tdx,:]
        saved_data['X'][t_num:,:] = sc.X[ntdx,:]

        with open(params.fout,'wb') as outfile:
            pickle.dump(saved_data,outfile,protocol=pickle.HIGHEST_PROTOCOL)
            print(f'Wrote to {params.fout}')

    p1 = sc.D.components[:,0]
    p2 = sc.D.components[:,1]
    fig,ax = plt.subplots(1,1,figsize=(5,5))
    ax.scatter(p1,p2,s=t_msize,marker='s',c=t_col,cmap=t_cmap,norm=t_norm)
    ax.scatter(_Xnt[:,0],_Xnt[:,1],s=nt_msize,c=nt_col,cmap=nt_cmap,norm=nt_norm)
    #add_hover(sc.cells,_hover_cells)
    #add_hover(ord_ntcells,hover_cells)
    #add_hover(ord_tcells,hover_cells)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    if params.fout is not None:
        fout = params.fout.replace('.pickle', '.svg')
        plt.savefig(fout)
    plt.show()

def gaussian(x0,y0,A,X,Y,sigma=1):
    return A*np.exp(-(((X-x0)**2 + (Y-y0)**2)/(2*sigma**2)))


def init_expression(data,gene,scale=100,margin=5):
    gdx = data['genes'].index(gene)
    scale = 100
    margin = 5
    C = np.true_divide(scale*(data['C'] - data['C'].min(axis=0,keepdims=True)), 
                        data['C'].max(0) - data['C'].min(0))
    C = np.around(C).astype(int) + margin
    xmax = scale+2*margin 
    X = np.linspace(0,xmax,xmax+1)
    Y = np.linspace(0,xmax,xmax+1)
    X,Y = np.meshgrid(X,Y)
    Z = np.zeros_like(X)
    for i in range(C.shape[0]): 
        Z += gaussian(C[i,0],xmax-C[i,1],data['X'][i,gdx],X,Y,sigma=2)
    return Z

def run_diffusion(params):
    data =  pickle.load( open( params.fin, "rb" ) )
    gene = 'cwn-2' 
    zmin,zmax = 0,np.around(data['X'].max()+1)
    Z = run_gene_diffusion(data,gene) 
    fig,ax = plt.subplots(1,1,figsize=(5,5))
    plot_heatmap(ax,Z,gene,zmin=zmin,zmax=0.2*zmax)
    plt.show()

def run_gene_diffusion(data,gene,nsteps=100):
    Z = init_expression(data,gene)
    #D = Diffusion2D(dx=1,dy=1,D=4)
    #Z0 = Z.copy() 
    #for i in range(nsteps):Z0,Z = D.iter(Z0,Z) 
    #return Z0 
    return Z

def batch_gene_diffusion(params):
    @plot_multi_pages()
    def run_plot(data,index=None,ax=None,**kwargs):
        gene,grp = kwargs['genes'][index]
        grpmap = ['Sparse','Broad 0','SD 3/4','Broad 1','Broad 2'] 
        zmin,zmax = 0,np.around(data['X'].max()+1)
        #zmin,zmax = 0,0.1*np.around(data['X'].max()+1)
        Z = run_gene_diffusion(data,gene) 
        label = f"{gene}~~{grpmap[int(grp)-1]}"
        plot_heatmap(ax,Z,label,zmin=zmin,zmax=zmax)

    data =  pickle.load( open( params.fin, "rb" ) )
    num_genes = len(data['genes'])
    ordered_genes = sorted(data['genes'])
    gfile = 'data/pioneer_embedding/clr_gene_groups.csv'
    gene_grps = read.into_list(gfile,multi_dim=True)
    #gg = defaultdict(list)
    #for (gene,grp) in gene_grps: gg[int(grp)].append(gene)
    #for (k,v) in gg.items(): gg[k] = sorted(v)
    gorder = sorted(gene_grps,key=lambda x: (x[1],x[0]))
    run_plot(data,num_genes,fout=params.fout,genes=gorder)

def compile_gene_diffusions(params):
    data =  pickle.load( open( params.fin, "rb" ) )
    genes = data['genes']

def plot_heatmap(ax,Z,label,zmin=None,zmax=None,cmap='plasma'):
    if zmin is None: zmin = Z.min()
    if zmax is None: zmax = Z.max()
    ax.imshow(Z,vmin=zmin,vmax=zmax,cmap=cmap,interpolation='nearest')
    ax.text(0.05,0.9,f"{label}",transform=ax.transAxes,fontsize=8,color='w') 
    ax.set_xticks([])
    ax.set_yticks([])
 
if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument('fin',
                action = 'store',
                help = 'Input file. Config file for aggregator, pickle file for tsne')

   
    parser.add_argument('mode',
                        action = 'store',
                        choices = dir(explore) + [t for (t,o) in getmembers(sys.modules[__name__]) if isfunction(o)],
                        help = 'Function call')
   
    parser.add_argument('--filter_level',
                        dest='filter_level',
                        action = 'store',
                        default = -1,
                        required = False,
                        type = int,
                        help = 'Keep cells with filter_level above value')
    
    parser.add_argument('--weighted',
                    dest='weighted',
                    action = 'store_true',
                    default = False, 
                    required = False,
                    help = 'If flag, use weighted aggregator') 
    
    parser.add_argument('-o',
                        dest='dout',
                        action = 'store',
                        default = 'data/sc_aggr/',
                        required = False,
                        help = 'Output directory')
    
    parser.add_argument('--fout',
                        dest='fout',
                        action = 'store',
                        default = None,
                        required = False,
                        help = 'Path to output file')
    
    parser.add_argument('--genes',
                        dest='genes',
                        action = 'store',
                        default = 'all',
                        required = False,
                        help = ('Path to gene list file, if provided '
                            'analysis will be restricted genes in the list'))
    
    parser.add_argument('--target_cells',
                        dest='target_cells',
                        action = 'store',
                        default = None,
                        required = False,
                        help = 'Path to list of target cells')

    params = parser.parse_args()
    eval(params.mode + '(params)')
