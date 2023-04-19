"""
@name: sc_extension.py
@description:
    Extension to base class SingleCell are kept here

@author: Christopher Brittin
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05
"""
import numpy as np
import networkx as nx


import toolbox.scale as tsc
from sctool.sc import SingleCell
import sctool.feature_selection
from toolbox.scale import standardize
import sctool.decomposition as decomp
from sctool import explore



class SCFeatureSelect(SingleCell):
    def __init__(self,cfg,load_light=False):
        SingleCell.__init__(self,cfg,load_light)

    def filter_hvg(self,method='hvg_seurat3',num_genes=2000):
        """
        Only keeps the num_genes of most highly variable genes.

        Parameters:
        -----------
        method: str (optional, default hvg_seurat3)
            Method used to compute HVG. Currently only hvg_seurat3 is supported

        num_genes: int (optional, default 2000)
            Number of HVGs to keep.
        """
        idx = getattr(sctool.feature_selection,method)(self,num_genes=num_genes)
        self.select_genes(idx)

    def log_standardize(self):
        """
        Applies log10 and then standardizes the count matrix.
        It is suggested to used this function in order to 
        prevent erroneous repeated log applications.
        
        NOTE:: Standardization requires converting from sparse
        to normal array.
        """
        if not self.is_log_standardized: 
            self.X = standardize(np.log10(self.X.toarray() + 1)) 
            self.is_log_standardized = True

    
    def pca(self,n_components=50):
        self.log_standardize() 
        decomp.pca(self,n_components=n_components)

    
    def plot_scree(self,ax=None,**kwargs):
        explore.scree_plot(self,ax=ax,**kwargs) 
   
class SCAggregator(SCFeatureSelect):
    def __init__(self,cfg,load_light=False):
        SCFeatureSelect.__init__(self,cfg,load_light)
        self.A = None

    def load_aggregator(self):
        self.nodes = self.cells[self.cell_key].tolist()
        self.G = nx.read_graphml(self.cfg['files']['aggregator'])
        self.nodes = sorted(list(set(self.nodes) & set(self.G.nodes())))
        self.filter_cells_isin(self.cell_key,self.nodes)
        rm_nodes = [n for n in self.G.nodes() if n not in self.nodes]
        self.G.remove_nodes_from(rm_nodes)
    
    def load_lineage(self):
        self.L = nx.read_graphml(self.cfg['files']['lineage'])

    def get_aggregator(self,edge_attr='weight'):
        return nx.to_numpy_array(self.G,nodelist=self.nodes,weight=edge_attr)
        
    def local_mean_aggregate(self,A):
        #A += np.eye(A.shape[0])
        deg = A.sum(1)
        deg[deg==0] = 1
        D = np.diag(1/deg)
        E = np.dot(D,np.dot(A,self.X))
        return E
    
    def local_diff_aggregate(self,A=None):
        if A is None: A = self.A 
        I = np.eye(A.shape[0])
        deg = A.sum(1)
        deg[deg==0] = 1
        D = np.diag(1/deg)
        A = I - np.dot(D,A)
        E = np.dot(A,self.X)
        return E

    def log_scale(self,scale=1000):
        self.X = tsc.sum_to_target(self.X,scale,axis=1)
        self.X.data = np.log(self.X.data)

    def gene_outer(self,g1,g2):
        if self.gene_idx is None: self.index_genes()
        return np.outer(self.get_gene_vector(g1),self.get_gene_vector(g2))

    def filter_graph_edges(self,weight='weight',min_weight=0):
        nodes = self.cells[self.cell_key].tolist()
        rmedge = [] 
        for (u,v,w) in self.G.edges(data=weight):
            if w < min_weight: rmedge.append((u,v))
        self.G.remove_edges_from(rmedge)
        nodes = list(set(nodes) & set([n for n in self.G.nodes() if self.G.degree(n) > 0]))
        self.filter_cells_isin(self.cell_key,nodes)
 



