"""
@name: sc.py
@description:
Contains the SingleCell class for handling/processing single cell data

@author: Christopher Brittin
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05
"""

import scipy.sparse as sp
import numpy as np
import time
import pandas as pd

from pycsvparser import read

class SingleCell:
    def __init__(self,cfg,load_light=False):
        """"
        Initialize class.

        Load meta data and binary count matrix.
        Assumes [genes x cells] matrix. Convert to [cells x genes]

        Parameters
        ----------
        cfg: str
            Path to config file
        load_light: bool, default: False
            If True, does not load the count matrix. Useful for just loading
            the meta data for the purposes of prototyping. 
        
        Attributes
        ----------
        cfg: ConfigParser 
            Holds config settings for the single cell data
        cells: Pandas dataframe 
            cells metadata
        genes: Pandas dataframe,
            genes metadata
        X: scipy sparse coo matrix [cells x genes] 
            counts matrix
        
        Methods
        -------
        restore_data: Loads data from config file
        select_cells: Select cells from data
        select_genes: Select genes from data
        """
        self.cfg = cfg 
        self.gene_key = self.cfg['keys']['meta_key_gene']
        self.cell_key = self.cfg['keys']['meta_key_cell']
        self.load_light = load_light
        self.restore_data()
        self.gene_list = []
        self.meta = {}

        """
        self.gene_idx = None
        self.cell_idx = None
        """

    def restore_data(self):
        """
        Loads data specifed by config file path
        """
        self.cells = pd.read_csv(self.cfg['files']['cells_meta'])
        self.genes = pd.read_csv(self.cfg['files']['genes_meta'])
        if not self.load_light: self.load_matrices()
    
    def load_gene_list(self,lstname):
        """
        Convenience function for cacheing a list of genes

        Args:
        -----
        lstname: string
            path to gene list text file
        """
        self.gene_list = read.into_list(self.cfg['gene_list'][lstname])
    
    def backup_count_matrix(self):
        self._X = self.X.copy()
   
    def set_count_matrix(self,X,backup=True):
        if backup: self.backup_count_matrix()
        self.X = X

    def save_cells(self,fout):
        fout = self.cfg['files']['top_dir'] + '/' + fout
        self.cells.to_csv(fout,index=False) 

    def load_matrices(self):
        self.X = sp.load_npz(self.cfg['files']['count_matrix'])
        #Ensure that there are no zeros in the sparse matrix 
        self.X.eliminate_zeros()
        if self.X.shape[0] == self.genes.shape[0]:
            #Convert from [genes x cells] -> [cells x genes]
            self.X = self.X.transpose()      
        
        check = (self.X.shape[0] == self.cells.shape[0]) and (self.X.shape[1] == self.genes.shape[0])
        assert check, f"Shapes do not match: Count matrix is {self.X.shape} but (cell,genes) = {self.cells.shape[0],self.genes.shape[0]}"

    def reorder_rows(self,idx):
        self.cells = self.cells.reindex(idx)
        self.filter_matrices_rows(idx)
        if self.cell_idx is not None: self.index_cells()

    def filter_matrices_rows(self,idx):
        if sp.issparse(self.X): 
            self.X = self.X.tocsr()[idx,:].tocoo()
        else:
            self.X = self.X[idx,:]
    
    def filter_matrices_cols(self,jdx):
        if sp.issparse(self.X): 
            self.X = self.X.tocsr()[:,jdx].tocoo()
        else:
            self.X = self.X[:,jdx]
    
    def cell_columns(self):
        return list(self.cells.columns)
    
    def gene_columns(self):
        return list(self.genes.columns)
    
    def get_unique_cell_meta(self,key):
        return self.cells[key].unique().tolist()
   
    def get_unique_cells(self):
        return self.get_unique_cell_meta(self.cell_key)

    def select_cells(self,idx):
        """
        Select cells (matrix rows) from the data.

        Parameters
        ----------
        idx: list
            Cell indicies to keep. Indicies can be obtained from 
            dataframe query. 
        
        Example
        -------
        >> sc = SingleCell(cfg)
        >> idx = sc.cells.index[sc.cells["attribute"] == "value"].tolist()
        >> sc.select_cell(idx)

        """
        self.cells = self.cells.iloc[idx]#.reset_index()
        self.cells.index = range(len(self.cells.index))
        if not self.load_light: self.filter_matrices_rows(idx) 

    def filter_cells_isin(self,attr,values):
        """
        Keep cells (rows) with meta attibute in some values list.

        Parameters
        ----------
        attr: string
            Cells dataframe column name
        values: list
            Cells with attr in list values will be kept 
        
        """
        jdx = self.cells.index[self.cells[attr].isin(values)].tolist()
        self.select_cells(jdx)
    
    def filter_cells_bool(self,attr,value):
        """
        Keep cells (rows) with meta attibute that has boolean value.

        Parameters
        ----------
        attr: string
            Cells dataframe column name
        value: bool
            Cells with attr with bool value will be kept 
        
        """
        jdx = self.cells.index[self.cells[attr]==value].tolist()
        self.select_cells(jdx)
    
    def filter_cells_eq(self,attr,value):
        """
        Keep cells (rows) with meta attibute eq (==) to value.

        Parameters
        ----------
        attr: string
            Cells dataframe column name
        value: bool
            Cells with attr with bool value will be kept 
        
        """
        jdx = self.cells.index[self.cells[attr]==value].tolist()
        self.select_cells(jdx)
    
    def filter_cells_gt(self,attr,value):
        """
        Keep cells (rows) with meta attibute gt (>) value.

        Parameters
        ----------
        attr: string
            Cells dataframe column name
        value: bool
            Cells with attr with bool value will be kept 
        
        """
        jdx = self.cells.index[self.cells[attr]>value].tolist()
        self.select_cells(jdx)
    
    def filter_cells_lt(self,attr,value):
        """
        Keep cells (rows) with meta attibute lt (<) value.

        Parameters
        ----------
        attr: string
            Cells dataframe column name
        value: bool
            Cells with attr with bool value will be kept 
        
        """
        jdx = self.cells.index[self.cells[attr]<value].tolist()
        self.select_cells(jdx)

    def filter_cells_gte(self,attr,value):
        """
        Keep cells (rows) with meta attibute gt (>=) value.

        Parameters
        ----------
        attr: string
            Cells dataframe column name
        value: bool
            Cells with attr with bool value will be kept 
        
        """
        jdx = self.cells.index[self.cells[attr]>=value].tolist()
        self.select_cells(jdx)
    
    def filter_cells_lte(self,attr,value):
        """
        Keep cells (rows) with meta attibute lt (<=) value.

        Parameters
        ----------
        attr: string
            Cells dataframe column name
        value: bool
            Cells with attr with bool value will be kept 
        
        """
        jdx = self.cells.index[self.cells[attr]<=value].tolist()
        self.select_cells(jdx)

    def select_genes(self,jdx):
        """
        Select genes (matrix columns) from the data.

        Parameters
        ----------
        jdx: list
            Gene indicies to keep. Indicies can be obtained from 
            dataframe query.
        
        Example
        -------
        >> sc = SingleCell(cfg)
        >> jdx = sc.genes.index[sc.genes["name"].isin(['gene1','gene2','gene3'])].tolist()
        >> sc.select_genes(jdx)

        """
        self.genes = self.genes.iloc[jdx].reset_index()
        if not self.load_light: self.filter_matrices_cols(jdx)
    

    def filter_genes_isin(self,attr,values):
        """
        Keep genes (cols) with meta attibute in some values list.

        Parameters
        ----------
        attr: string
            Genes dataframe column name
        values: list
            Genes with attr in list values will be kept 
        
        """
        jdx = self.genes.index[self.genes[attr].isin(values)].tolist()
        self.select_genes(jdx)

    def filter_genes_bool(self,attr,value):
        """
        Keep genes (columns) with meta attibute in some values list.

        Parameters
        ----------
        attr: string
            Genes dataframe column name
        value: bool
            Genes with attr with bool value will be kept 
        
        """
        jdx = self.genes.index[self.genes[attr]==value].tolist()
        self.select_genes(jdx)
    
    def get_gene_index(self,genes,key=None):
        if type(genes) is not list: genes = [genes]
        if key is None: key = self.gene_key
        jdx = []
        for g in genes:
            _jdx = self.genes.index[self.genes[key] == g].tolist()
            jdx.append(_jdx[0])
        return jdx

    def index_genes(self):
        self.gene_idx = dict([(g,i) for (i,g) in 
                            enumerate(self.genes[self.gene_key].tolist())])
    
    def index_cells(self):
        self.cell_idx = dict([(g,i) for (i,g) in 
                            enumerate(self.cells[self.cell_key].tolist())])
    
    def get_gene_vector(self,gene):
        return self.X[:,self.gene_idx[gene]]
    
    def get_cell_vector(self,cell):
        return self.X[self.cell_idx[cell],:]

    def get_batch_names(self):
        return getattr(self.cells,self.cfg['meta_keys']['batch']).unique()
    
    def get_batch_indicies(self,batch):
        return self.cells.sc.column_has_values(self.cfg['meta_keys']['batch'],batch)
    
    def encode_cell_values(self,attr=None):
        """
        Return a numerically encoded list of attribute values
        
        Useful for when needing a quick list to encode colors for scatter plots
        """
        if attr is None: attr = self.cell_key
        amap = dict([(v,i) for (i,v) in enumerate(set(self.cells[attr].tolist()))])
        return [amap[v] for v in self.cells[attr].tolist()]

