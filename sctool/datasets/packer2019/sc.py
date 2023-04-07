"""
@name: packer2019.sc.py                      
@description:                  
    SC class extension for Packer (2019) dataset

@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05              
"""

from pycsvparser import read
from sctool.sc import SingleCell as SC

TORDER = [
        '< 100', 
        '100-130',
        '130-170', 
        '170-210', 
        '210-270', 
        '270-330', 
        '330-390', 
        '390-450', 
        '450-510', 
        '510-580', 
        '580-650', 
        '> 650' 
        ]



class SingleCell(SC):
    def __init__(self,cfg,load_light=False):
        SC.__init__(self,cfg,load_light)
        self.gene_list = []

    def load_gene_list(self,lstname):
        self.gene_list = read.into_list(self.cfg['gene_list'][lstname])
    

