"""
@name: util.py                      
@description:                  
    Utility functions for sctool

@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05              
"""

from configparser import ConfigParser,ExtendedInterpolation
import importlib


def checkout(cfile,dname):
    """ 
    Checkout data from config file 

    Args:
    -----
    cfile: str, path to config file
    dname: str, key name in the config file
    
    Returns:
    --------
    named tuple
    """
    cfg = ConfigParser(interpolation=ExtendedInterpolation()) 
    cfg.read(cfile)
    
    dfile = cfg['datasets'][dname]
    cfg = ConfigParser(interpolation=ExtendedInterpolation()) 
    cfg.read(dfile)
    return cfg

def load_sc(cfg,**kwargs):
    """ 
    Loads SingleCell class from module specified by the cfg file

    Args:
    -----
    cfg: ConfigParser
    kwargs: Keyword args for the SingleCell class

    Returns:
    --------
    SingleCell class
    """
    #module = importlib.import_module(f"sctool.datasets.{cfg['meta']['alias']}.sc") 
    module = importlib.import_module(f"sctool.sc")
    return module.SingleCell(cfg,**kwargs)

