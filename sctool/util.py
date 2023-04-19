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
import pickle

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

def to_pickle(sc,fout):
    """
    Convenience function for pickling SingleCell instance
    
    Args:
    -----
    sc: SingleCell object
    fout: str, Path to output file

    """
    with open(fout, 'wb') as f: pickle.dump(sc, f)

def from_pickle(fin):
    """
    Convenience function loading SingleCell instance from pickle
    
    Args:
    -----
    fout: str, Path to input pickle file

    """
    with open(fin, 'rb') as f: 
        return pickle.load(f)


