#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper functions for pyphi
"""

import os

def create_resdir(mdlname):
    """
    Create a directory to store the results of model 'mdlname'. If the directory
    already exists, a message will alert that the directory exists and will not
    delete the directory

    Parameters
    ----------
    mdlname : str name given to a model when running pca or pls

    Returns
    -------
    None.

    """
    if os.path.isdir(mdlname):
        print('Directory ',mdlname,' already exists, choose another name')
        return
    else:
        os.mkdir(mdlname)
    return    


