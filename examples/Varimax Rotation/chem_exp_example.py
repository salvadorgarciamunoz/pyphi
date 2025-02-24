#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 09:45:58 2022

@author: c184156
"""
import pandas as pd
import numpy as np
import pyphi as phi
import pyphi_plots as pp
import matplotlib.pyplot as plt


ra_dataset=pd.read_excel('chemical_experiments_dataset.xlsx',sheet_name='data')

ra_dataset,ecols=phi.clean_low_variances(ra_dataset)

x_cols=[
 'Lot',
 'SM (eq)',
 'A (eq)',
 'B (eq)',
 'C (eq)',
 'Solvent (Vol)',
 'drops C',
 'drops D',
 'Water (vol)',
 'N addition order',
 'SM addition order',
 'A addition order',
 'B addition order',
 'C Addition Temp C',
 'N Addition Temp C',
 'Reaction Temp C',
 'number of portions N']
y_cols=[
 'Lot',
 'IPC_N',
 'IPC_H',
 'IPC_RT_17min',
 'IPC_RT_19min']

ra_X=ra_dataset[x_cols]
ra_Y=ra_dataset[y_cols]


#%%
plsobj=phi.pls(ra_X,ra_Y,5)
#pp.loadings(plsobj)
#pp.weighted_loadings(plsobj)
pp.weighted_loadings(plsobj)
pp.loadings_map(plsobj,[1,2],plotwidth=600)
pp.score_scatter(plsobj,[1,2],add_ci=True)
pp.r2pv(plsobj)
plsobjR=phi.varimax_rotation(plsobj,ra_X,Y=ra_Y)

pp.weighted_loadings(plsobjR)
pp.loadings_map(plsobjR,[1,2],plotwidth=600)
pp.score_scatter(plsobjR,[1,2],add_ci=True)
pp.r2pv(plsobjR)
#%%

