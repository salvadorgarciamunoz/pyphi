# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 11:45:35 2023

@author: sgarciam@ic.uk.ac
"""

import pandas as pd
import pyphi as phi
import pyphi_plots as pp


#This is how to fit a basic LPLS model and do some plots
X=pd.read_excel('lpls_dataset.xlsx',sheet_name='X')
R=pd.read_excel('lpls_dataset.xlsx',sheet_name='R')
Y=pd.read_excel('lpls_dataset.xlsx',sheet_name='Y')

lpls_obj = phi.lpls(X,R,Y,4)

#%%
pp.loadings_map(lpls_obj, [1,2],addtitle='LPLS Model')
pp.loadings(lpls_obj,addtitle='LPLS Model')
pp.weighted_loadings(lpls_obj)
pp.vip(lpls_obj)
pp.r2pv(lpls_obj)
pp.score_scatter(lpls_obj, [1,2],add_labels=True,addtitle='Scores for blends')
pp.score_scatter(lpls_obj, [1,2],add_labels=True, rscores=True, addtitle='Scores for Materials')
