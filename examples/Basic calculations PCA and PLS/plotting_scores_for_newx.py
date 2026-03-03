# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 08:37:36 2024
Example on how to plot scores for observations not in the model
@author: salva
"""

import pandas as pd
import numpy as np
import pyphi.calc as phi
import pyphi.plots as pp

# Load the data from Excel
Cars_Features_MD    = pd.read_excel('Automobiles PCA w MD.xlsx', 'Features', index_col=None, na_values=np.nan,engine='openpyxl')
Cars_Performance    = pd.read_excel('Automobiles PLS.xlsx', 'Performance', index_col=None, na_values=np.nan,engine='openpyxl')
Cars_Features       = pd.read_excel('Automobiles PLS.xlsx', 'Features', index_col=None, na_values=np.nan,engine='openpyxl')
Cars_CLASSID        = pd.read_excel('Automobiles PCA w MD.xlsx', 'CLASSID', index_col=None, na_values=np.nan,engine='openpyxl')

Cars_Features_new = Cars_Features.iloc[:100,:]
Cars_Features      = Cars_Features.iloc[100:,:]
Cars_CLASSID_new  = Cars_CLASSID.iloc[:100,:]
Cars_CLASSID      = Cars_CLASSID.iloc[100:,:]
#Cars_CLASSID.reset_index(inplace=True,drop=True)
# Build a PCA model with 3 PC's,
pcaobj=phi.pca(Cars_Features,3)

pp.score_scatter(pcaobj,[1,2],CLASSID=Cars_CLASSID,colorby='Cylinders')
pp.score_scatter(pcaobj,[1,2],Xnew=Cars_Features_new, CLASSID=Cars_CLASSID_new,colorby='Cylinders',addtitle='For New Observations')
pp.score_scatter(pcaobj,[1,2],Xnew=Cars_Features_new, include_model=True,addtitle='For New Observations w/ Model')

pp.score_scatter(pcaobj,[1,2],Xnew=Cars_Features_new, CLASSID=Cars_CLASSID_new,
                 colorby='Cylinders',include_model=True,
                 addtitle='For New Observations w Classifiers and Model')
