#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 22:36:20 2019

@author: famgarciatorres
"""

import pandas as pd
import numpy as np
import pyphi as phi
import pyphi_plots as pp

Cars_Features=pd.read_excel('Automobiles PLS.xls', 'Features', index_col=None, na_values=np.nan)
Cars_Features_data=np.array(Cars_Features.values[:,1:]).astype(float)

Cars_Performance=pd.read_excel('Automobiles PLS.xls', 'Performance', index_col=None, na_values=np.nan)
Cars_Performance_data=np.array(Cars_Performance.values[:,1:]).astype(float)


X=Cars_Features_data[19:59,:]
Y=Cars_Performance_data[19:59,:]
#full data set, will use SVD
pcaobj1=phi.pca(X,3)
#full data set, will use NIPALS
pcaobj1b=phi.pca(X,3,force_nipals=True)

#Data set with missing data will use NIPALS
pcaobj2=phi.pca(Cars_Features_data,3)
#Full data set, will use SVD
pls_obj1=phi.pls(X,Y,3)
#Full data set, forcing NIPALS
pls_obj2=phi.pls(X,Y,3,force_nipals=True)
#Data set with missing data, will use NIPALS
pls_obj3=phi.pls(Cars_Features_data,Cars_Performance_data,3)


pp.r2pv_plot(pls_obj3)
pp.r2pv_plot(pcaobj1)


pca_obj_wDF=phi.pca(Cars_Features,3)
pls_obj_wDF=phi.pls(Cars_Features,Cars_Performance,3)

pp.r2pv_plot(pca_obj_wDF)
pp.r2pv_plot(pls_obj_wDF)
