#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example script to build a PCA and a PLS model
"""

import pandas as pd
import numpy as np
import pyphi as phi
import pyphi_plots as pp

# Load the data from Excel
Cars_Features_MD    = pd.read_excel('Automobiles PCA w MD.xls', 'Features', index_col=None, na_values=np.nan)
Cars_Performance    = pd.read_excel('Automobiles PLS.xls', 'Performance', index_col=None, na_values=np.nan)
Cars_Features       = pd.read_excel('Automobiles PLS.xls', 'Features', index_col=None, na_values=np.nan)
Cars_CLASSID        = pd.read_excel('Automobiles PCA w MD.xls', 'CLASSID', index_col=None, na_values=np.nan)

# Build a PCA model with 3 PC's,
pcaobj1=phi.pca(Cars_Features_MD,3,md_algorithm='nlp')
print(pcaobj1['P'])
pp.score_scatter(pcaobj1,[1,2],CLASSID=Cars_CLASSID,colorby='Cylinders')
pp.loadings(pcaobj1)

pcaobj2=phi.pca(Cars_Features_MD,3)
print(pcaobj2['P'])
pp.score_scatter(pcaobj2,[1,2],CLASSID=Cars_CLASSID,colorby='Cylinders')
pp.loadings(pcaobj2)

pcaobj3=phi.pca(Cars_Features,3)
print(pcaobj3['P'])
pp.score_scatter(pcaobj3,[1,2],CLASSID=Cars_CLASSID,colorby='Cylinders')
pp.loadings(pcaobj3)


plsobj=phi.pls(Cars_Features,Cars_Performance,3)
pp.r2pv(plsobj)
pp.score_scatter(plsobj,[1,2],CLASSID=Cars_CLASSID,colorby='Origin',add_ci=True)
pp.loadings(plsobj)

plsobj_nlp=phi.pls(Cars_Features_MD,Cars_Performance,3,md_algorithm='nlp')
pp.r2pv(plsobj_nlp)
pp.score_scatter(plsobj_nlp,[1,2],CLASSID=Cars_CLASSID,colorby='Origin',add_ci=True)
pp.loadings(plsobj_nlp)
               