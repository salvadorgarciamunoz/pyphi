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
Cars_Features    = pd.read_excel('Automobiles PLS.xls', 'Features', index_col=None, na_values=np.nan)
Cars_Performance = pd.read_excel('Automobiles PLS.xls', 'Performance', index_col=None, na_values=np.nan)
Cars_CLASSID     = pd.read_excel('Automobiles PLS.xls', 'CLASSID', index_col=None, na_values=np.nan)

# Build a PCA model with 3 PC's, cross validating by elements removing 5% of the data per round
pcaobj=phi.pca(Cars_Features,3,cross_val=5)

#Make some plots
# Captured variance per component per variable
pp.r2pv(pcaobj)
#t1  Line plot color coded by Car Origin
pp.score_line(pcaobj,1,CLASSID=Cars_CLASSID,colorby='Origin',add_ci=True,add_labels=True)
#Loadings map for 1 and 2nd PC
pp.loadings_map(pcaobj,[1,2])
#t1 vs t2 color coded by car origin
pp.score_scatter(pcaobj,[1,2],CLASSID=Cars_CLASSID,colorby='Origin')
#t1 vs t2 color coded by cylinder (where Cylinder is a discrete class)
pp.score_scatter(pcaobj,[1,2],CLASSID=Cars_CLASSID,colorby='Cylinders')
#t1 vs t2 color coded by Car Wegith using 4 bins to group 
pp.score_scatter(pcaobj,[1,2],CLASSID=Cars_Features,colorby='Weight',nbins=4,
                 addtitle='Color coded by Car Weight')
#Bar plots of loadings
pp.loadings(pcaobj)
#Bar plots of loadings weighted by the R2 per variable per component
pp.weighted_loadings(pcaobj)
#Hotellings T2 and SPE and Outlier plot, adding a t1-t2 plot at the end
pp.diagnostics(pcaobj, score_plot_xydim=[1,2])
#Plot contributions from mean to Car1
pp.contributions_plot(pcaobj,Cars_Features,'scores',to_obs=['Car1'])
#Plot contributions from Car1 to Car4
pp.contributions_plot(pcaobj,Cars_Features,'scores',to_obs=['Car1'],from_obs=['Car4'])

# Build a PLS model with 3 PC's, cross validating by elements 
# removing 5% of the data per round
plsobj=phi.pls(Cars_Features,Cars_Performance,3,cross_val=5)
# Build a PLS model with 3 PC's, cross validating by elements 
# removing 5% of the data per round adding crossval of X Space
plsobj=phi.pls(Cars_Features,Cars_Performance,3,cross_val=5,cross_val_X=True)

#Make some plots
pp.r2pv(plsobj)
pp.score_scatter(plsobj,[1,2],CLASSID=Cars_CLASSID,colorby='Cylinders')
pp.score_scatter(plsobj,[1,2],CLASSID=Cars_CLASSID,colorby='Origin',add_ci=True)
pp.loadings(plsobj)
pp.weighted_loadings(plsobj)
pp.diagnostics(plsobj, score_plot_xydim=[1,2])
pp.contributions_plot(plsobj,Cars_Features,'scores',to_obs=['Car1'])
pp.contributions_plot(plsobj,Cars_Features,'scores',to_obs=['Car1'],from_obs=['Car4'])
pp.vip(plsobj)
pp.loadings_map(plsobj,[1,2])
pp.predvsobs(plsobj,Cars_Features,Cars_Performance)
pp.predvsobs(plsobj,Cars_Features,Cars_Performance,CLASSID=Cars_CLASSID,colorby='Origin')
pp.predvsobs(plsobj,Cars_Features,Cars_Performance,CLASSID=Cars_CLASSID,colorby='Origin',x_space=True)


