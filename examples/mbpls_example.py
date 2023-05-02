# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 18:40:39 2022

@author: salva
"""

import pyphi as phi
import pyphi_plots as pp
import pandas as pd


x1_data=pd.read_excel('MBDataset.xlsx','X1')
x2_data=pd.read_excel('MBDataset.xlsx','X2')
x3_data=pd.read_excel('MBDataset.xlsx','X3')
x4_data=pd.read_excel('MBDataset.xlsx','X4')
x5_data=pd.read_excel('MBDataset.xlsx','X5')
x6_data=pd.read_excel('MBDataset.xlsx','X6')
y_data=pd.read_excel('MBDataset.xlsx','Y')

mbdata={'X1':x1_data,
        'X2':x2_data,
        'X3':x3_data,
        'X4':x4_data,
        'X5':x5_data,
        'X6':x6_data
        }

mbpls_obj=phi.mbpls(mbdata,y_data,2)

preds=phi.pls_pred(mbdata, mbpls_obj)

pp.score_scatter(mbpls_obj, [1,2])
pp.weighted_loadings(mbpls_obj)
pp.loadings(mbpls_obj)
pp.vip(mbpls_obj)

pp.r2pv(mbpls_obj)
pp.mb_r2pb(mbpls_obj)
pp.mb_weights(mbpls_obj)
pp.mb_vip(mbpls_obj)


