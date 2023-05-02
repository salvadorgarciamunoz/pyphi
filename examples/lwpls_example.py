#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Data set discussed in:

M. Dyrby, S.B. Engelsen, L. Nørgaard, M. Bruhn and L. Lundsberg Nielsen
Chemometric Quantitation of the Active Substance in a Pharmaceutical Tablet Using Near Infrared (NIR) Transmittance and NIR FT Raman Spectra
Applied Spectroscopy 56(5): 579 585 (2002)


"""

import scipy.io as spio
import pyphi as phi
import pyphi_plots as pp
import numpy as np
from bokeh.io import show, output_file
from bokeh.plotting import figure

NIRData=spio.loadmat('NIRdata_tablets.MAT')
X = np.array(NIRData['Matrix'][:,3:])
Y = np.array(NIRData['Matrix'][:,0])

Xcal = X[::2,:]
Xval = X[1:X.shape[1]:2,:]
Xcal = phi.snv(Xcal)
Xval = phi.snv(Xval)
Xcal,M = phi.savgol(5,1,2,Xcal)
Xval,M = phi.savgol(5,1,2,Xval)
pp.plot_spectra(Xcal)

Ycal = Y[::2]
Ycal = np.reshape(Ycal,(len(Ycal),-1))
Yval = Y[1:X.shape[1]:2]
Yval = np.reshape(Yval,(len(Yval),-1))

mvm_pls=phi.pls(Xcal,Ycal,1,mcsX='center',mcsY='center')
yhatval_pls = phi.pls_pred(Xval,mvm_pls)
yhatcal_pls = phi.pls_pred(Xcal,mvm_pls)

PLSerrrCAL=[]
LWPLSerrCAL=[]

#This for loop will make predictions for the calibration set using various values for the localization parameter
for loc_param in [5,10,15,20,25,30,35,40,45,50,100,250]:
    yhatlw=[]
    for o in list(range(Xcal.shape[0])):
        yhatlw_=phi.lwpls(Xcal[o,:],loc_param,mvm_pls,Xcal,Ycal,shush=True)
        yhatlw.append(yhatlw_[0])
    yhatlw=np.array(yhatlw)
    yhatlw=np.reshape(yhatlw,(-1,1))
    print("Localization = ",loc_param)
    errorPLS   = Ycal-yhatcal_pls['Yhat']
    errorLWPLS = Ycal-yhatlw
    RMSE_PLS =np.sqrt(np.mean(errorPLS**2,axis=0))
    RMSE_LWPLS=np.sqrt(np.mean(errorLWPLS**2,axis=0))
    print("PLS   RMSE:",RMSE_PLS)
    print("LWPLS RMSE:",RMSE_LWPLS)
    PLSerrrCAL.append(RMSE_PLS)
    LWPLSerrCAL.append(RMSE_LWPLS)
 
    
TOOLS = "save,wheel_zoom,box_zoom,pan,reset,box_select,lasso_select"
TOOLTIPS = [
            ("index", "$index"),
            ("(x,y)", "($x, $y)"),
                ]
l_param=[5,10,15,20,25,30,35,40,45,50,100,250]

rnd_num=str(int(np.round(1000*np.random.random_sample())))
output_file("LocParam"+rnd_num+".html",title='LocParam')
p = figure(tools=TOOLS, tooltips=TOOLTIPS,plot_width=600, plot_height=600, title="Obs vs Pred")
p.circle(l_param,PLSerrrCAL,size=7,color='darkblue',legend_label="PLS Error CAL")
p.circle(l_param,LWPLSerrCAL,size=7,color='red',legend_label="LWPLS Error CAL")
p.xaxis.axis_label = 'Localization Parameter'
p.yaxis.axis_label = 'RMSE'
show(p)

PLSerrrVAL=[]
LWPLSerrVAL=[]
#This for loop will make predictions for the validation set using various values for the localization parameter
for loc_param in [5,10,15,20,25,30,35,40,45,50,100,250]:
    yhatlw=[]
    for o in list(range(Xval.shape[0])):
        yhatlw_=phi.lwpls(Xval[o,:],loc_param,mvm_pls,Xcal,Ycal,shush=True)
        yhatlw.append(yhatlw_[0])
    yhatlw=np.array(yhatlw)
    yhatlw=np.reshape(yhatlw,(-1,1))
    print("Localization = ",loc_param)
    errorPLS   = Yval-yhatval_pls['Yhat']
    errorLWPLS = Yval-yhatlw
    RMSE_PLS =np.sqrt(np.mean(errorPLS**2,axis=0))
    RMSE_LWPLS=np.sqrt(np.mean(errorLWPLS**2,axis=0))
    print("PLS   RMSE:",RMSE_PLS)
    print("LWPLS RMSE:",RMSE_LWPLS)
    PLSerrrVAL.append(RMSE_PLS)
    LWPLSerrVAL.append(RMSE_LWPLS)
 
rnd_num=str(int(np.round(1000*np.random.random_sample())))
output_file("LocParam"+rnd_num+".html",title='LocParam')
p = figure(tools=TOOLS, tooltips=TOOLTIPS,plot_width=600, plot_height=600, title="Obs vs Pred")
p.circle(l_param,PLSerrrVAL,size=7,color='darkblue',legend_label="PLS Error VAL")
p.circle(l_param,LWPLSerrVAL,size=7,color='red',legend_label="LWPLS Error VAL")
p.xaxis.axis_label = 'Localization Parameter'
p.yaxis.axis_label = 'RMSE'
show(p)


