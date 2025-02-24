#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 09:54:43 2019

@author: Sal Garcia
"""
import pandas as pd
import numpy as np
import pyphi as phi
import pyphi_plots as pp

# Data taken from:
# Dyrby, M., Engelsen, S.B., Nørgaard, L., Bruhn, M. and Lundsberg-Nielsen, L., 2002. 
# Chemometric quantitation of the active substance (containing C≡ N) in a pharmaceutical 
# tablet using near-infrared (NIR) transmittance and NIR FT-Raman spectra. 
# Applied spectroscopy, 56(5), pp.579-585.

# Load the data from Excel
NIR_Spectra     = pd.read_excel('NIR.xlsx', 'NIR', index_col=None, na_values=np.nan)
API_Conc          = pd.read_excel('NIR.xlsx', 'Y', index_col=None, na_values=np.nan)
Tablet_Categories = pd.read_excel('NIR.xlsx', 'Categorical', index_col=None, na_values=np.nan)

#Use pyphi_plots to plot spectra
pp.plot_spectra(NIR_Spectra,plot_title='NIR Spectra',tab_title='Spectra Raw',
             xaxis_label='channel',yaxis_label='a.u')    

# Create a new Pandas Data Frame with spectra_snv(NIR_Spectra)
NIR_Spectra_spectra_snv=phi.spectra_snv(NIR_Spectra)

#Use pyphi_plots to plot spectra
pp.plot_spectra(NIR_Spectra_spectra_snv,plot_title='NIR with spectra_snv',tab_title='Spectra spectra_snv',
             xaxis_label='channel',yaxis_label='a.u')

# Create a new Pandas Data Frame with Pre-processed spectra
# SavGol transform with: 
#   window_size      = 10
#   derivative order = 1
#   polynomial order = 2
NIR_Spectra_spectra_snv_savgol,M=phi.spectra_savgol(10,1,2,NIR_Spectra_spectra_snv)

#Use pyphi_plots to plot spectra
pp.plot_spectra(NIR_Spectra_spectra_snv_savgol,plot_title='NIR with spectra_snv and Savitzky Golay [10,1,2]',tab_title='Spectra spectra_snv+SavGol',
             xaxis_label='channel',yaxis_label='a.u')

#Build three models and contrast
pls_NIR_calibration=phi.pls(NIR_Spectra,API_Conc,3,mcsX='center',mcsY='center')

pp.predvsobs(pls_NIR_calibration,NIR_Spectra,API_Conc)

pls_NIR_calibration=phi.pls(NIR_Spectra_spectra_snv,API_Conc,3,mcsX='center',mcsY='center',cross_val=10)
pp.predvsobs(pls_NIR_calibration,NIR_Spectra_spectra_snv,API_Conc)

pls_NIR_calibration=phi.pls(NIR_Spectra_spectra_snv_savgol,API_Conc,1,mcsX='center',mcsY='center',cross_val=10)
pp.predvsobs(pls_NIR_calibration,NIR_Spectra_spectra_snv_savgol,API_Conc,CLASSID=Tablet_Categories,colorby='Type')
pp.predvsobs(pls_NIR_calibration,NIR_Spectra_spectra_snv_savgol,API_Conc,CLASSID=Tablet_Categories,colorby='Scale')
