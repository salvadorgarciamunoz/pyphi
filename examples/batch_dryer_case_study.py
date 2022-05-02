# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 22:33:38 2022

@author: salva
"""


import pandas as pd
import numpy as np
import pyphi_batch as phibatch
import matplotlib.pyplot as plt
import pyphi_plots as pp
        

bdata        = pd.read_excel('Batch Dryer Case Study.xlsx',sheet_name='Trajectories')
cqa          = pd.read_excel('Batch Dryer Case Study.xlsx',sheet_name='ProductQuality')
char         = pd.read_excel('Batch Dryer Case Study.xlsx',sheet_name='ProcessCharacteristics')
initial_chem = pd.read_excel('Batch Dryer Case Study.xlsx',sheet_name='InitialChemistry')
cat          = pd.read_excel('Batch Dryer Case Study.xlsx',sheet_name='classifiers')

phibatch.plot_var_all_batches(bdata)
#%%
samples={'Deagglomerate':20,'Heat':30,'Cooldown':40};
bdata_aligned_phase=phibatch.phase_simple_align(bdata,samples)
phibatch.plot_var_all_batches(bdata_aligned_phase,
                  plot_title='simple alignment /phase',
                  phase_samples=samples)
#%%
mpca_obj=phibatch.mpca(bdata_aligned_phase,3,phase_samples=samples)
mpca_mon=phibatch.monitor(mpca_obj, bdata_aligned_phase)
mon_data =phibatch.monitor(mpca_obj, bdata_aligned_phase,which_batch='Batch 5')

#Show forecast of x

var='Dryer Temp'
point_in_time=20
batch2forecast='Batch 5'
forecast=mon_data[0]['forecast']
mdata=bdata_aligned_phase[bdata_aligned_phase['Batch number']==batch2forecast]

plt.figure()
f=forecast[point_in_time]
plt.plot(mdata[var][:point_in_time],'o',label='Measured')
aux=[np.nan]*point_in_time
aux.extend(f[var].values[point_in_time:].tolist())
plt.plot(np.array(aux),label='Forecast')
plt.plot(mdata[var][point_in_time:],'o',label='Known trajectory',alpha=0.3)
plt.xlabel('sample')
plt.ylabel(var)
plt.legend()
plt.title('Forecast for '+var+' at sample '+str(point_in_time)+' for '+batch2forecast )

#%%
mpls_obj=phibatch.mpls(bdata_aligned_phase,cqa,3,phase_samples=samples)
pp.r2pv(mpls_obj)
pp.score_scatter(mpls_obj, [1,2],CLASSID=cat,colorby='Quality')
mpls_obj=phibatch.monitor(mpls_obj, bdata_aligned_phase)
mon_data =phibatch.monitor(mpls_obj, bdata_aligned_phase,which_batch='Batch 5')


#%%
mpls_obj=phibatch.mpls(bdata_aligned_phase,cqa,5,zinit=initial_chem,
                       phase_samples=samples)
phibatch.loadings(mpls_obj,1)
pp.r2pv(mpls_obj)
mpls_obj=phibatch.monitor(mpls_obj, bdata_aligned_phase,zinit=initial_chem)
mon_data =phibatch.monitor(mpls_obj, bdata_aligned_phase,which_batch='Batch 5',zinit=initial_chem)

#%%

mpls_obj=phibatch.mpls(bdata_aligned_phase,cqa,5,zinit=initial_chem,
                      phase_samples=samples,mb_each_var='True',cross_val=5)
phibatch.loadings(mpls_obj,1)
pp.r2pv(mpls_obj)
pp.score_scatter(mpls_obj, [1,2],CLASSID=cat,colorby='Quality')
mpls_obj=phibatch.monitor(mpls_obj, bdata_aligned_phase,zinit=initial_chem)
mon_data =phibatch.monitor(mpls_obj, bdata_aligned_phase,which_batch='Batch 5',zinit=initial_chem)
