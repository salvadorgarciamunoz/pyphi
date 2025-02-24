# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 20:53:21 2022

@author: salva
"""

import pandas as pd
import numpy as np
import pyphi_batch as phibatch
import pyphi_plots as pp
import matplotlib.pyplot as plt

bdata=pd.read_excel('Batch Film Coating.xlsx')

#Plot variables for all batches
#plot all variables
phibatch.plot_var_all_batches(bdata)
#plot a variable
phibatch.plot_var_all_batches(bdata,which_var='INLET_AIR_TEMP')
#plot some variables
phibatch.plot_var_all_batches(bdata,which_var=['INLET_AIR_TEMP','EXHAUST_AIR_TEMP'])


#plot a variable for a single batch or groups of batches or contrast trajectories
#Plot single variable for single batch
phibatch.plot_batch(bdata, which_batch='B1805', which_var='INLET_AIR_TEMP')

#Plot single variable for single batch and for contrast add the same variable for the rest of the set
phibatch.plot_batch(bdata, which_batch='B1805', which_var='INLET_AIR_TEMP',include_set=True)

#Plot single variable for single batch and for contrast add the same variable for the rest of the set
#and include the mean trajectory calculated from the rest of the set (excluding the one being plotted)
phibatch.plot_batch(bdata, which_batch='B1805', which_var='INLET_AIR_TEMP',include_set=True,include_mean_exc=True)


#%% Simplistic alignment simply taking the same number of samples per batch    
bdata_aligned=phibatch.simple_align(bdata,250)
phibatch.plot_var_all_batches(bdata_aligned,
                   plot_title='With simple alignment')
#%%
#Count how many samples each batch has per phase
phibatch.phase_sampling_dist(bdata)
    
#%% Better alignment taking advantage of the phase information
samples_per_phase={'STARTUP':3, 'HEATING':20,'SPRAYING':40, 
                   'DRYING':40,'DISCHARGING':5}

bdata_aligned_phase=phibatch.phase_simple_align(bdata,samples_per_phase)
phibatch.plot_var_all_batches(bdata_aligned_phase,
                   plot_title='Batch data synchronized by phase',
                   phase_samples=samples_per_phase)



#%% Alignment using the temperature during heating as an Indicator Variable
# this in this example the sampler will take 35 samples from the inital value of Inlet Temp
# until it reaches 67 C
samples_per_phase={'STARTUP':3, 'HEATING':['INLET_AIR_TEMP',35,67],'SPRAYING':40, 
                   'DRYING':40,'DISCHARGING':5}

bdata_aligned_phase=phibatch.phase_iv_align(bdata,samples_per_phase)
phibatch.plot_var_all_batches(bdata_aligned_phase,
                   plot_title='Batch data synchronized by phase',
                   phase_samples=samples_per_phase)

#%% Staying with non IV alignment for the sake of the example

samples_per_phase={'STARTUP':3, 'HEATING':20,'SPRAYING':40, 
                   'DRYING':40,'DISCHARGING':5}
bdata_aligned_phase=phibatch.phase_simple_align(bdata,samples_per_phase)




#%% Build a model with all batches and use scores to understand spread
mpca_obj=phibatch.mpca(bdata_aligned_phase,2, phase_samples=samples_per_phase)
pp.score_scatter(mpca_obj,[1,2],add_labels=True,marker_size=10)
pp.diagnostics(mpca_obj)
phibatch.r2pv(mpca_obj,which_var='TOTAL_SPRAY_USED')
pp.r2pv(mpca_obj,plotwidth=1500)

#Suggestion: Use the "Indicator Variable" alignment approach and see how the
#score space changes

#%%
# Calculate the contributions to the sores for some batches on the outskits of the score space
#Contributions can be summarized by the summation of the abs value wrt time
phibatch.contributions(mpca_obj, bdata_aligned_phase, 'scores',to_obs=['B1905'],plot_title='Cont. to B1905')

#This also displays the dynamics of Contributions 
phibatch.contributions(mpca_obj, bdata_aligned_phase, 'scores',to_obs=['B1905'],dyn_conts=True,plot_title='Cont. to B1905')


#%% Plotting some of the raw variables identifued as large contributors
phibatch.plot_batch(bdata_aligned_phase,
            which_batch=['B1905'],
            which_var=['INLET_AIR_TEMP','EXHAUST_AIR_TEMP','INLET_AIR'],
            include_mean_exc=True,
            include_set=True,
            phase_samples=samples_per_phase)


#%% Take out unusual batches and fit model to normal data use scores and loadings to understand variations
#normal batches with no abnormal batches
noc_batch_data=bdata_aligned_phase[np.logical_and(bdata_aligned_phase['BATCH NUMBER']!='B1905', 
                                                    bdata_aligned_phase['BATCH NUMBER']!='B1805')]

#batches with deviations
dev_batch_data=bdata_aligned_phase[np.logical_or(bdata_aligned_phase['BATCH NUMBER']=='B1905', 
                                      bdata_aligned_phase['BATCH NUMBER']=='B1805')]

#Build a Multi-way PCA model with 2PC's and display some basic diagnostics
mpca_obj=phibatch.mpca(noc_batch_data,2, phase_samples=samples_per_phase,cross_val=5)
pp.score_scatter(mpca_obj,[1,2],add_labels=True)
pp.diagnostics(mpca_obj)
#%% Make some diagnostic plots to understand the data and the model
phibatch.r2pv(mpca_obj)
phibatch.loadings_abs_integral(mpca_obj)
phibatch.loadings(mpca_obj,2,which_var=['INLET_AIR_TEMP','EXHAUST_AIR_TEMP','TOTAL_SPRAY_USED'],
                  r2_weighted=True)
#%% PLot the raw variables
phibatch.plot_batch(bdata_aligned_phase,
            which_batch=['B1910','B2110'],
            which_var=['TOTAL_SPRAY_USED'],
            include_mean_exc=True,
            include_set=True,
            phase_samples=samples_per_phase,single_plot=True)

batch_list=['B1910','B1205','B2510','B2210','B1810','B2110']
phibatch.plot_batch(bdata_aligned_phase,
            which_batch=batch_list,
            which_var=['EXHAUST_AIR_TEMP'],
            include_mean_exc=True,
            include_set=True,
            phase_samples=samples_per_phase,single_plot=True)

phibatch.plot_batch(bdata_aligned_phase,
            which_batch=batch_list,
            which_var=['INLET_AIR_TEMP'],
            include_mean_exc=True,
            include_set=True,
            phase_samples=samples_per_phase,single_plot=True)

#%% Prepare the model for monitorig

phibatch.monitor(mpca_obj, noc_batch_data)
#%%Monitor  all normal batches and show diagnostics
all_batches=np.unique(noc_batch_data['BATCH NUMBER'].values.tolist())
mon_all_batches=phibatch.monitor(mpca_obj,noc_batch_data,
                                  which_batch=all_batches)

#%% Monitor batch 1905 and diagnose

mon_1905 = phibatch.monitor(mpca_obj,dev_batch_data,which_batch=['B1905'])

#contribution to instantaneous SPE at sample 17
sam_num = 5
plt.figure()
plt.bar(mon_1905['cont_spei'].columns ,mon_1905['cont_spei'].iloc[sam_num-1])
plt.xticks(rotation=90)
plt.ylabel('Contributions to i-SPE')
plt.title('Contributions to instantaneous up to sample #'+str(sam_num))
plt.tight_layout()


phibatch.plot_batch(bdata_aligned_phase,
            which_batch=['B1905'],
            which_var=['INLET_AIR_TEMP','EXHAUST_AIR_TEMP','INLET_AIR'],
            include_mean_exc=True,
            include_set=True,
            phase_samples=samples_per_phase)
#%% Prepare forecasting for batch B2510
batch2forecast='B2510'
mon_data=phibatch.monitor(mpca_obj,noc_batch_data,
                          which_batch=batch2forecast)
#%% Plot forecast for the inlet_temp at various points in time
var='INLET_AIR_TEMP'
point_in_time_=[5,20,50,80]
for point_in_time in point_in_time_:
    forecast=mon_data['forecast']
    mdata=noc_batch_data[noc_batch_data['BATCH NUMBER']==batch2forecast]
    x_axis=np.arange(mdata.shape[0])+1
    plt.figure()
    f=forecast[point_in_time-1]
    x_axis_=np.arange( len(mdata[var][:point_in_time-1]))+1
    plt.plot(x_axis_,mdata[var][:point_in_time-1],'o',label='Measured')
    aux=[np.nan]*point_in_time
    aux.extend(f[var].values[point_in_time:].tolist())
    plt.plot(x_axis,np.array(aux),label='Forecast')
    x_axis_=np.arange(len(mdata[var][point_in_time:] ))+point_in_time+1
    plt.plot(x_axis_,mdata[var][point_in_time:],'o',label='Known trajectory',alpha=0.3)
    plt.xlabel('sample')
    plt.ylabel(var)
    plt.legend()
    plt.title('Forecast for '+var+' at sample '+str(point_in_time)+' for '+batch2forecast )
    
        
    
    





    