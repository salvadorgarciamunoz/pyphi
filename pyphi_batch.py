# -*- coding: utf-8 -*-
"""


Created on Mon Apr 11 14:58:35 2022

Batch data is assumed to come in an excel file 
with first column being batch identifier and following columns
being process variables.
Optionally the second column labeled 'PHASE','Phase' or 'phase' indicating
the phase of exceution

Change log:
* added Dec 28 2023  Titles can be sent to contribution plots via plot_title flag    
                     Monitoring diagnostics are also plotted against sample starting with 1

* added Dec 27 2023  Corrected plots to use correct xaxis starting with sample =1 
                     Ammended indicator variable alignment not to replace the IV with a linear sequence but to keep orginal data
                     
* added Dec 4  2023  Added a BatchVIP calculation
* added Apr 23 2023  Corrected a very dumb mistake I made coding when tired    
    
* added Apr 18 2023 Added descriptors routine to obtain landmarks of the batch
                    such as min,max,ave of a variable [during a phase if indicated so]    
                    Modifed plot_var_all_batches to plot against the values in a Time column
                    and also add the legend for the BatchID
                    
* added Apr 10 2023 Added batch contribution plots
                    Added build_rel_time to create a tag of relative 
                    run time from a timestamp
                    
* added Apr 7 2023  Added alignment using indicator variable per phase


* added Apr 5 2023  Added the capability to monitor a variable in "Soft Sensor" mode
                    which implies there are no measurements for it (pure prediction)
                    as oppose to a forecast where there are new measurments coming in time.
                    
* added Jul 20 2022 Distribution of number of samples per phase plot
* added Aug 10 2022 refold_horizontal | clean_empty_rows | predict 
* added Aug 12 2022 replicate_batch

@author: S. Garcia-Munoz sgarciam@ic.ak.uk salg@andrew.cmu.edu
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pyphi as phi
# Sequence of color blind friendly colors.
cb_color_seq=['b','r','m','navy','bisque','silver','aqua','pink','gray']
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=cb_color_seq) 

def unique(df,colid):    
    ''' Replacement of the np.unique routine, specifically for dataframes
    
    unique(df,colid)
    
    Args:
        df:     A pandas dataframe
        colid:  Column identifier 
    
    Returns:
        A list with unique values in the order found in the dataframe
    
    by Salvador Garcia (sgarciam@ic.ac.uk)
    '''
    
    aux=df.drop_duplicates(subset=colid,keep='first')
    unique_entries=aux[colid].values.tolist()
    return unique_entries
    
def mean(X,axis):
    X_nan_map = np.isnan(X)
    X_ = X.copy()
    if X_nan_map.any():
        X_nan_map       = X_nan_map*1
        X_[X_nan_map==1] = 0
       
        #Calculate mean without accounting for NaN'
        if axis==0:
            aux             = np.sum(X_nan_map,axis=0)
            x_mean = np.sum(X_,axis=0,keepdims=1)/(np.ones((1,X_.shape[1]))*X_.shape[0]-aux)
        else:
            aux             = np.sum(X_nan_map,axis=1)
            x_mean = np.sum(X_,axis=1,keepdims=1)/(np.ones((X_.shape[0],1))*X_.shape[1]-aux)
    else:
        x_mean = np.mean(X_,axis=axis,keepdims=1)
    return x_mean.reshape(-1)

def simple_align(bdata,nsamples):
    ''' Simple alignment for bacth data using row number to linearly interpolate
        to the same number of samples 
    
    bdata_aligned= simple_align(bdata,nsamples)
    
    Args:
        bdata is a Pandas DataFrame where 1st column is Batch Identifier
              and following columns are variables, each row is a new
              time sample. Batches are concatenated vertically.
          
        nsamples is the new number of samples to generate per batch 
            irrespective of phase
     
    Returns:
        A pandas dataframe with batch data resampled to nsamples
        for all batches
    
    by Salvador Garcia Munoz (sgarciam@imperial.ac.uk, salvadorgarciamunoz@gmail.com)
    '''
    bdata_a=[]
    if (bdata.columns[1]=='PHASE') or \
        (bdata.columns[1]=='phase') or \
        (bdata.columns[1]=='Phase'):
        phase=True
    else:
        phase = False
    aux=bdata.drop_duplicates(subset=bdata.columns[0],keep='first')
    unique_batches=aux[aux.columns[0]].values.tolist()
    for b in unique_batches:
        data_=bdata[bdata[bdata.columns[0]]==b]
        indx=np.arange(data_.shape[0])
        new_indx=np.linspace(0,data_.shape[0]-1,nsamples)
        bname_rs=[data_[bdata.columns[0]].values[0]]*nsamples
        if phase:
            phase_list=data_[bdata.columns[1]].values.tolist()
            roundindx=np.round(new_indx).astype('int').tolist()
            phase_rs=[]
            for i in roundindx:
                phase_rs.append(phase_list[i])
            vals=data_.values[:,2:]
            cols=data_.columns[2:]
            
        else:
            vals=data_.values[:,1:]
            cols=data_.columns[1:]
        vals_rs=np.zeros((nsamples,vals.shape[1]))  
        for i in np.arange(vals.shape[1]):
            vals_rs[:,i]=np.interp(new_indx,indx.astype('float'),vals[:,i].astype('float'))
        
        df_=pd.DataFrame(vals_rs,columns=cols)
        if phase:
            df_.insert(0,bdata.columns[1],phase_rs)
        df_.insert(0,bdata.columns[0],bname_rs)
        bdata_a.append(df_)
    bdata_a=pd.concat(bdata_a)    
    return bdata_a            
            

def phase_simple_align(bdata,nsamples):
    ''' Simple batch alignment (0 to 100%) per phase
    
    bdata_aligned = phase_simple_align(bdata,nsamples)
    
    Args:
        
        bdata:  is a Pandas DataFrame where 1st column is Batch Identifier
              the second column is a phase indicator
              and following columns are variables, each row is a new
              time sample. Batches are concatenated vertically.
        
        nsamples: if integer: Number of samples to collect per phase
                            
                  if dictionary: samples to generate per phase e.g.
    
        nsamples = {'Heating':100,'Reaction':200,'Cooling':10}
        
        resampling is linear with respect to row number 
     
    Returns:
        a pandas dataframe with batch data resampled (aligned)
    
    by Salvador Garcia Munoz 
    (sgarciam@imperial.ac.uk , salvadorgarciamunoz@gmail.com)
    '''
    bdata_a=[]
    if (bdata.columns[1]=='PHASE') or \
        (bdata.columns[1]=='phase') or \
        (bdata.columns[1]=='Phase'):
        phase=True
    else:
        phase = False
    if phase:
        aux=bdata.drop_duplicates(subset=bdata.columns[0],keep='first')
        unique_batches=aux[aux.columns[0]].values.tolist()
        
        for b in unique_batches:
            data_=bdata[bdata[bdata.columns[0]]==b]
            vals_rs=[]
            bname_rs=[]
            phase_rs=[]
            firstone=True
            for p in nsamples.keys():
                p_data= data_[data_[data_.columns[1]]==p]
                samps=nsamples[p]
                
                indx=np.arange(p_data.shape[0])
                new_indx=np.linspace(0,p_data.shape[0]-1,samps)
                bname_rs_=[p_data[p_data.columns[0]].values[0]]*samps
                phase_rs_=[p_data[p_data.columns[1]].values[0]]*samps

                vals=p_data.values[:,2:]
                cols=p_data.columns[2:]
                vals_rs_=np.zeros((samps,vals.shape[1]))  
                for i in np.arange(vals.shape[1]):
                    vals_rs_[:,i]=np.interp(new_indx,indx.astype('float'),vals[:,i].astype('float'))
                if firstone:
                    vals_rs=vals_rs_
                    firstone=False
                else:
                    vals_rs=np.vstack((vals_rs,vals_rs_))
                bname_rs.extend(bname_rs_)
                phase_rs.extend(phase_rs_)
                 
            df_=pd.DataFrame(vals_rs,columns=cols)
            df_.insert(0,bdata.columns[1],phase_rs)
            df_.insert(0,bdata.columns[0],bname_rs)
            bdata_a.append(df_)
        bdata_a=pd.concat(bdata_a)    
        return bdata_a            
    
def phase_iv_align(bdata,nsamples):
    ''' Batch alignment using an indicator variable
    
    batch_aligned_data = phase_iv_align(bdata,nsamples)
    
    Args:
        
        bdata is a Pandas DataFrame where 1st column is Batch Identifier
              the second column is a phase indicator
              and following columns are variables, each row is a new
              time sample. Batches are concatenated vertically.
              
       nsamples:
                     
            if nsamples is a dictionary: samples to generate per phase e.g.
            
            nsamples = {'Heating':100,'Reaction':200,'Cooling':10}
            
            * If an indicator variable is used, with known start and end values
            indicate it with a list like this:
            
                [IVarID,num_samples,start_value,end_value]    
            
            example:
                
            nsamples = {'Heating':['TIC101',100,30,50],'Reaction':200,'Cooling':10}
            
            
                During the 'Heating' phase use TIC101 as an indicator variable
                take 100 samples equidistant from TIC101=30 to TIC101=50 
                and align against that variable as a measure of batch evolution (instead of time)
                
            * If an indicator variable is used, with unknown start but known end values
            indicate it with a list like this:
            
                [IVarID,num_samples,end_value]    
            
            example:
                
            nsamples = {'Heating':['TIC101',100,50],'Reaction':200,'Cooling':10}
            
            
                During the 'Heating' phase use TIC101 as an indicator variable
                take 100 samples equidistant from the value of TIC101 at the start of the phase
                to the point when TIC101=50 and align against that variable as a measure 
                of batch evolution (instead of time)    
                
                If no IV is sent, the resampling is linear with respect to row number per phase

             
    Returns:
            A pandas dataframe with batch data resampled (aligned)
    
    by Salvador Garcia Munoz
    sgarciam@imperial.ac.uk  salvadorgarciamunoz@gmail.com
    '''
    bdata_a=[]
    if (bdata.columns[1]=='PHASE') or \
        (bdata.columns[1]=='phase') or \
        (bdata.columns[1]=='Phase'):
        phase=True
    else:
        phase = False
    if phase:
        aux=bdata.drop_duplicates(subset=bdata.columns[0],keep='first')
        unique_batches=aux[aux.columns[0]].values.tolist()
        
        for b in unique_batches:
            data_=bdata[bdata[bdata.columns[0]]==b]
            vals_rs=[]
            bname_rs=[]
            phase_rs=[]
            firstone=True
            for p in nsamples.keys():
                p_data= data_[data_[data_.columns[1]]==p]
                samps=nsamples[p]
                
                if not(isinstance(samps,list)): #Linear alignment
                    indx=np.arange(p_data.shape[0])
                    new_indx=np.linspace(0,p_data.shape[0]-1,samps)
                    bname_rs_=[p_data[p_data.columns[0]].values[0]]*samps
                    phase_rs_=[p_data[p_data.columns[1]].values[0]]*samps
    
                    vals=p_data.values[:,2:]
                    cols=p_data.columns[2:]
                    vals_rs_=np.zeros((samps,vals.shape[1]))  
                    for i in np.arange(vals.shape[1]):
                        x_=indx.astype('float')
                        y_=vals[:,i].astype('float')
                        x_=x_[~np.isnan(y_)]
                        y_=y_[~np.isnan(y_)]                        
                        #vals_rs_[:,i]=np.interp(new_indx,indx.astype('float'),vals[:,i].astype('float'))
                        if len(y_)==0:
                            vals_rs_[:,i]=np.nan
                        else:
                            vals_rs_[:,i]=np.interp(new_indx,x_,y_,left=np.nan,right=np.nan)
                elif isinstance(samps,list): #align this phase with IV
                    if len(samps)==4:
                        iv_id=samps[0]
                        nsamp=samps[1]
                        start=samps[2]
                        end  =samps[3]
                        new_indx=np.linspace(start,end,nsamp)
                        iv = p_data[iv_id].values.astype(float)
                    if len(samps)==3    :
                        iv_id    = samps[0]
                        nsamp    = samps[1]
                        end      = samps[2]
                        
                        if (not( isinstance(end,int) or isinstance(end,float)) and (isinstance(end,dict))
                           and not(firstone)):
                            #end is  model to estimate the end value based on previous trajectory
                            cols=p_data.columns[2:].tolist()
                            fakebatch=pd.DataFrame(vals_rs,columns=cols)
                            fakebatch.insert(0,'phase',[p]*vals_rs.shape[0])
                            fakebatch.insert(0,'BatchID',[b]*vals_rs.shape[0])
                            preds=predict(fakebatch, end)
                            end_hat=preds['Yhat'][preds['Yhat'].columns[1]].values[0]                                     
                            iv = p_data[iv_id].values.astype(float)
                            end_hat = end_hat + iv[0]
                            new_indx=np.linspace(iv[0],end_hat,nsamp)                        
                        else:    
                            iv = p_data[iv_id].values.astype(float)
                            new_indx=np.linspace(iv[0],end,nsamp)                        
                    
                    cols=p_data.columns[2:].tolist()
                    #indx_2_iv=cols.index(iv_id)
                    bname_rs_=[p_data[p_data.columns[0]].values[0]]*nsamp
                    phase_rs_=[p_data[p_data.columns[1]].values[0]]*nsamp      
                    
                    #Check monotonicity
                    if (np.all(np.diff(iv)<0) or np.all(np.diff(iv)>0)):
                        indx=iv                                          
                        vals=p_data.values[:,2:]                              
                        #replace the IV with a linear variable (time surrogate)
                        #linear_indx=np.arange(vals.shape[0])
                        #vals[:,indx_2_iv]=linear_indx
                        vals_rs_=np.zeros((nsamp,vals.shape[1])) 
                        for i in np.arange(vals.shape[1]):
                            x_=indx.astype('float')
                            y_=vals[:,i].astype('float')
                            x_=x_[~np.isnan(y_)]
                            y_=y_[~np.isnan(y_)]                        
                            #vals_rs_[:,i]=np.interp(new_indx,indx.astype('float'),vals[:,i].astype('float'))
                            #vals_rs_[:,i]=np.interp(new_indx,indx                ,vals[:,i].astype('float'),left=np.nan,right=np.nan)
                            if len(y_)==0:
                                vals_rs_[:,i]=np.nan
                            else:                            
                                vals_rs_[:,i]=np.interp(new_indx,x_,y_,left=np.nan,right=np.nan)
                            
                        
                    else: #if monotonicity fails try to remove non-monotonic vals
                        print('Indicator variable '+iv_id+' for batch '+ b +' is not monotinic')
                        print('this is not ideal, maybe rethink your IV')
                        print('trying to remove non-monotonic samples')  
                        x=np.arange(0,len(iv))
                        m=(1/sum(x**2))*sum(x*(iv-iv[0]))
                      
                        #if iv[0]-iv[-1] > 0 :
                        if m<0:    
                            expected_direction = -1 #decreasing IV
                        else:
                            expected_direction = 1  #increasing IV                            
                        vals=p_data.values[:,2:]                        
                        new_vals=vals[0,:]
                        prev_iv=iv[0]
                        mon_iv=[iv[0]]
                        for i in np.arange(1,vals.shape[0]):
                            if (np.sign(iv[i] - prev_iv) * expected_direction)==1: #monotonic
                                new_vals=np.vstack((new_vals,vals[i,:]))
                                prev_iv=iv[i]
                                mon_iv.append(iv[i])                        
                        indx=np.array(mon_iv)    
                        vals=new_vals.copy()
                        #replace the IV with a linear variable (time surrogate)
                        #linear_indx=np.arange(vals.shape[0])
                        #vals[:,indx_2_iv]=linear_indx
                        vals_rs_=np.zeros((nsamp,vals.shape[1])) 
                        for i in np.arange(vals.shape[1]):
                            x_=indx.astype('float')
                            y_=vals[:,i].astype('float')
                            x_=x_[~np.isnan(y_)]
                            y_=y_[~np.isnan(y_)]                               
                            #vals_rs_[:,i]=np.interp(new_indx,indx,vals[:,i].astype('float'),left=np.nan,right=np.nan)    
                            if len(y_)==0:
                                vals_rs_[:,i]=np.nan
                            else:
                                vals_rs_[:,i]=np.interp(new_indx,x_,y_,left=np.nan,right=np.nan)                                                
                if firstone:
                    vals_rs=vals_rs_
                    firstone=False
                else:
                    vals_rs=np.vstack((vals_rs,vals_rs_))
                bname_rs.extend(bname_rs_)
                phase_rs.extend(phase_rs_)
                 
                
            df_=pd.DataFrame(vals_rs,columns=cols)
            df_.insert(0,bdata.columns[1],phase_rs)
            df_.insert(0,bdata.columns[0],bname_rs)
            bdata_a.append(df_)
        bdata_a=pd.concat(bdata_a)    
        return bdata_a    
    
def plot_var_all_batches(bdata,*,which_var=False,plot_title='',mkr_style='.-',
                         phase_samples=False,alpha_=0.2,timecolumn=False,lot_legend=False):
    '''Plotting routine for batch data plot data for all batches in a dataset
    
    plot_var_all_batches(bdata,*,which_var=False,plot_title='',mkr_style='.-',
                             phase_samples=False,alpha_=0.2,timecolumn=False,lot_legend=False):

    Args:
        bdata: Batch data organized as: column[0] = Batch Identifier column name is unrestricted
                                        column[1] = Phase information per sample must be called 'Phase','phase', or 'PHASE'
                                                    this information is optional
                                        column[2:]= Variables measured throughout the batch
                                        
                                        The data for each batch is one on top of the other in a vertical matrix
        which_var:     Which variables are to be plotted, if not sent, all are.
        plot_title:    Optional text to be used as the title of all figures
        phase_samples: information used to align the batch, so that phases are marked in the plot
        alpha_:        Transparency for the phase dividing line
        timecolumn:    Name of the column that indicates time, if given all data is plotted against time
        lot_legend:    Flag to add a legend for the batch identifiers
        
    by Salvador Garcia Munoz
    sgarciam@imperial.ac.uk salvadorgarciamunoz@gmail.com
    '''
    
    var_list=which_var
    if (bdata.columns[1]=='PHASE') or \
        (bdata.columns[1]=='phase') or \
        (bdata.columns[1]=='Phase'):
        if isinstance(var_list,bool):    
            var_list=bdata.columns[2:].tolist()
    else:
        if isinstance(var_list,bool):
            var_list=bdata.columns[1:].tolist()
    if isinstance(var_list,str) and not(isinstance(var_list,list)):
         var_list=[var_list]    
         
    for v in var_list:
        plt.figure()
        if not(isinstance(timecolumn,bool)):
            dat=bdata[[bdata.columns[0],timecolumn,v]]
        else:    
            dat=bdata[[bdata.columns[0],v]]
        for b in np.unique(dat[bdata.columns[0]]):
            data_=dat[v][dat[dat.columns[0]]==b]
            if not(isinstance(timecolumn,bool)):
                tim=dat[timecolumn][dat[dat.columns[0]]==b]
                if lot_legend:
                    plt.plot(tim.values,data_.values,mkr_style,label=b)
                else:    
                    plt.plot(tim.values,data_.values,mkr_style)
                plt.xlabel(timecolumn)                
            else:
                if lot_legend:
                    plt.plot(1+np.arange(len(data_.values)),data_.values,mkr_style,label=b)
                else:
                    plt.plot(1+np.arange(len(data_.values)),data_.values,mkr_style)
                plt.xlabel('Sample')
            plt.ylabel(v)
        if lot_legend:
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        if not(isinstance(phase_samples,bool)):   
            s_txt=1
            s_lin=1
            plt.axvline(x=1,color='magenta',alpha=alpha_)
            for p in phase_samples.keys():                
                if isinstance(phase_samples[p],list):
                    s_lin+=phase_samples[p][1]
                else:                    
                    s_lin+=phase_samples[p]
                plt.axvline(x=s_lin,color='magenta',alpha=alpha_)
                ylim_=plt.ylim()
                plt.annotate(p, (s_txt,ylim_[0]),rotation=90,alpha=0.5,color='magenta')
                if isinstance(phase_samples[p],list):
                    s_txt+=phase_samples[p][1]
                else:                                    
                    s_txt+=phase_samples[p]
        plt.title(plot_title)  
        plt.tight_layout()
        
def plot_batch(bdata,which_batch,which_var,*,include_mean_exc=False,include_set=False,
               phase_samples=False,single_plot=False,plot_title=''):
    '''Plotting routine for batch data
    
    plot_batch(bdata,which_batch,which_var,*,include_mean_exc=False,include_set=False,
                   phase_samples=False,single_plot=False,plot_title='')
    
    Args:  
        bdata: Batch data organized as: column[0] = Batch Identifier column name is unrestricted
                                        column[1] = Phase information per sample must be called 'Phase','phase', or 'PHASE'
                                                    this information is optional
                                        column[2:]= Variables measured throughout the batch
                                        
                                        The data for each batch is one on top of the other in a vertical matrix
        
        which_batch:      Which batches to plot
        which_var:        Which variables are to be plotted, if not sent, all are.       
        include_mean_exc: Include the mean trajectory of the set EXCLUDING the one batch being plotted
        include_set:      Include all other trajectories (will be colored in light gray)
        phase_samples:    Information used to align the batch, so that phases are marked in the plot
        single_plot:      If True => Plot everything in a single axis
        plot_title:       Optional text to be added to the title of all figures
        
        by Salvador Garcia Munoz
        sgarciam@imperial.ac.uk salvadorgarciamunoz@gmail.com
    '''

    if isinstance(which_batch,str) and not(isinstance(which_batch,list)):
        which_batch=[which_batch]
    if isinstance(which_var,str) and not(isinstance(which_var,list)):
        which_var=[which_var]    
    if single_plot:
        plt.figure()
        first_pass=True
    for b in which_batch:
        this_batch=bdata[bdata[bdata.columns[0]]==b]
        all_others=bdata[bdata[bdata.columns[0]]!=b]
        for v in which_var:
            this_var_this_batch=this_batch[v].values
            if not(single_plot):
                plt.figure()
            x_axis=np.arange(len(this_var_this_batch ))+1
            plt.plot(x_axis,this_var_this_batch,'k',label=b)
            if include_mean_exc or include_set:
                this_var_all_others=[]
                max_obs=0
                for bb in np.unique(all_others[bdata.columns[0]]):
                    this_var_all_others.append(all_others[v][all_others[bdata.columns[0]]==bb ].values)
                    if len(all_others[v][all_others[bdata.columns[0]]==bb ].values ) > max_obs:
                        max_obs = len(all_others[v][all_others[bdata.columns[0]]==bb ].values )
                this_var_all_others_=[]
                for bb in np.unique(all_others[bdata.columns[0]]):
                    _to_append=all_others[v][all_others[bdata.columns[0]]==bb ].values
                    this_len=len(_to_append)
                    _to_append=np.append(_to_append,np.tile(np.nan,max_obs-this_len))
                    this_var_all_others_.append(_to_append)
                this_var_all_others_=np.array(this_var_all_others_)    
                x_axis=np.arange(max_obs)+1
                if not(single_plot):
                    if include_set:
                        for t in this_var_all_others:
                            x_axis=np.arange(len(t))+1
                            plt.plot(x_axis,t,'m',alpha=0.1)
                        plt.plot(x_axis,this_var_all_others[-1],'m',alpha=0.1,label='rest of set')
                    if include_mean_exc:
                        plt.plot(x_axis,mean(this_var_all_others_,axis=0),'r',label='Mean without '+b )  
                else:
                    if first_pass:
                        if include_set:
                            plt.plot(x_axis,this_var_all_others_.T,'m',alpha=0.1)
                            plt.plot(x_axis,this_var_all_others_[0,:],'m',alpha=0.1,label='rest of set')
                        if include_mean_exc:
                            plt.plot(x_axis,mean(this_var_all_others_,axis=0),'r',label='Mean without '+b )
                        first_pass=False
            if not(isinstance(phase_samples,bool)):   
                s_txt=1
                s_lin=1
                plt.axvline(x=1,color='magenta',alpha=0.2)
                for p in phase_samples.keys():
                    if isinstance(phase_samples[p],list):
                        s_lin+=phase_samples[p][1]
                    else:                    
                        s_lin+=phase_samples[p]                   
                    plt.axvline(x=s_lin,color='magenta',alpha=0.2)
                    ylim_=plt.ylim()
                    plt.annotate(p, (s_txt,ylim_[0]),rotation=90,alpha=0.5,color='magenta')
                    if isinstance(phase_samples[p],list):
                        s_txt+=phase_samples[p][1]
                    else:                                    
                        s_txt+=phase_samples[p]     
            plt.title(b+' ' + plot_title)
            plt.xlabel('sample')
            plt.ylabel(v)
            plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0)
            plt.tight_layout()
        
def unfold_horizontal(bdata):
    if (bdata.columns[1]=='PHASE') or \
        (bdata.columns[1]=='phase') or \
        (bdata.columns[1]=='Phase'):
        phase=True
    else:
        phase = False

    firstone=True
    clbl=[]
    bid=[]
    aux=bdata.drop_duplicates(subset=bdata.columns[0],keep='first')
    unique_batches=aux[aux.columns[0]].values.tolist()
    for b in unique_batches:
        data_=bdata[bdata[bdata.columns[0]]==b]
        if phase:
            vals=data_.values[:,2:]
            cols=data_.columns[2:]
        else:
            vals=data_.values[:,1:]
            cols=data_.columns[1:]
        row_=[]
        firstone_c=True
        for c in np.arange(vals.shape[1]):
            
           r_= vals[:,[c]].reshape(1,-1)
           if firstone:
               for i in np.arange(1,r_.shape[1]+1):
                   clbl.append(cols[c]+'_'+str(i))
                   bid.append(cols[c] )
           if firstone_c:
               row_=r_
               firstone_c=False
           else:
               row_=np.hstack((row_,r_))
        if firstone:
           bdata_hor=row_    
           firstone=False
        else:
           bdata_hor=np.vstack((bdata_hor,row_))     
    bdata_hor= pd.DataFrame(bdata_hor,columns=clbl)
    bdata_hor.insert(0, bdata.columns[0],unique_batches)
    return bdata_hor,clbl,bid
            
def refold_horizontal(xuf,nvars,nsamples):
    #xuf is strictly numberical
    Xb=[]
    for i in np.arange(xuf.shape[0]):
        r=xuf[i,:]
        for v in np.arange(nvars):
            var=r[:nsamples]
            var=var.reshape(-1,1)
            if v<(nvars-1):
                r=r[nsamples:]
            if v==0:
                batch=var
            else:
                batch=np.hstack((batch,var))
        if i==0:
            Xb=batch
        else:
            Xb=np.vstack((Xb,batch))
    return Xb       
 
def _uf_l(L,spb,vpb):
    first_=True
    for i in np.arange(L.shape[1]):
        col_=L[:,[i]]
        s_=np.arange( spb )
        first_flag=True
        for v in np.arange(vpb):
            s_2use=(s_ + v*spb)
            if first_flag:
                col_rearranged=col_[s_2use]
                first_flag=False
            else:
                col_rearranged = np.hstack((col_rearranged,col_[s_2use]))
        col_rearranged = col_rearranged.reshape(1,-1)
        if first_:
            L_uf_hor_mon = col_rearranged.T
            first_       = False
        else:
            L_uf_hor_mon = np.hstack((L_uf_hor_mon,col_rearranged.T ))
    return L_uf_hor_mon

def _uf_hor_mon_loadings(mvmobj):
    spb = mvmobj['nsamples']
    vpb = mvmobj['nvars']
    is_pls=False
    if 'Q' in mvmobj:
        is_pls=True
        ninit=mvmobj['ninit']
    
    if is_pls:
        if ninit > 0:
            z_ws   = mvmobj['Ws'][:ninit,:]
            z_w    = mvmobj['W'][:ninit,:]
            z_p    = mvmobj['P'][:ninit,:]
            z_mx   = mvmobj['mx'][:ninit]
            z_sx   = mvmobj['sx'][:ninit]
            
            x_ws   = mvmobj['Ws'][ninit:,:]
            x_w    = mvmobj['W'][ninit:,:]
            x_p    = mvmobj['P'][ninit:,:]
            x_mx   = mvmobj['mx'][ninit:]
            x_sx   = mvmobj['sx'][ninit:]
            Ws_ufm = np.vstack(( z_ws,_uf_l(x_ws,spb,vpb)))
            W_ufm  = np.vstack(( z_w ,_uf_l(x_w ,spb,vpb)))
            P_ufm  = np.vstack(( z_p ,_uf_l(x_p,spb,vpb)))
            mx_ufm = np.vstack(( z_mx.reshape(-1,1),_uf_l( x_mx.reshape(-1,1),spb,vpb ))).reshape(-1)
            sx_ufm = np.vstack(( z_sx.reshape(-1,1),_uf_l( x_sx.reshape(-1,1),spb,vpb ))).reshape(-1)
        else:  
            Ws_ufm = _uf_l(mvmobj['Ws'],spb,vpb)
            W_ufm  = _uf_l(mvmobj['W'],spb,vpb)
            P_ufm  = _uf_l(mvmobj['P'],spb,vpb)
            mx_ufm = _uf_l( mvmobj['mx'].reshape(-1,1),spb,vpb ).reshape(-1)
            sx_ufm = _uf_l( mvmobj['sx'].reshape(-1,1),spb,vpb ).reshape(-1)
    else:
        P_ufm  =  _uf_l(mvmobj['P'],spb,vpb)
        mx_ufm = _uf_l( mvmobj['mx'].reshape(-1,1),spb,vpb ).reshape(-1)
        sx_ufm = _uf_l( mvmobj['sx'].reshape(-1,1),spb,vpb ).reshape(-1)
    
    if is_pls:
        mvmobj['Ws_ufm'] = Ws_ufm
        mvmobj['W_ufm']  = W_ufm
        mvmobj['P_ufm']  = P_ufm
    else:
        mvmobj['P_ufm']  = P_ufm
    
    mvmobj['mx_ufm']=mx_ufm
    mvmobj['sx_ufm']=sx_ufm
    
    return mvmobj

def loadings(mmvm_obj,dim,*,r2_weighted=False,which_var=False):
    ''' Plot batch loadings for variables as a function of time/sample
    
    loadings(mmvm_obj,dim,*,r2_weighted=False,which_var=False)
    
    Args:
        mmvm_obj:    Multiway PCA or PLS object
        dim:         What component or latent variable to plot
        r2_weighted: If True => weight the loading by the R2pv
        which_var:   Variable for which the plot is done, if not sent all are plotted
        
    by Salvador Garcia Munoz
    sgarciam@imperial.ac.uk salvadorgarciamunoz@gmail.com
    '''
    dim=dim-1
    
    if 'Q' in mmvm_obj:
        if mmvm_obj['ninit']==0:
            if r2_weighted:
                aux_df=pd.DataFrame(mmvm_obj['Ws']*mmvm_obj['r2xpv'])
            else:
                aux_df=pd.DataFrame(mmvm_obj['Ws'])
            
            aux_df.insert(0,'bid',mmvm_obj['bid'])
            if isinstance(which_var,bool):
                vars_to_plot=unique(aux_df,'bid')
            else:
                if isinstance(which_var,str):
                    vars_to_plot=[which_var]
                if isinstance(which_var,list):
                    vars_to_plot=which_var
            for i,v in enumerate(vars_to_plot):
                plt.figure()
                dat=aux_df[dim][aux_df['bid']==v].values
            
                plt.fill_between(np.arange(mmvm_obj['nsamples'])+1, dat )
                plt.xlabel('sample')
                if r2_weighted:
                    plt.ylabel('$W^* * R^2$ ['+str(dim+1)+']')
                else:
                    plt.ylabel('$W^*$ ['+str(dim+1)+']')
                plt.title(v)
                
                plt.ylim(mmvm_obj['Ws'][:,dim].min()*1.2,mmvm_obj['Ws'][:,dim].max()*1.2 )
                ylim_=plt.ylim()
                phase_samples=mmvm_obj['phase_samples']
                if not(isinstance(phase_samples,bool)):   
                    s_txt=1
                    s_lin=1
                    plt.axvline(x=1,color='magenta',alpha=0.2)
                    for p in phase_samples.keys():
                        if isinstance(phase_samples[p],list):
                            s_lin+=phase_samples[p][1]
                        else:                    
                            s_lin+=phase_samples[p] 
                        plt.axvline(x=s_lin,color='magenta',alpha=0.2)
                       # ylim_=[mmvm_obj['Ws'][dim,:].min()*1.2,mmvm_obj['Ws'][dim,:].max()*1.2  ]
                        plt.annotate(p, (s_txt,ylim_[0]),rotation=90,alpha=0.5,color='magenta')
                        if isinstance(phase_samples[p],list):
                            s_txt+=phase_samples[p][1]
                        else:                                    
                            s_txt+=phase_samples[p]
                plt.tight_layout()   
        else:
            z_loadings= mmvm_obj['Ws'] [np.arange(mmvm_obj['ninit'])]
            r2pvz     = mmvm_obj['r2xpv'] [np.arange(mmvm_obj['ninit']),:]
            if r2_weighted:
                z_loadings = z_loadings *r2pvz
                
            zvars     = mmvm_obj['varidX'][0:mmvm_obj['ninit']]
            plt.figure()
            plt.bar(zvars,z_loadings[:,dim] )
            plt.xticks(rotation=90)
            if r2_weighted:
                plt.ylabel('$W^* * R^2$ ['+str(dim+1)+']')
            else:
                plt.ylabel('$W^*$ ['+str(dim+1)+']')
            plt.title('Loadings for Initial Conditions')
            plt.tight_layout()
            
            rows_=np.arange( mmvm_obj['nsamples']*mmvm_obj['nvars'])+mmvm_obj['ninit']
            if r2_weighted:
                aux_df=pd.DataFrame(mmvm_obj['Ws'][rows_,:]*mmvm_obj['r2xpv'][rows_,:] )
            else:
                aux_df=pd.DataFrame(mmvm_obj['Ws'][rows_,:] )
            aux_df.insert(0,'bid',mmvm_obj['bid'])
            if isinstance(which_var,bool):
                vars_to_plot=unique(aux_df,'bid')
            else:
                if isinstance(which_var,str):
                    vars_to_plot=[which_var]
                if isinstance(which_var,list):
                    vars_to_plot=which_var
                    
            for i,v in enumerate(vars_to_plot):
                plt.figure()
                dat=aux_df[dim][aux_df['bid']==v].values
            
                plt.fill_between(np.arange(mmvm_obj['nsamples'])+1, dat )
                plt.xlabel('sample')
                if r2_weighted:
                    plt.ylabel('$W^* * R^2$ ['+str(dim+1)+']')
                else:
                    plt.ylabel('$W^*$ ['+str(dim+1)+']')
                plt.title(v)
                
                plt.ylim(mmvm_obj['Ws'][:,dim].min()*1.2,mmvm_obj['Ws'][:,dim].max()*1.2 )
                ylim_=plt.ylim()
                phase_samples=mmvm_obj['phase_samples']
                if not(isinstance(phase_samples,bool)):   
                    s_txt=1
                    s_lin=1
                    plt.axvline(x=1,color='magenta',alpha=0.2)
                    for p in phase_samples.keys():
                        if isinstance(phase_samples[p],list):
                            s_lin+=phase_samples[p][1]
                        else:                    
                            s_lin+=phase_samples[p] 
                        plt.axvline(x=s_lin,color='magenta',alpha=0.2)
                       # ylim_=[mmvm_obj['Ws'][dim,:].min()*1.2,mmvm_obj['Ws'][dim,:].max()*1.2  ]
                        plt.annotate(p, (s_txt,ylim_[0]),rotation=90,alpha=0.5,color='magenta')
                        if isinstance(phase_samples[p],list):
                            s_txt+=phase_samples[p][1]
                        else:                                    
                            s_txt+=phase_samples[p]
                plt.tight_layout()            
    else:
        if r2_weighted:
            aux_df=pd.DataFrame(mmvm_obj['P']*mmvm_obj['r2xpv'])
        else:
            aux_df=pd.DataFrame(mmvm_obj['P'])
            
        aux_df.insert(0,'bid',mmvm_obj['bid'])
        if isinstance(which_var,bool):
            vars_to_plot=unique(aux_df,'bid')
        else:
            if isinstance(which_var,str):
                vars_to_plot=[which_var]
            if isinstance(which_var,list):
                vars_to_plot=which_var
        for i,v in enumerate(vars_to_plot):
            plt.figure()
            dat=aux_df[dim][aux_df['bid']==v].values
            
            plt.fill_between(np.arange(mmvm_obj['nsamples'])+1, dat )
            plt.xlabel('sample')
            if r2_weighted:
                plt.ylabel('P * $R^2$ ['+str(dim+1)+']')
            else:
                plt.ylabel('P ['+str(dim+1)+']')
            plt.title(v)
            
            plt.ylim(mmvm_obj['P'][:,dim].min()*1.2,mmvm_obj['P'][:,dim].max()*1.2 )
            ylim_=plt.ylim()
            phase_samples=mmvm_obj['phase_samples']
            if not(isinstance(phase_samples,bool)):   
                s_txt=1
                s_lin=1
                plt.axvline(x=1,color='magenta',alpha=0.2)
                for p in phase_samples.keys():
                    if isinstance(phase_samples[p],list):
                        s_lin+=phase_samples[p][1]
                    else:                    
                        s_lin+=phase_samples[p] 
                    plt.axvline(x=s_lin,color='magenta',alpha=0.2)
                    #ylim_=[mmvm_obj['P'][dim,:].min()*1.2,mmvm_obj['P'][dim,:].max()*1.2 ]
                    plt.annotate(p, (s_txt,ylim_[0]),rotation=90,alpha=0.5,color='magenta')
                    if isinstance(phase_samples[p],list):
                        s_txt+=phase_samples[p][1]
                    else:                                    
                        s_txt+=phase_samples[p]
            plt.tight_layout()   
            
def loadings_abs_integral(mmvm_obj,*,r2_weighted=False,addtitle=False):
    '''Plot the integral of the absolute value of loadings for a batch
    
    loadings_abs_integral(mmvm_obj,*,r2_weighted=False,addtitle=False)
    
    Args:
        mmvm_obj: A multiway PCA or PLS model
        r2_weighted: Boolean flag, if True then in weights the loading by the R2pv
        addtitle: Text to place in the title of the figure
        
    by Salvador Garcia Munoz
    sgarciam@imperial.ac.uk salvadorgarciamunoz@gmail.com
    '''

    if 'Q' in mmvm_obj:
        if mmvm_obj['ninit']==0:
            if r2_weighted:
                aux_df=pd.DataFrame(mmvm_obj['Ws']*mmvm_obj['r2xpv'])
            else:
                aux_df=pd.DataFrame(mmvm_obj['Ws'])
            
            aux_df.insert(0,'bid',mmvm_obj['bid'])
        else:
            rows_=np.arange( mmvm_obj['nsamples']*mmvm_obj['nvars'])+mmvm_obj['ninit']
            if r2_weighted:
                aux_df=pd.DataFrame(mmvm_obj['Ws'][rows_,:]*mmvm_obj['r2xpv'][rows_,:] )
            else:
                aux_df=pd.DataFrame(mmvm_obj['Ws'][rows_,:] )
            aux_df.insert(0,'bid',mmvm_obj['bid'])
            
        integral_of_loadings=[]
        aux_vname=[]
        for i,v in enumerate(unique(aux_df,'bid')):                
            dat=aux_df[aux_df['bid']==v].values[:,1:]
            integral_of_loadings.append(np.sum(np.abs(dat),axis=0,keepdims=True)[0])
            aux_vname.append(v)
        integral_of_loadings=np.array(integral_of_loadings)
        
        for a in np.arange(mmvm_obj['T'].shape[1]):
            plt.figure()
            plt.bar(aux_vname,integral_of_loadings[:,a])
            if r2_weighted:
                plt.ylabel(r'$\sum (|W^*| * R^2$ ['+str(a+1)+'])')
            else:
                plt.ylabel(r'$\sum (|W^*|$ ['+str(a+1)+'])')
            plt.xticks(rotation=75)
            if not(isinstance(addtitle,bool)) and isinstance(addtitle,str):
                plt.title(addtitle)
            plt.tight_layout()   
                
    else:
        if r2_weighted:
            aux_df=pd.DataFrame(mmvm_obj['P']*mmvm_obj['r2xpv'])
        else:
            aux_df=pd.DataFrame(mmvm_obj['P'])
            
        aux_df.insert(0,'bid',mmvm_obj['bid'])
       
        integral_of_loadings=[]
        aux_vname=[]
        for i,v in enumerate(unique(aux_df,'bid')):                  
            dat=aux_df[aux_df['bid']==v].values[:,1:]
            integral_of_loadings.append(np.sum(np.abs(dat),axis=0,keepdims=True)[0])
            aux_vname.append(v)
        integral_of_loadings=np.array(integral_of_loadings)
        
        for a in np.arange(mmvm_obj['T'].shape[1]):
            plt.figure()
            plt.bar(aux_vname,integral_of_loadings[:,a])
            if r2_weighted:
                plt.ylabel(r'$\sum (|P| * R^2$ ['+str(a+1)+'])')
            else:
                plt.ylabel(r'$\sum$ (|P| ['+str(a+1)+'])')
            plt.xticks(rotation=75)
            if not(isinstance(addtitle,bool)) and isinstance(addtitle,str):
                plt.title(addtitle)
            plt.tight_layout()   

def batch_vip(mmvm_obj,*,addtitle=False):
    ''' plot the summation across componets of the integral of the absolute value of loadings for a batch
    multiplied by the R2 [which kinda mimicks the VIP]
    
    batch_vip(mmvm_obj,*,addtitle=False)
    
    Args:
        mmvm_obj: A multiway PCA or PLS model
        
    by Salvador Garcia Munoz
    sgarciam@imperial.ac.uk salvadorgarciamunoz@gmail.com
    '''
    r2_weighted=True
    if 'Q' in mmvm_obj:
        if mmvm_obj['ninit']==0:
            if r2_weighted:
                aux_df=pd.DataFrame(mmvm_obj['Ws']*np.tile(mmvm_obj['r2y'],(mmvm_obj['Ws'].shape[0],1)))
            else:
                aux_df=pd.DataFrame(mmvm_obj['Ws'])
            
            aux_df.insert(0,'bid',mmvm_obj['bid'])
        else:
            rows_=np.arange( mmvm_obj['nsamples']*mmvm_obj['nvars'])+mmvm_obj['ninit']
            if r2_weighted:
                aux_df=pd.DataFrame(mmvm_obj['Ws'][rows_,:]*mmvm_obj['r2xpv'][rows_,:] )
            else:
                aux_df=pd.DataFrame(mmvm_obj['Ws'][rows_,:] )
            aux_df.insert(0,'bid',mmvm_obj['bid'])
            
        integral_of_loadings=[]
        aux_vname=[]
        for i,v in enumerate(unique(aux_df,'bid')):                
            dat=aux_df[aux_df['bid']==v].values[:,1:]
            integral_of_loadings.append(np.sum(np.abs(dat),axis=0,keepdims=True)[0])
            aux_vname.append(v)
        integral_of_loadings=np.array(integral_of_loadings)
        plt.figure()
        plt.bar(aux_vname,np.sum(integral_of_loadings,axis=1))
        plt.ylabel('Batch VIP')
        plt.xticks(rotation=75)
        if not(isinstance(addtitle,bool)) and isinstance(addtitle,str):
            plt.title(addtitle)
        plt.tight_layout()   
                
    else:
        if r2_weighted:
            aux_df=pd.DataFrame(mmvm_obj['P']*mmvm_obj['r2xpv'])
        else:
            aux_df=pd.DataFrame(mmvm_obj['P'])
            
        aux_df.insert(0,'bid',mmvm_obj['bid'])
       
        integral_of_loadings=[]
        aux_vname=[]
        for i,v in enumerate(unique(aux_df,'bid')):                  
            dat=aux_df[aux_df['bid']==v].values[:,1:]
            integral_of_loadings.append(np.sum(np.abs(dat),axis=0,keepdims=True)[0])
            aux_vname.append(v)
        integral_of_loadings=np.array(integral_of_loadings)
        

        plt.figure()
        plt.bar(aux_vname,np.sum(integral_of_loadings,axis=1))
        
        plt.ylabel(r'$\sum_{a} \sum (|P| * R^2$')

        plt.xticks(rotation=75)
        if not(isinstance(addtitle,bool)) and isinstance(addtitle,str):
            plt.title(addtitle)
        plt.tight_layout()   
            

def r2pv(mmvm_obj,*,which_var=False):
    ''' Plot batch r2 for variables as a function of time/sample
    
    r2pv(mmvm_obj,*,which_var=False)
    
    Args:
        mmvm_obj:    Multiway PCA or PLS object
        which_var:   Variable for which the plot is done, if not sent all are plotted
        
    by Salvador Garcia Munoz
    sgarciam@imperial.ac.uk salvadorgarciamunoz@gmail.com
    '''
    
    if mmvm_obj['ninit']==0:
        aux_df=pd.DataFrame(mmvm_obj['r2xpv'])
        aux_df.insert(0,'bid',mmvm_obj['bid'])
        if isinstance(which_var,bool):
            vars_to_plot=unique(aux_df,'bid')
        else:
            if isinstance(which_var,str):
                vars_to_plot=[which_var]
            if isinstance(which_var,list):
                vars_to_plot=which_var
        for i,v in enumerate(vars_to_plot):
            
            dat=aux_df[aux_df['bid']==v].values*100
            dat=dat[:,1:].astype(float)               
            dat=np.cumsum(dat,axis=1)
            dat=np.hstack((np.zeros((dat.shape[0],1)) ,dat))
            plt.figure()
            for a in np.arange(mmvm_obj['A'])+1:
                plt.fill_between(np.arange(mmvm_obj['nsamples'])+1, dat[:,a],dat[:,a-1],label='LV #'+str(a) )
            plt.xlabel('sample')
            plt.ylabel('$R^2$pvX (%)')
            plt.legend()
            plt.title(v)                
            plt.ylim(0,100)
            ylim_=plt.ylim()
            phase_samples=mmvm_obj['phase_samples']
            if not(isinstance(phase_samples,bool)):   
                 s_txt=1
                 s_lin=1
                 plt.axvline(x=1,color='black',alpha=0.2)
                 for p in phase_samples.keys():
                     if isinstance(phase_samples[p],list):
                         s_lin+=phase_samples[p][1]
                     else:                    
                         s_lin+=phase_samples[p] 
                     plt.axvline(x=s_lin,color='black',alpha=0.2)
                     plt.annotate(p, (s_txt,ylim_[0]),rotation=90,alpha=0.5,color='black')
                     if isinstance(phase_samples[p],list):
                         s_txt+=phase_samples[p][1]
                     else:                                    
                         s_txt+=phase_samples[p]
            plt.tight_layout()   
        if 'Q' in mmvm_obj:    
            r2pvy = mmvm_obj['r2ypv']*100    
            #yvars=mmvm_obj['varidY']
            lbls=[]
            for a in np.arange(1,mmvm_obj['A']+1):
                lbls.append('LV #'+str(a))
                
            r2pvy_pd=pd.DataFrame(r2pvy,index=mmvm_obj['varidY'],columns=lbls)
            fig1,ax1=plt.subplots()
            r2pvy_pd.plot(kind='bar', stacked=True,ax=ax1)            
            #ax.set_xticks(rotation=90)
            ax1.set_ylabel('$R^2$pvY')
            ax1.set_title('$R^2$ per LV for Y-Space')
            fig1.tight_layout()
        
    else:
        r2pvz     = mmvm_obj['r2xpv'] [np.arange(mmvm_obj['ninit']),:]*100
        zvars     = mmvm_obj['varidX'][0:mmvm_obj['ninit']]
        lbls=[]
        for a in np.arange(1,mmvm_obj['A']+1):
            lbls.append('LV #'+str(a))
        r2pvz_pd=pd.DataFrame(r2pvz,index=zvars,columns=lbls)
        fig2,ax2=plt.subplots()
        r2pvz_pd.plot(kind='bar', stacked=True,ax=ax2)                    
        ax2.set_ylabel('$R^2$pvZ')
        ax2.set_title('$R^2$ Initial Conditions')
        fig2.tight_layout()
        
        rows_=np.arange( mmvm_obj['nsamples']*mmvm_obj['nvars'])+mmvm_obj['ninit']
        aux_df=pd.DataFrame(mmvm_obj['r2xpv'][rows_,:] )
        aux_df.insert(0,'bid',mmvm_obj['bid'])
        if isinstance(which_var,bool):
            vars_to_plot=unique(aux_df,'bid')
        else:
            if isinstance(which_var,str):
                vars_to_plot=[which_var]
            if isinstance(which_var,list):
                vars_to_plot=which_var
                
        for i,v in enumerate(vars_to_plot):
            
            dat=aux_df[aux_df['bid']==v].values*100
            dat=dat[:,1:].astype(float)               
            dat=np.cumsum(dat,axis=1)
            dat=np.hstack((np.zeros((dat.shape[0],1)) ,dat))
            plt.figure()
            for a in np.arange(mmvm_obj['A'])+1:
                plt.fill_between(np.arange(mmvm_obj['nsamples'])+1, dat[:,a],dat[:,a-1],label='LV #'+str(a) )
            plt.xlabel('sample')
            plt.ylabel('$R^2$pvX (%)')
            plt.legend()
            plt.title(v)                
            plt.ylim(0,100)
            ylim_=plt.ylim()
            phase_samples=mmvm_obj['phase_samples']
            if not(isinstance(phase_samples,bool)):   
                 s_txt=1
                 s_lin=1
                 plt.axvline(x=1,color='black',alpha=0.2)
                 for p in phase_samples.keys():
                     if isinstance(phase_samples[p],list):
                         s_lin+=phase_samples[p][1]
                     else:                    
                         s_lin+=phase_samples[p] 
                     plt.axvline(x=s_lin,color='black',alpha=0.2)
                     plt.annotate(p, (s_txt,ylim_[0]),rotation=90,alpha=0.5,color='black')
                     if isinstance(phase_samples[p],list):
                         s_txt+=phase_samples[p][1]
                     else:                                    
                         s_txt+=phase_samples[p]
            plt.tight_layout()   
            
        if 'Q' in mmvm_obj:
            r2pvy = mmvm_obj['r2ypv']*100    
            #yvars=mmvm_obj['varidY']
            lbls=[]
            for a in np.arange(1,mmvm_obj['A']+1):
                lbls.append('LV #'+str(a))
                
            r2pvy_pd=pd.DataFrame(r2pvy,index=mmvm_obj['varidY'],columns=lbls)
            fig1,ax1=plt.subplots()
            r2pvy_pd.plot(kind='bar', stacked=True,ax=ax1)            
            ax1.set_ylabel('$R^2$pvY')
            ax1.set_title('$R^2$ per LV for Y-Space')
            fig1.tight_layout()            
    
def mpca(xbatch,a,*,unfolding='batch wise',phase_samples=False,cross_val=0):
    ''' Multi-way PCA for batch analysis
    
    mpca_obj= mpca(xbatch,a,*,unfolding='batch wise',phase_samples=False,cross_val=0)
    
    Args:
        xbatch: Pandas dataframe with aligned batch data it is assumed 
                 that all batches have the same number of samples
                
        a:      Number of PC's to fit
        
        unfolding: 'batch wise' or 'variable wise'
        
        phase_samples: information about samples per phase [optional]
        cross_val: percent of elements for cross validation (defult is 0 = no cross val)

    Returns:
        mpca_obj: A Dictionary with all the parameters for the MPCA model
        
    by Salvador Garcia Munoz
    sgarciam@imperial.ac.uk salvadorgarciamunoz@gmail.com
    '''    
    if (xbatch.columns[1]=='PHASE') or \
        (xbatch.columns[1]=='phase') or \
        (xbatch.columns[1]=='Phase'):
        nvars = xbatch.shape[1]-2
    else:
        nvars = xbatch.shape[1]-1
    nbatches = len(np.unique(xbatch[xbatch.columns[0]]))    
    nsamples = xbatch.shape[0]/nbatches
    
    if unfolding=='batch wise':        
        # remove low variance columns keeping record of the original order
        x_uf_,colnames,bid_o = unfold_horizontal(xbatch)  # colnames is original set of columns
        x_uf,colsrem         = phi.clean_low_variances(x_uf_,shush=True) # colsrem are columns removed
        mx_rem=x_uf_[colsrem].mean().tolist()
        mx_rem=np.array(mx_rem)
        mpca_obj=phi.pca(x_uf,a,cross_val=cross_val)    
        if len(colsrem)>0:
            xtra_col    = np.zeros((2+2*a,1 ))    
            xtra_col[0] = 1
            xtra_cols   = np.tile(xtra_col,(1,len(colsrem))) 
            xtra_cols[1,:] = mx_rem
            aux         = np.vstack((mpca_obj['sx'],mpca_obj['mx'],mpca_obj['P'].T,mpca_obj['r2xpv'].T))
            aux         = np.hstack((aux,xtra_cols))
            all_cols    = x_uf.columns[1:].tolist()
            all_cols.extend(colsrem)
            aux_pd     = pd.DataFrame(aux,columns=all_cols)
            aux_pd     = aux_pd[colnames]
            aux_new    = aux_pd.values
            
            sx_               = aux_new[0,:].reshape(1,-1)
            mpca_obj['sx']    = sx_
            mx_               = aux_new[1,:].reshape(1,-1)
            mpca_obj['mx']    = mx_
            aux_new           = aux_new[2:,:]
            p_                = aux_new[0:a,:]
            mpca_obj['P']     = p_.T 
            r2xpv_            = aux_new[a:,:]
            mpca_obj['r2xpv'] = r2xpv_.T
        mpca_obj['varidX']= colnames
        mpca_obj['bid']   = bid_o
        mpca_obj['uf']    ='batch wise'
        mpca_obj['phase_samples'] = phase_samples
        mpca_obj['nvars']    = int(nvars)
        mpca_obj['nbatches'] = int(nbatches)
        mpca_obj['nsamples'] = int(nsamples)
        mpca_obj['ninit']    = 0
        mpca_obj['A']        = a
            
    elif unfolding=='variable wise':
       
        if  (xbatch.columns[1]=='PHASE') or \
            (xbatch.columns[1]=='phase') or \
            (xbatch.columns[1]=='Phase'):    
             xbatch_=xbatch.copy()
             xbatch_.drop(xbatch.columns[1],axis=1,inplace=True)
        else:
             xbatch_=xbatch.copy()
        xbatch_,colsrem = phi.clean_low_variances(xbatch_) # colsrem are columns removed
        xbatch_ = phi.clean_empty_rows(xbatch_)
        mpca_obj=phi.pca(xbatch_,a)        
        mpca_obj['uf']            ='variable wise'
        mpca_obj['phase_samples'] = phase_samples
        mpca_obj['nvars']    = nvars
        mpca_obj['nbatches'] = nbatches
        mpca_obj['nsamples'] = nsamples
        mpca_obj['ninit']    = 0
        mpca_obj['A']        = a
    else:
        mpca_obj=[]
    return mpca_obj
            
def _mimic_monitoring(mmvm_obj_f,bdata,which_batch,*,zinit=False,shush=False,soft_sensor=False):
    if (not(isinstance(zinit,bool)) and mmvm_obj_f['ninit']==0) |   \
        ((isinstance(zinit,bool)) and mmvm_obj_f['ninit']>0):
        print ('Model and data do not correspond')
    else:
        T_mon     = []
        ht2_mon   = []
        spe_mon   = []
        spei_mon  = []
        cont_spei = []
        cont_spe  = []
        cont_ht2  = []
        forecast  = []
        if 'Q' in mmvm_obj_f:
            forecast_y=[]
        this_batch=bdata[bdata[bdata.columns[0]]==which_batch]
        
        if (bdata.columns[1]=='PHASE') or \
            (bdata.columns[1]=='phase') or \
            (bdata.columns[1]=='Phase'):
            colnames=this_batch.columns[2:].tolist()
        else:
            colnames=this_batch.columns[1:].tolist()
            
        ss_indx=[]    
        #Make soft_sensor var all missing data
        if not(isinstance(soft_sensor,bool)):            
            if isinstance(soft_sensor,list):
                for v in soft_sensor:
                    if not(v in colnames):
                        print('Soft sensor '+v+' is not a variable in this dataset')
                    else:
                        ss_indx.append(colnames.index(v))
            elif isinstance(soft_sensor,str):
                if not(soft_sensor in colnames):
                    print('Soft sensor '+soft_sensor+' is not a variable in this dataset')
                else:
                    ss_indx=[colnames.index(soft_sensor)]

        if not(isinstance(zinit,bool)):
            this_z = zinit[zinit[zinit.columns[0]]==which_batch]
            vals_z = this_z.values[:,1:]
            cols_z = this_z.columns[1:]
            vals_z = np.array(vals_z,dtype=np.float64)
            vals_z = vals_z.reshape(-1)

        if (bdata.columns[1]=='PHASE') or \
            (bdata.columns[1]=='phase') or \
            (bdata.columns[1]=='Phase'):
            vals=this_batch.values[:,2:]
        else:
            vals=this_batch.values[:,1:]

            
        vals=np.array(vals,dtype=np.float64)
        
        if len(ss_indx)>0:
            vals[:,ss_indx]=np.nan
        
        if not(shush):
            print('Running batch: '+which_batch)
        
        if 'Q' in mmvm_obj_f:
            forecast_y=[]
            forecast_y_=[]
        for k in np.arange(mmvm_obj_f['nsamples']):
            x_uf_k   = vals[:k+1,:].reshape(1,-1) 
            x_uf_k   = x_uf_k[0].tolist()
            num_nans = mmvm_obj_f['nvars']*mmvm_obj_f['nsamples']-len(x_uf_k)
            x_uf_k.extend([np.nan]*num_nans )
            x_uf_k   = np.array(x_uf_k)
            if 'Q' in mmvm_obj_f:
                if not(isinstance(zinit,bool)) and mmvm_obj_f['ninit']>0:
                    x_2_pred = np.hstack((vals_z,x_uf_k))
                    preds    = phi.pls_pred(x_2_pred, mmvm_obj_f)
                elif (isinstance(zinit,bool)) and mmvm_obj_f['ninit']==0:    
                    preds    = phi.pls_pred(x_uf_k, mmvm_obj_f)
                else:
                    print('Model and data do not correspond')
                    preds = False
            else:    
                preds    = phi.pca_pred(x_uf_k, mmvm_obj_f)
            
            T_mon.append(preds['Tnew'][0])
            ht2_mon.append(preds['T2'][0])
            #spe_mon.append(preds['speX'][0])
            
      
            if 'Q' in mmvm_obj_f:
               forecast_y_.append(preds['Yhat'][0].reshape(-1))
               if not(isinstance(zinit,bool)) and mmvm_obj_f['ninit']>0:
                   ninit=mmvm_obj_f['ninit']
                   preds_x = preds['Xhat'][0,ninit:]
                   preds_z = preds['Xhat'][0,:ninit]
                   aux_df=pd.DataFrame(preds_x.reshape(mmvm_obj_f['nsamples'],-1),columns=colnames)
                   forecast.append(aux_df)    
                   inst_preds = preds_x.reshape(-1)  
                   inst_preds = (inst_preds - mmvm_obj_f['mx'][ninit:] )/mmvm_obj_f['sx'][ninit:]
                   #mean center current sample and current prediction for instantaneous SPE
                   x_uf_k     = (x_uf_k - mmvm_obj_f['mx'][ninit:] )/mmvm_obj_f['sx'][ninit:] 
               else:
                   aux_df=pd.DataFrame(preds['Xhat'].reshape(mmvm_obj_f['nsamples'],-1),columns=colnames)
                   forecast.append(aux_df)        
                   inst_preds = preds['Xhat'].reshape(-1)  
                   inst_preds = (inst_preds - mmvm_obj_f['mx'])/mmvm_obj_f['sx']
                   #mean center current sample and current prediction for instantaneous SPE
                   x_uf_k     = (x_uf_k - mmvm_obj_f['mx'])/mmvm_obj_f['sx']
            else:
                aux_df=pd.DataFrame(preds['Xhat'].reshape(mmvm_obj_f['nsamples'],-1),columns=colnames)
                forecast.append(aux_df)  
                inst_preds = preds['Xhat'].reshape(-1)  
                inst_preds = (inst_preds - mmvm_obj_f['mx'])/mmvm_obj_f['sx']
                #mean center current sample and current prediction for instantaneous SPE
                x_uf_k     = (x_uf_k - mmvm_obj_f['mx'])/mmvm_obj_f['sx']
                
            inst_preds[np.isnan(x_uf_k)]    = 0
            x_uf_k[np.isnan(x_uf_k)]        = 0
            
            if not(isinstance(zinit,bool)) and mmvm_obj_f['ninit']>0:
                cont_ht2_=np.zeros(len(x_2_pred))
            else:
                cont_ht2_=np.zeros(len(x_uf_k))
            
            var_t=np.var(mmvm_obj_f['T'],ddof=1,axis=0)
            for a in np.arange(mmvm_obj_f['A']):
                if 'Q' in mmvm_obj_f:
                    if not(isinstance(zinit,bool)) and mmvm_obj_f['ninit']>0:
                        x_2_pred_= (x_2_pred- mmvm_obj_f['mx'])/mmvm_obj_f['sx']
                        cont_ht2_+= (x_2_pred_ * mmvm_obj_f['Ws'][:,a])**2 / var_t[a]
                        
                    else:
                        cont_ht2_+= (x_uf_k   * mmvm_obj_f['Ws'][:,a])**2 / var_t[a]
                else:
                    cont_ht2_+= (x_uf_k * mmvm_obj_f['P'][:,a])**2 / var_t[a]
            
            if not(isinstance(zinit,bool)) and mmvm_obj_f['ninit']>0:
                conts_ht2_z = cont_ht2_[:ninit]
                aux_df=pd.DataFrame(cont_ht2_[ninit:].reshape(mmvm_obj_f['nsamples'],-1),columns=colnames)    
                cont_ht2.append(aux_df)
            else:
                aux_df=pd.DataFrame(cont_ht2_.reshape(mmvm_obj_f['nsamples'],-1),columns=colnames)    
                cont_ht2.append(aux_df)
      
            spe_       = (inst_preds - x_uf_k)**2
            aux_df     = pd.DataFrame(spe_.reshape(mmvm_obj_f['nsamples'],-1),columns=colnames)
            cont_spe.append(aux_df)
            spe_mon.append(np.sum(spe_))
            
            inst_samp  = x_uf_k[k*mmvm_obj_f['nvars']:(k+1)*mmvm_obj_f['nvars']]
            inst_preds = inst_preds[k*mmvm_obj_f['nvars']:(k+1)*mmvm_obj_f['nvars']]
            spei_      = (inst_preds - inst_samp)**2
            
            cont_spei.append(spei_)
            spei_mon.append(np.sum(spei_) )
         
        diags={'Batch':which_batch,'t_mon':np.array(T_mon),'HT2_mon':np.array(ht2_mon),
               'spe_mon':np.array(spe_mon).reshape(-1),'cont_spe':cont_spe,
               'spei_mon':np.array(spei_mon),'cont_spei':pd.DataFrame(np.array(cont_spei),columns=colnames),
               'cont_ht2':cont_ht2,'forecast':forecast} 
        if 'Q' in mmvm_obj_f:
            forecast_y=pd.DataFrame(np.array(forecast_y_),columns=mmvm_obj_f['varidY'])
            diags['forecast y']=forecast_y
            
            if not(isinstance(zinit,bool)) and mmvm_obj_f['ninit']>0:
                vals_z  = (vals_z - mmvm_obj_f['mx'][:ninit] )/mmvm_obj_f['sx'][:ninit]
                preds_z_ = (preds_z - mmvm_obj_f['mx'][:ninit] )/mmvm_obj_f['sx'][:ninit]
                spe_z = (vals_z - preds_z_)
                cont_spe_z = spe_z**2
                spe_z = np.sum(cont_spe_z)
                cont_spe_z  = pd.DataFrame(cont_spe_z,index=cols_z,columns=['Vars'])
                conts_ht2_z =pd.DataFrame(conts_ht2_z,index=cols_z,columns=['Vars'])
                diags['reconstructed z'] = preds_z
                diags['spe z']           = spe_z
                diags['cont_spe_z']      = cont_spe_z
                diags['cont_ht2_z']      = conts_ht2_z
        return diags  
   
def monitor(mmvm_obj,bdata,*,which_batch=False,zinit=False,build_ci=True,shush=False,soft_sensor=False):
    ''' Routine to mimic the real-time monitoring of a batch given a model
    
    monitor(mmvm_obj,bdata,*,which_batch=False,zinit=False,build_ci=True,shush=False,soft_sensor=False):
     
  
     usage: 1st you need to run: monitor(mmvm_obj,bdata)  
                to mimic monitoring for all bdata batches and build CI
                these new parameters are written back to mmvm_obj
                
            Then you can run:
                
            diagnostics = monitor(mmvm_obj,bdata,which_batch=your_batchid) 
                to mimic monitoring for your_batchid and will produce all dynamic metrics
                and forecasts
                
            diagnostics = monitor(mmvm_obj,bdata,which_batch=your_batchid,zinit=your_z_data)  
                to mimic monitoring for your_batchid  using initial conditions
                will produce all dynamic metrics and forecasts 
                
            diagnostics = monitor(mmvm_obj,bdata,which_batch=your_batchid,soft_sensor=your_variable)  
                to mimic monitoring for your_batchid will produce all dynamic metrics 
                and forecasts and produce soft-sensor predictions for your_variable
     Returns:
         
        diagnostics:A dictionary with all the monitoring diagnostics and contributions

    by Salvador Garcia Munoz
        sgarciam@imperial.ac.uk salvadorgarciamunoz@gmail.com           
    '''
    mmvm_obj_f       = mmvm_obj.copy()
    mmvm_obj_f       = _uf_hor_mon_loadings(mmvm_obj_f)
    mmvm_obj_f['mx'] = mmvm_obj_f['mx_ufm']
    mmvm_obj_f['sx'] = mmvm_obj_f['sx_ufm']
    mmvm_obj_f['P']  = mmvm_obj_f['P_ufm']
    aux=bdata.drop_duplicates(subset=bdata.columns[0],keep='first')
    unique_batches=aux[aux.columns[0]].values.tolist()
    
    if 'Q' in mmvm_obj:
        mmvm_obj_f['W']   = mmvm_obj_f['W_ufm']
        mmvm_obj_f['Ws']  = mmvm_obj_f['Ws_ufm']
    
    if isinstance(which_batch,bool) and build_ci:
        if mmvm_obj['nbatches']==len(np.unique(bdata[bdata.columns[0]]) ):
            SPE  = []
            SPEi = []
            T    = np.zeros((mmvm_obj['nsamples'],mmvm_obj['A'],mmvm_obj['nbatches']))
            #build confidence intervals and update model
            if not(shush):
                print('Building real_time confidence intervals')
            for i,b in enumerate(unique_batches):
                diags = _mimic_monitoring(mmvm_obj_f,bdata,b,zinit=zinit,soft_sensor=soft_sensor)
                SPE.append(diags['spe_mon'])
                SPEi.append(diags['spei_mon'])
                T[:,:,i]=diags['t_mon']
                
            #calculate conf int for scores
            t_mon_ci_95 = []
            t_mon_ci_99 = []
            for a in np.arange(mmvm_obj['A']):
                t_rt_ci_95 = []
                t_rt_ci_99 = []
                t          = T[:,a,:].T
                for j in np.arange(mmvm_obj_f['nsamples']):
                    l95,l99 = phi.single_score_conf_int(t[:,[j]])
                    t_rt_ci_95.append(l95)
                    t_rt_ci_99.append(l99)
                t_mon_ci_95.append(np.array(t_rt_ci_95))
                t_mon_ci_99.append(np.array(t_rt_ci_99))
            #calculate conf. int for spe and HT2
            
            #HT2 limits
            n  = mmvm_obj['nbatches']
            A= mmvm_obj['A']
            ht2_mon_ci_99 = (((n-1)*(n+1)*A)/(n*(n-A)))*phi.f99(A,(n-A))
            ht2_mon_ci_95 = (((n-1)*(n+1)*A)/(n*(n-A)))*phi.f95(A,(n-A)) 
  
            spe_mon_ci_95  = []
            spe_mon_ci_99  = []
            spei_mon_ci_95 = []
            spei_mon_ci_99 = []
            SPE            = np.array(SPE)
            SPEi           = np.array(SPEi)
            for j in np.arange(mmvm_obj_f['nsamples']):
                l95,l99=phi.spe_ci(SPE[:,[j]])
                spe_mon_ci_95.append(l95)
                spe_mon_ci_99.append(l99)
                l95,l99=phi.spe_ci(SPEi[:,[j]])
                spei_mon_ci_95.append(l95)
                spei_mon_ci_99.append(l99)
                
            mmvm_obj['t_mon_ci_95']     = t_mon_ci_95
            mmvm_obj['t_mon_ci_99']     = t_mon_ci_99
            mmvm_obj['ht2_mon_ci_99']   = ht2_mon_ci_99
            mmvm_obj['ht2_mon_ci_95']   = ht2_mon_ci_95
            mmvm_obj['spe_mon_ci_95']   = spe_mon_ci_95
            mmvm_obj['spe_mon_ci_99']   = spe_mon_ci_99
            mmvm_obj['spei_mon_ci_95']  = spei_mon_ci_95
            mmvm_obj['spei_mon_ci_99'] = spei_mon_ci_99
            if not(shush):
                print('Done')
            #return mmvm_obj
        else:
            print("This ain't the data this model was trained on")
    else:
        
        if not('t_mon_ci_95' in mmvm_obj):
            print('No monitoring conf. int. have been calculated')
            print('for this model object please run: "monitor(model_obj,training_data)" ')
            has_ci=False
        else:
            has_ci=True
        if isinstance(which_batch,str):
            which_batch=[which_batch]
        diags=[]
        allok=True
        for b in which_batch:
            if b in bdata[bdata.columns[0]].values.tolist():
            #run monitoring on a batch
                diags_ =_mimic_monitoring(mmvm_obj_f,bdata,b,zinit=zinit,soft_sensor=soft_sensor)
                diags.append(diags_)
            else:
                print('Batch not found in data set')
                allok=False
        if allok:
        #plot scores
            for a in np.arange(mmvm_obj['A']):
                plt.figure()
                for i,b in enumerate(which_batch):
                    x_axis=np.arange(len(diags[i]['t_mon'][:,[a]] ) )+1
                    plt.plot(x_axis,diags[i]['t_mon'][:,[a]],'o',label=b)
                if has_ci:
                    plt.plot(x_axis,mmvm_obj['t_mon_ci_95'][a],'y',alpha=0.3)
                    plt.plot(x_axis,-mmvm_obj['t_mon_ci_95'][a],'y',alpha=0.3)
                    plt.plot(x_axis,mmvm_obj['t_mon_ci_99'][a],'r',alpha=0.3)
                    plt.plot(x_axis,-mmvm_obj['t_mon_ci_99'][a],'r',alpha=0.3)
                plt.xlabel('sample')
                plt.ylabel('$t_'+str(a+1)+'$')     
                plt.title('Real time monitoring: Score plot $t_'+str(a+1)+'$')                                
                box = plt.gca().get_position()
                plt.gca().set_position([box.x0, box.y0, box.width * 0.8, box.height])
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))   

                
            #plot ht2
            plt.figure()
            for i,b in enumerate(which_batch):
                plt.plot(x_axis,diags[i]['HT2_mon'],'o',label=b)
                xlim_=plt.xlim()
            if has_ci:
                plt.plot([0,xlim_[1]],[mmvm_obj['ht2_mon_ci_95'],mmvm_obj['ht2_mon_ci_95']],'y',alpha=0.3)
                plt.plot([0,xlim_[1]],[mmvm_obj['ht2_mon_ci_99'],mmvm_obj['ht2_mon_ci_99']],'r',alpha=0.3)
            plt.xlabel('sample')
            plt.ylabel("Hotelling's $T^2$")
            plt.title("Real time monitoring: Hotelling's $T^2$")
            box = plt.gca().get_position()
            plt.gca().set_position([box.x0, box.y0, box.width * 0.8, box.height])
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))             
            
            #plot spe
            plt.figure()
            for i,b in enumerate(which_batch):
                plt.plot(x_axis,diags[i]['spe_mon'],'o',label=b)
                xlim_=plt.xlim()
            if has_ci:
                plt.plot(x_axis,mmvm_obj['spe_mon_ci_95'],'y',alpha=0.3)
                plt.plot(x_axis,mmvm_obj['spe_mon_ci_99'],'r',alpha=0.3)
            plt.xlabel('sample')
            plt.ylabel("Global SPE")
            plt.title("Real time monitoring: Global SPE")
            
            box = plt.gca().get_position()
            plt.gca().set_position([box.x0, box.y0, box.width * 0.8, box.height])
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5)) 
            #plot spei
            plt.figure()
            for i,b in enumerate(which_batch):
                plt.plot(x_axis,diags[i]['spei_mon'],'o',label=b)
                xlim_=plt.xlim()
            if has_ci:
                plt.plot(x_axis,mmvm_obj['spei_mon_ci_95'],'y',alpha=0.3)
                plt.plot(x_axis,mmvm_obj['spei_mon_ci_99'],'r',alpha=0.3)
            plt.xlabel('sample')
            plt.ylabel("Instantaneous SPE")
            plt.title("Real time monitoring: Instantaneous SPE")
            box = plt.gca().get_position()
            plt.gca().set_position([box.x0, box.y0, box.width * 0.8, box.height])
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5)) 
            
            if 'Q' in mmvm_obj:
                for v in diags[0]['forecast y'].columns:
                    plt.figure()
                    for i,b in enumerate(which_batch):
                        plt.plot(x_axis,diags[i]['forecast y'][v],'o',label=b)
                    plt.xlabel('sample')
                    plt.ylabel(v)
                    plt.title('Dynamic forecast of Y')
            if len(diags)==1:
                diags=diags[0]
            return diags
        else:
            return 'error batch not found'

def mpls(xbatch,y,a,*,zinit=False,phase_samples=False,mb_each_var=False,cross_val=0,cross_val_X=False):
    ''' Multi-way PLS for batch analysis
    
    mpls_obj = mpls(xbatch,y,a,*,zinit=False,phase_samples=False,mb_each_var=False,cross_val=0,cross_val_X=False):

    Args:
        xbatch: Pandas dataframe with aligned batch data it is assumed 
                 that all batches have the same number of samples
                 
        y     : Response to predict, one row per batch
                
        a:      Number of PC's to fit
        
        zinit: Initial conditions <optional>
        
        phase_samples: alignment information
        
        mb_each_var: if "True" will make each variable measured a block
                     otherwise zinit is one block and xbatch another
    Returns:
        mpls_obj: A dictionary with all the parameters of the MPLS model
        
    by Salvador Garcia Munoz
        sgarciam@imperial.ac.uk salvadorgarciamunoz@gmail.com   
    '''    
    if (xbatch.columns[1]=='PHASE') or \
        (xbatch.columns[1]=='phase') or \
        (xbatch.columns[1]=='Phase'):
        nvars = xbatch.shape[1]-2
    else:
        nvars = xbatch.shape[1]-1
    nbatches = len(np.unique(xbatch[xbatch.columns[0]]))    
    nsamples = xbatch.shape[0]/nbatches
    
    # remove low variance columns keeping record of the original order
    x_uf_,colnames,bid_o = unfold_horizontal(xbatch)  # colnames is original set of columns
    x_uf,colsrem        = phi.clean_low_variances(x_uf_,shush=True) # colsrem are columns removed
    mx_rem=x_uf_[colsrem].mean().tolist()
    mx_rem=np.array(mx_rem)
    aux               = np.array([colnames,bid_o])
    col_names_bid_pd  = pd.DataFrame(aux.T,columns=['col name','bid'])
    col_names_bid_pd_ = col_names_bid_pd[col_names_bid_pd['col name'].isin(x_uf.columns[1:].tolist())]
    #bid               = col_names_bid_pd_['bid'].values # bid is vector indicating to what variable each col belongs 
    #                                                   # useful in figuring out the blocks
    aux=col_names_bid_pd_.drop_duplicates(subset='bid',keep='first')
    unique_bid=aux['bid'].values.tolist()                                                   
    
    if not(isinstance(zinit,bool)):
        zinit,rc=phi.clean_low_variances(zinit)
        zcols=zinit.columns[1:].tolist()
        XMB={'Initial Conditions':zinit}
    else:
        XMB=dict()
    
    if mb_each_var:
        for v in unique_bid:
           these_cols=[x_uf.columns[0]]
           these_cols.extend(col_names_bid_pd_['col name'][col_names_bid_pd_['bid']==v].values.tolist())
           varblock=x_uf[these_cols]
           XMB[v]=varblock
    else:
        if not(isinstance(zinit,bool)):
            XMB['Trajectories']=x_uf
        else:
            XMB = x_uf
            
    if not(isinstance(XMB,dict)):
        mpls_obj=phi.pls(XMB,y,a,cross_val=cross_val,cross_val_X=cross_val_X,force_nipals=True)
        yhat=phi.pls_pred(XMB,mpls_obj)
        yhat=yhat['Yhat']
    else:    
        mpls_obj=phi.mbpls(XMB,y,a,cross_val_=cross_val,cross_val_X_=cross_val_X,force_nipals_=True)    
        yhat=phi.pls_pred(XMB,mpls_obj)
        yhat=yhat['Yhat']
    if len(colsrem)>0:    
        if not(isinstance(zinit,bool)):
            ninit_vars=zinit.shape[1]-1
            z_sx   = mpls_obj['sx'][np.arange(ninit_vars)].reshape(-1)
            z_mx   = mpls_obj['mx'][np.arange(ninit_vars)].reshape(-1)
            z_ws   = mpls_obj['Ws'][np.arange(ninit_vars),:]   
            z_w    = mpls_obj['W'][np.arange(ninit_vars),:]   
            z_p    = mpls_obj['P'][np.arange(ninit_vars),:]   
            z_r2pv = mpls_obj['r2xpv'][np.arange(ninit_vars),:]   
            xc_    = np.arange(ninit_vars,x_uf.shape[1]+ninit_vars-1 )
            xuf_sx = mpls_obj['sx'][xc_]
            xuf_mx = mpls_obj['mx'][xc_]
            xuf_ws = mpls_obj['Ws'][xc_,:]     
            xuf_w  = mpls_obj['W'][xc_,:]
            xuf_p  = mpls_obj['P'][xc_,:]   
            xuf_r2pv = mpls_obj['r2xpv'][xc_,:]
        else:
            xuf_sx = mpls_obj['sx']
            xuf_mx = mpls_obj['mx']
            xuf_ws = mpls_obj['Ws']   
            xuf_w  = mpls_obj['W'] 
            xuf_p  = mpls_obj['P'] 
            xuf_r2pv = mpls_obj['r2xpv']
        

        #add removed columns with mean of zero and stdev =1 and loadings = 0
        xtra_col    = np.zeros((2+4*a,1 ))    
        xtra_col[0] = 1
        xtra_cols   = np.tile(xtra_col,(1,len(colsrem))) 
        xtra_cols[1,:] = mx_rem
        aux         = np.vstack((xuf_sx,xuf_mx,xuf_ws.T,xuf_w.T,xuf_p.T,xuf_r2pv.T))
        aux         = np.hstack((aux,xtra_cols))
        all_cols    = x_uf.columns[1:].tolist()
        all_cols.extend(colsrem)
        aux_pd      = pd.DataFrame(aux,columns=all_cols)
        aux_pd      = aux_pd[colnames]
        aux_new     = aux_pd.values
        
        if not(isinstance(zinit,bool)):   
             sx_               = np.hstack((z_sx,aux_new[0,:].reshape(-1)))
             mpls_obj['sx']    = sx_
             mx_               = np.hstack((z_mx,aux_new[1,:].reshape(-1)))
             mpls_obj['mx']    = mx_
             aux_new           = aux_new[2:,:]
             ws_               = aux_new[0:a,:]
             mpls_obj['Ws']    = np.vstack((z_ws,ws_.T))         
             aux_new           = aux_new[a:,:]
             w_                = aux_new[0:a,:]
             mpls_obj['W']     = np.vstack((z_w,w_.T))         
             aux_new           = aux_new[a:,:]
             p_                = aux_new[0:a,:]
             mpls_obj['P']     = np.vstack((z_p,p_.T)) 
             aux_new           = aux_new[a:,:]
             r2xpv_            = aux_new[0:a,:]
             mpls_obj['r2xpv'] = np.vstack((z_r2pv,r2xpv_.T))
             zcols.extend(colnames)
             colnames=zcols
        else:
             sx_               = aux_new[0,:].reshape(1,-1)
             mpls_obj['sx']    = sx_
             mx_               = aux_new[1,:].reshape(1,-1)
             mpls_obj['mx']    = mx_
             aux_new           = aux_new[2:,:]
             ws_               = aux_new[0:a,:]
             mpls_obj['Ws']    = ws_.T    
             aux_new           = aux_new[a:,:]
             w_                = aux_new[0:a,:]
             mpls_obj['W']     = w_.T    
             aux_new           = aux_new[a:,:]
             p_                = aux_new[0:a,:]
             mpls_obj['P']     = p_.T
             aux_new           = aux_new[a:,:]
             r2xpv_            = aux_new[0:a,:]
             mpls_obj['r2xpv'] = r2xpv_.T
    mpls_obj['Yhat']          = yhat
    mpls_obj['varidX']        = colnames
    mpls_obj['bid']           = bid_o
    mpls_obj['uf']            ='batch wise'
    mpls_obj['nvars']         = int(nvars)
    mpls_obj['nbatches']      = int(nbatches)
    mpls_obj['nsamples']      = int(nsamples)
    mpls_obj['A']             = a
    mpls_obj['phase_samples'] = phase_samples  
    mpls_obj['mb_each_var']   = mb_each_var
        
    if not(isinstance(zinit,bool)):
        mpls_obj['ninit']=int(zinit.shape[1]-1)
    else:
        mpls_obj['ninit']=0
            
    return mpls_obj       

def find(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]
     
def clean_empty_rows(X,*,shush=False):
    ''' Cleans empty rows in batch data
    Input: 
        X: Batch data to be cleaned of empty rows (all np.nan) DATAFRAME
    Output:
        X: Batch data Without observations removed
    '''

    if  (X.columns[1]=='PHASE') or \
        (X.columns[1]=='phase') or \
        (X.columns[1]=='Phase'): 
            X_     = np.array(X.values[:,2:]).astype(float)
            ObsID_ = X.values[:,0].astype(str)
            ObsID_ = ObsID_.tolist()
    else:
            X_     = np.array(X.values[:,1:]).astype(float)
            ObsID_ = X.values[:,0].astype(str)
            ObsID_ = ObsID_.tolist()
                      
    #find rows with all data missing
    X_nan_map = np.isnan(X_)
    Xmiss = X_nan_map*1
    Xmiss = np.sum(Xmiss,axis=1)
    indx = find(Xmiss, lambda x: x==X_.shape[1])
       
    if len(indx)>0:
        for i in indx:
            if not(shush):
                print('Removing row from ', ObsID_[i], ' due to 100% missing data')

        X_=X.drop(X.index.values[indx].tolist())

        return X_
    else:
        return X
    
def phase_sampling_dist(bdata,time_column=False,addtitle=False,use_phases=False):    
    ''' Count and plot a histogram of the distribution of  samples (or time if time_column is indicated)
    consumed per phase on a batch dataset
    
    phase_sampling_dist(bdata,time_column=False,addtitle=False,use_phases=False)
    
    Args:
        
        bdata: Batch data organized as: column[0] = Batch Identifier column name is unrestricted
                                        column[1] = Phase information per sample must be called 'Phase','phase', or 'PHASE'
                                                    this information is optional
                                        column[2:]= Variables measured throughout the batch
        time_column: Indicates the name of the column with time, if not sent, counting is done in terms samples
        
        add_title:  Optional text to be placed as the figure title
        use_phases: In case the user wants to only do counting for a subset of phases
        
    by Salvador Garcia Munoz
        sgarciam@imperial.ac.uk salvadorgarciamunoz@gmail.com   
        
    '''
    data={}
    if bdata.columns[1]=='phase' or bdata.columns[1]=='Phase' or bdata.columns[1]=='PHASE':        
        bids   = unique(bdata,bdata.columns[0])
        if isinstance(use_phases,bool):
            phases = unique(bdata,bdata.columns[1])
        else:
            phases = use_phases
        
        #samps_per_phase=[]
        fig,ax=plt.subplots(1,len(phases)+1)
        
        for i,p in enumerate(phases):
            totsamps=[]
            samps_=[]
            samps_ind={}
            for b in bids:
              bdat= bdata[ (bdata[bdata.columns[1]]==p) & (bdata[bdata.columns[0]]==b)]
              if len(bdat)==0:
                  samps_.append(0)
                  samps_ind[b]=0
              elif not(isinstance(time_column,bool)):
                  samps_.append(bdat[time_column].values[-1]-bdat[time_column].values[0] )
                  totsamps.append(bdata[time_column][(bdata[bdata.columns[0]]==b)].values[-1])
                  samps_ind[b]=bdat[time_column].values[-1]-bdat[time_column].values[0]
              else:    
                  samps_.append(len(bdat))
                  totsamps.append(len(bdata[(bdata[bdata.columns[0]]==b)]))                  
                  samps_ind[b]=len(bdat)                  
            ax[i].hist(samps_)
            data[p]=samps_ind
            if not(isinstance(time_column,bool)):
                ax[i].set_xlabel(time_column)
            else:    
                ax[i].set_xlabel('# Samples')
            ax[i].set_ylabel('Count')
            ax[i].set_title(p)
        ax[-1].hist(totsamps)
        if not(isinstance(time_column,bool)):
            ax[-1].set_xlabel(time_column)
        else:    
            ax[-1].set_xlabel('# Samples')
        
        ax[-1].set_ylabel('Count')
        ax[-1].set_title('Total')
        if not(isinstance(addtitle,bool)):
            fig.suptitle(addtitle)
        fig.tight_layout()           
        return data            
    else:
        print('Data is missing phase information or phase column is nor properly labeled')
        
def predict(xbatch,mmvm_obj,*,zinit=False):    
    ''' Generate predictions for a Multi-way PCA/PLS model
    
    predictions = predict(xbatch,mmvm_obj,*,zinit=False)
    
    Args:
        xbatch:   Batch data with same variables and alignment as model will generate predictions for all batches
        mmvm_obj: Multi-way PLS or PCA
        zinit:    Initial conditions [if any]
        
    Returns:
        preds: A dictionary with keys ['Yhat', 'Xhat', 'Tnew', 'speX', 'T2']
   
            
    by Salvador Garcia Munoz
        sgarciam@imperial.ac.uk salvadorgarciamunoz@gmail.com   
    '''
    if 'Q' in mmvm_obj:    
        x_uf,colnames,bid_o = unfold_horizontal(xbatch)  # colnames is original set of columns        
        aux                 = np.array([colnames,bid_o])
        col_names_bid_pd    = pd.DataFrame(aux.T,columns=['col name','bid'])        
        #bid                 = col_names_bid_pd['bid'].values # bid is vector indicating to what variable each col belongs 
        #                                                    # useful in figuring out the blocks
        aux=col_names_bid_pd.drop_duplicates(subset='bid',keep='first')
        unique_bid=aux['bid'].values.tolist()   
            
        if not(isinstance(zinit,bool)):
            XMB={'Initial Conditions':zinit}
        else:
            XMB=dict()
        
        if mmvm_obj['mb_each_var']:
            
            for v in unique_bid:
               these_cols=[x_uf.columns[0]]
               these_cols.extend(col_names_bid_pd['col name'][col_names_bid_pd['bid']==v].values.tolist())
               varblock=x_uf[these_cols]
               XMB[v]=varblock
        else:
            if not(isinstance(zinit,bool)):
                XMB['Trajectories']=x_uf
            else:
                XMB = x_uf           
        pred=phi.pls_pred(XMB,mmvm_obj)
        
        if not(isinstance(zinit,bool)):
            Zhat=pred['Xhat'][:,:mmvm_obj['ninit']]            
            Xhat=pred['Xhat'][:,mmvm_obj['ninit']:]
            Xb=refold_horizontal(Xhat,mmvm_obj['nvars'],mmvm_obj['nsamples'] )
            if  (xbatch.columns[1]=='PHASE') or \
                (xbatch.columns[1]=='phase') or \
                (xbatch.columns[1]=='Phase'): 
                Xb=pd.DataFrame(Xb,columns=xbatch.columns[2:])
                Xb.insert(0,xbatch.columns[1],xbatch[xbatch.columns[1]].values)
                Xb.insert(0,xbatch.columns[0],xbatch[xbatch.columns[0]].values)
            else:
                Xb=pd.DataFrame(Xb,columns=xbatch.columns[1:])                
                Xb.insert(0,xbatch.columns[0],xbatch[xbatch.columns[0]].values)                
            pred['Xhat']=Xb
            
            
            Zhat_df=pd.DataFrame(Zhat,columns=zinit.columns[1:].tolist())
            Zhat_df.insert(0,zinit.columns[0],zinit[zinit.columns[0]].values.astype(str).tolist())    
            pred['Zhat']=Zhat_df
            
            test=xbatch.drop_duplicates(subset=xbatch.columns[0],keep='first')            
            Y_df=pd.DataFrame(pred['Yhat'],columns=mmvm_obj['varidY'])                
            Y_df.insert(0,test.columns[0],test[test.columns[0]].values.astype(str).tolist())                
            pred['Yhat']=Y_df
            
        else:
            Xb=refold_horizontal(pred['Xhat'],mmvm_obj['nvars'],mmvm_obj['nsamples'] )
            if  (xbatch.columns[1]=='PHASE') or \
                (xbatch.columns[1]=='phase') or \
                (xbatch.columns[1]=='Phase'): 
                Xb=pd.DataFrame(Xb,columns=xbatch.columns[2:])
                Xb.insert(0,xbatch.columns[1],xbatch[xbatch.columns[1]].values)
                Xb.insert(0,xbatch.columns[0],xbatch[xbatch.columns[0]].values)
            else:
                Xb=pd.DataFrame(Xb,columns=xbatch.columns[1:])                
                Xb.insert(0,xbatch.columns[0],xbatch[xbatch.columns[0]].values)                
            pred['Xhat']=Xb
            test=xbatch.drop_duplicates(subset=xbatch.columns[0],keep='first')
            Y_df=pd.DataFrame(pred['Yhat'],columns=mmvm_obj['varidY'])                
            Y_df.insert(0,test.columns[0],test[test.columns[0]].values.astype(str).tolist())                
            pred['Yhat']=Y_df

    else:
        if mmvm_obj['uf'] =='batch wise':
            x_uf,colnames,bid_o = unfold_horizontal(xbatch)              
            pred=phi.pca_pred(x_uf,mmvm_obj)               
            Xb=refold_horizontal(pred['Xhat'],mmvm_obj['nvars'],mmvm_obj['nsamples'] )
            if  (xbatch.columns[1]=='PHASE') or \
                (xbatch.columns[1]=='phase') or \
                (xbatch.columns[1]=='Phase'): 
                Xb=pd.DataFrame(Xb,columns=xbatch.columns[2:])
                Xb.insert(0,xbatch.columns[1],xbatch[xbatch.columns[1]].values)
                Xb.insert(0,xbatch.columns[0],xbatch[xbatch.columns[0]].values)
            else:
                Xb=pd.DataFrame(Xb,columns=xbatch.columns[1:])                
                Xb.insert(0,xbatch.columns[0],xbatch[xbatch.columns[0]].values)                
            pred['Xhat']=Xb
        if mmvm_obj['uf'] =='variable wise':
            if  (xbatch.columns[1]=='PHASE') or \
                (xbatch.columns[1]=='phase') or \
                (xbatch.columns[1]=='Phase'):    
                 xbatch_=xbatch.copy()
                 xbatch_.drop(xbatch.columns[1],axis=1,inplace=True)
            else:
                 xbatch_=xbatch.copy()            
            pred=phi.pca_pred(xbatch_,mmvm_obj)    
            Xb=pred['Xhat']
            if  (xbatch.columns[1]=='PHASE') or \
                (xbatch.columns[1]=='phase') or \
                (xbatch.columns[1]=='Phase'): 
                Xb=pd.DataFrame(Xb,columns=xbatch.columns[2:])
                Xb.insert(0,xbatch.columns[1],xbatch[xbatch.columns[1]].values)
                Xb.insert(0,xbatch.columns[0],xbatch[xbatch.columns[0]].values)
            else:
                Xb=pd.DataFrame(Xb,columns=xbatch.columns[1:])                
                Xb.insert(0,xbatch.columns[0],xbatch[xbatch.columns[0]].values)      
            pred['Xhat']=Xb                   
    return pred

def _plot_contribs(bdata,ylims,*,var_list=False,phase_samples=False,alpha_=0.2,plot_title_=''):
    '''
    Internal routine please use : contributions
    '''
    if (bdata.columns[1]=='PHASE') or \
        (bdata.columns[1]=='phase') or \
        (bdata.columns[1]=='Phase'):
        if isinstance(var_list,bool):    
            var_list=bdata.columns[2:].tolist()
    else:
        if isinstance(var_list,bool):
            var_list=bdata.columns[1:].tolist()
    if isinstance(var_list,str) and not(isinstance(var_list,list)):
         var_list=[var_list]    
         
    for v in var_list:
        plt.figure()
        dat=bdata[[bdata.columns[0],v]]
        for b in np.unique(dat[bdata.columns[0]]):
            data_=dat[v][dat[dat.columns[0]]==b]
            plt.fill_between(np.arange(len(data_.values))+1,data_.values)
            plt.xlabel('Sample')
            plt.ylabel(v)
        if not(isinstance(phase_samples,bool)):   
            s_txt=1
            s_lin=1
            plt.axvline(x=1,color='magenta',alpha=alpha_)
            for p in phase_samples.keys():                
                if isinstance(phase_samples[p],list):
                    s_lin+=phase_samples[p][1]
                else:                    
                    s_lin+=phase_samples[p]
                plt.axvline(x=s_lin,color='magenta',alpha=alpha_)
                ylim_=plt.ylim()
                plt.annotate(p, (s_txt,ylim_[0]),rotation=90,alpha=0.5,color='magenta')
                if isinstance(phase_samples[p],list):
                    s_txt+=phase_samples[p][1]
                else:                                    
                    s_txt+=phase_samples[p]
        plt.ylim(ylims)
        plt.title(plot_title_)
        plt.tight_layout()
                    

def contributions (mmvmobj,X,cont_type,*,to_obs=False,from_obs=False,
                   lv_space=False,phase_samples=False,dyn_conts=False,which_var=False,
                   plot_title=''):
    ''' Plot batch contribution plots to Scores, HT2 or SPE
    
    contributions (mmvmobj,X,cont_type,*,to_obs=False,from_obs=False,
                       lv_space=False,phase_samples=False,dyn_conts=False,which_var=False,
                       plot_title='')
    
    Args:
        mmvmobj= Multiway Model
        X      = batch data
        cont_type = 'scores' | 'ht2' | 'spe'
        to_obs    = Observation to calculate contributions to
        from_obs  = Relative basis to calculate contributions to [for 'scores' and 'ht2' only]
                    if not sent the origin of the model us used as the base.
        
    Returns:
        contribution_vector
        
    by Salvador Garcia Munoz
        sgarciam@imperial.ac.uk salvadorgarciamunoz@gmail.com
    '''
    
    #translate from batch numbers to indexes
    #Check logic and warn user
    if (X.columns[1]=='PHASE') or \
        (X.columns[1]=='phase') or \
            (X.columns[1]=='Phase'):
        bvars = X.columns[2:]
    else:
        bvars = X.columns[1:]
    
    allok=True
    if  isinstance(to_obs,bool):
        print('Contribution calculations need a "to_obs" argument at minimum')
        allok=False
    if cont_type=='spe' and not(isinstance(from_obs,bool)):
        print('For SPE contributions the "from_obs" argument is ignored')
    if (isinstance(to_obs,list) and len(to_obs)>1) or (isinstance(from_obs,list) and len(from_obs)>1):
        print('Contributions for groups of observations will be averaged')
    if (cont_type=='scores') and isinstance(lv_space,bool):
        print('No dimensions specified, doing contributions across all dimensions')
    if allok:
        #find indexes to batch names
        Xuf,var1,var2=unfold_horizontal(X)
        bnames=Xuf[Xuf.columns[0]].values.tolist()
        to_o=[bnames.index(to_obs[i]) for i in np.arange(len(to_obs))]
        if (cont_type=='scores') or (cont_type=='ht2'):
            if not(isinstance(from_obs,bool)):
                   from_o = [bnames.index(from_obs[i]) for i in np.arange(len(from_obs))]
            else:
                    from_o =False
            cont_vec=phi.contributions(mmvmobj,Xuf,cont_type,Y=False,from_obs=from_o,to_obs=to_o,lv_space=lv_space)
        elif cont_type=='spe':
            cont_vec=phi.contributions(mmvmobj,Xuf,cont_type,Y=False,from_obs=False,to_obs=to_o)
        ymin=cont_vec.min()
        ymax=cont_vec.max()
        #Plot contributions
        if dyn_conts:
            batch_contribs=refold_horizontal(cont_vec, mmvmobj['nvars'],mmvmobj['nsamples'])
            batch_contribs = pd.DataFrame(batch_contribs,columns=bvars)
            oids=['Batch Contributions']*batch_contribs.shape[0]
            batch_contribs.insert(0,'BatchID',oids)
            _plot_contribs(batch_contribs,(ymin, ymax),phase_samples=phase_samples,var_list=which_var,plot_title_=plot_title)
            
        batch_contribs=refold_horizontal(abs(cont_vec), mmvmobj['nvars'],mmvmobj['nsamples'])
        batch_contribs=np.sum(batch_contribs,axis=0)
        plt.figure()
        plt.bar(bvars, batch_contribs)
        plt.xticks(rotation=75)
        plt.ylabel(r'$\Sigma$ (|Contribution to '+cont_type+'| )' )
        plt.title(plot_title)
        plt.tight_layout()
        
def build_rel_time(bdata,*,time_unit='min'):
    ''' Converts the column 'Timestamp' into 'Time' in time_units relative to
     the start of each batch
     
     bdata_new = build_rel_time(bdata,*,time_unit='min')
     
     
     by Salvador Garcia Munoz
         sgarciam@imperial.ac.uk salvadorgarciamunoz@gmail.com
         
    '''
    
    aux=bdata.drop_duplicates(subset=bdata.columns[0],keep='first')
    unique_batches=aux[aux.columns[0]].values.tolist()
    bdata_n=[]
    for b in unique_batches:
        this_batch=bdata[bdata[bdata.columns[0]]==b]
        ts=this_batch['Timestamp'].values
        deltat=ts-ts[0]
        tim=[np.timedelta64(deltat[i],'s').astype(float) for i in np.arange(len(deltat))]
        tim=np.array(tim)
        if time_unit == 'min':
            tim=tim/60
        if time_unit == 'hr':
            tim=tim/3600
        cols=this_batch.columns.tolist()
        this_batch.insert(cols.index('Timestamp'),'Time ('+time_unit+')' ,tim)
        this_batch=this_batch.drop('Timestamp',axis=1)
        bdata_n.append(this_batch)
    bdata_n = pd.concat(bdata_n,axis=0)
    return bdata_n

def descriptors(bdata,which_var,desc,*,phase=False):
    '''Get descriptor values for a batch trajectory
    
    descriptors_df = descriptors(bdata,which_var,desc,*,phase=False)
    
    Args:
        bdata: Dataframe of batch data, first column is batch ID, second column can be phase id
        
        which_var: List of variables to get descriptors for
        
        desc: List of descriptors to calculate, options are:
                'min'
                'max'
                'mean'
                'median'
                'std'
                'var'
                'range'
                'ave_slope'
                
        phase: to specify what phases to do this for
        
    Returns:
        descriptors: A dataframe with the descriptors per batch
        
     by Salvador Garcia Munoz
         sgarciam@imperial.ac.uk salvadorgarciamunoz@gmail.com
         
    '''
    has_phase=False
    if ((bdata.columns[1]=='Phase') or 
        (bdata.columns[1]=='phase') or 
        (bdata.columns[1]=='PHASE')):
           has_phase=True
           phase_col=bdata.columns[1]
    if not(has_phase) and not(isinstance(phase,bool)):
        print('Cannot process phase flag without phase id in data')
        phase=False
    desc_val=[]
    desc_lbl=[]    
    
    for i,b in enumerate(unique(bdata,bdata.columns[0])):
        this_batch=bdata[bdata[bdata.columns[0]]==b]
        desc_val_=[]
        for v in which_var:
            if not(isinstance(phase,bool)): # by phase
                for p in phase:
                    values=this_batch[v][this_batch[phase_col]==p].values.astype(float)
                    values=values[~(np.isnan(values))]
                    for d in desc:
                        if d=='min':
                            desc_val_.append(values.min())                           
                        if d=='max':
                            desc_val_.append(values.max())                                                 
                        if d=='mean':
                            desc_val_.append(values.mean())
                        if d=='median':
                            desc_val_.append(np.median(values))
                        if d=='std':
                            desc_val_.append(np.std(values,ddof=1))
                        if d=='var':
                            desc_val_.append(np.var(values,ddof=1))
                        if d=='range':
                            desc_val_.append(values.max()-values.min() )
                        if d=='ave_slope':
                            x=np.arange(0,len(values))
                            m=(1/sum(x**2))*sum(x*(values-values[0]))
                            desc_val_.append(m)
                        if i==0:    
                            desc_lbl.append(v +'_'+ p +'_'+ d)   
            else:
                values=this_batch[v].values.astype(float)
                values=values[~(np.isnan(values))]                
                for d in desc:
                    if d=='min':
                        desc_val_.append(values.min())                           
                    if d=='max':
                        desc_val_.append(values.max())                                                 
                    if d=='mean':
                        desc_val_.append(values.mean())
                    if d=='median':
                        desc_val_.append(np.median(values))
                    if d=='std':
                        desc_val_.append(np.std(values,ddof=1))
                    if d=='var':
                        desc_val_.append(np.var(values,ddof=1))
                    if d=='range':
                        desc_val_.append(values.max()-values.min() )
                    if d=='ave-slope':
                        x=np.arange(0,len(values))
                        m=(1/sum(x**2))*sum(x*(values-values[0]))
                        desc_val.apend(m)
                    if i==0:    
                        desc_lbl.append(v +'_' + d) 
                
        desc_val.append(desc_val_)   
    bnames = unique(bdata,bdata.columns[0])  
    desc_val=np.array(desc_val)
    descriptors_df=pd.DataFrame(desc_val,columns=desc_lbl )     
    descriptors_df.insert(0,bdata.columns[0],bnames )
    return descriptors_df                         
                                
                        
                    
            
        
        
        
        
        
        
