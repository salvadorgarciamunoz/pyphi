#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 12:01:54 2019

@author: Sal Garcia <sgarciam@ic.ac.uk> <salvadorgarciamunoz@gmail.com>
"""
import numpy as np
from bokeh.io import show, output_file
from bokeh.plotting import figure
from bokeh.layouts import column
from bokeh.models import ColumnDataSource,LabelSet,Span

import pyphi as phi
import pandas as pd

import matplotlib.cm as cm

def r2pv(mvmobj):
    """
    R2 per variabvle plots
    by Salvador Garcia-Munoz 
    (sgarciam@ic.ac.uk ,salvadorgarciamunoz@gmail.com)
    
    mvmobj: A model created with phi.pca or phi.pls
    """
    A= mvmobj['T'].shape[1]
    num_varX=mvmobj['P'].shape[0]
    
    if 'Q' in mvmobj:
        is_pls=True
        lv_prefix='LV #'
    else:
        is_pls=False
        lv_prefix='PC #'
        
    lv_labels = []   
    for a in list(np.arange(A)+1):
        lv_labels.append(lv_prefix+str(a))    
    

    if 'varidX' in mvmobj:
        r2pvX_dict = {'XVar': mvmobj['varidX']}
        XVar=mvmobj['varidX']
    else:
        XVar = []
        for n in list(np.arange(num_varX)+1):
            XVar.append('XVar #'+str(n))               
        r2pvX_dict = {'XVar': XVar}
        
    for i in list(np.arange(A)):
        r2pvX_dict.update({lv_labels[i] : mvmobj['r2xpv'][:,i].tolist()})
        
    if 'Q' in mvmobj:
        num_varY=mvmobj['Q'].shape[0]
        if 'varidY' in mvmobj:
            r2pvY_dict = {'YVar': mvmobj['varidY']}
            YVar=mvmobj['varidY']
        else:
            YVar = []
            for n in list(np.arange(num_varY)+1):
                YVar.append('YVar #'+str(n))               
            r2pvY_dict = {'YVar': YVar}
        for i in list(np.arange(A)):
            r2pvY_dict.update({lv_labels[i] : mvmobj['r2ypv'][:,i].tolist()})
            
    
    if is_pls:
        rnd_num=str(int(np.round(1000*np.random.random_sample())))
        output_file("r2xypv_"+rnd_num+".html",title="R2XYPV") 
        colormap =cm.get_cmap("rainbow")
        different_colors=A
        color_mapping=colormap(np.linspace(0,1,different_colors),1,True)
        bokeh_palette=["#%02x%02x%02x" % (r, g, b) for r, g, b in color_mapping[:,0:3]]  
               
        px = figure(x_range=XVar, plot_height=400, title="R2X Per Variable",
             tools="save,box_zoom,hover,reset", tooltips="$name @XVar: @$name")
        
        px.vbar_stack(lv_labels, x='XVar', width=0.9,color=bokeh_palette,source=r2pvX_dict)
        px.y_range.range_padding = 0.1
        px.ygrid.grid_line_color = None
        px.axis.minor_tick_line_color = None
        px.outline_line_color = None
        px.yaxis.axis_label = 'R2X'
        
        py = figure(x_range=YVar, plot_height=400, title="R2Y Per Variable",
            tools="save,box_zoom,hover,reset", tooltips="$name @YVar: @$name")
        
        py.vbar_stack(lv_labels, x='YVar', width=0.9,color=bokeh_palette,source=r2pvY_dict)
        py.y_range.range_padding = 0.1
        py.ygrid.grid_line_color = None
        py.axis.minor_tick_line_color = None
        py.outline_line_color = None
        py.yaxis.axis_label = 'R2Y'
        show(column(px,py))
        
        
    else:   
        rnd_num=str(int(np.round(1000*np.random.random_sample())))
        output_file("r2xpv_"+rnd_num+".html",title='R2XPV') 
        colormap =cm.get_cmap("rainbow")
        different_colors=A
        color_mapping=colormap(np.linspace(0,1,different_colors),1,True)
        bokeh_palette=["#%02x%02x%02x" % (r, g, b) for r, g, b in color_mapping[:,0:3]]  
               
        p = figure(x_range=XVar, plot_height=400, title="R2X Per Variable",
             tools="save,box_zoom,hover,reset", tooltips="$name @XVar: @$name")
        
        p.vbar_stack(lv_labels, x='XVar', width=0.9,color=bokeh_palette,source=r2pvX_dict)
        p.y_range.range_padding = 0.1
        p.ygrid.grid_line_color = None
        p.axis.minor_tick_line_color = None
        p.outline_line_color = None
        p.yaxis.axis_label = 'R2X'
        show(p)
    return
    
def loadings(mvmobj):
    """
    Column plots of loadings
    by Salvador Garcia-Munoz 
    (sgarciam@ic.ac.uk ,salvadorgarciamunoz@gmail.com)
    
    mvmobj: A model created with phi.pca or phi.pls
    
    """
    
    A= mvmobj['T'].shape[1]
    num_varX=mvmobj['P'].shape[0]    
    if 'Q' in mvmobj:
        is_pls=True
        lv_prefix='LV #'
    else:
        is_pls=False
        lv_prefix='PC #'       
    lv_labels = []   
    for a in list(np.arange(A)+1):
        lv_labels.append(lv_prefix+str(a))    
    if 'varidX' in mvmobj:
        X_loading_dict = {'XVar': mvmobj['varidX']}
        XVar=mvmobj['varidX']
    else:
        XVar = []
        for n in list(np.arange(num_varX)+1):
            XVar.append('XVar #'+str(n))               
        X_loading_dict = {'XVar': XVar}
    if 'Q' in mvmobj:
        for i in list(np.arange(A)):
            X_loading_dict.update({lv_labels[i] : mvmobj['Ws'][:,i].tolist()})
            
        num_varY=mvmobj['Q'].shape[0]
        if 'varidY' in mvmobj:
            Q_dict = {'YVar': mvmobj['varidY']}
            YVar=mvmobj['varidY']
        else:
            YVar = []
            for n in list(np.arange(num_varY)+1):
                YVar.append('YVar #'+str(n))               
            Q_dict = {'YVar': YVar}
        for i in list(np.arange(A)):
            Q_dict.update({lv_labels[i] : mvmobj['Q'][:,i].tolist()})
    else:
        for i in list(np.arange(A)):
            X_loading_dict.update({lv_labels[i] : mvmobj['P'][:,i].tolist()})
    if is_pls:
        rnd_num=str(int(np.round(1000*np.random.random_sample())))
        output_file("Loadings X Space_"+rnd_num+".html",title='X Loadings PLS')
        for i in list(np.arange(A)):
            p = figure(x_range=XVar, plot_height=250, title="X Space Loadings "+lv_labels[i],
                    tools="save,box_zoom,pan,reset")
            p.vbar(x=XVar, top=mvmobj['Ws'][:,i].tolist(), width=0.5)
            p.xgrid.grid_line_color = None
            p.yaxis.axis_label = 'W* ['+str(i+1)+']'
            hline = Span(location=0, dimension='width', line_color='black', line_width=2)
            p.renderers.extend([hline])
            if i==0:
                p_list=[p]
            else:
                p_list.append(p)
        show(column(p_list))    
        rnd_num=str(int(np.round(1000*np.random.random_sample())))
        output_file("Loadings Y Space_"+rnd_num+".html",title='Y Loadings PLS')
        for i in list(np.arange(A)):
            p = figure(x_range=YVar, plot_height=250, title="Y Space Loadings "+lv_labels[i],
                    tools="save,box_zoom,pan,reset")
            p.vbar(x=YVar, top=mvmobj['Q'][:,i].tolist(), width=0.5)
            p.xgrid.grid_line_color = None
            p.yaxis.axis_label = 'Q ['+str(i+1)+']'
            hline = Span(location=0, dimension='width', line_color='black', line_width=2)
            p.renderers.extend([hline])
            if i==0:
                p_list=[p]
            else:
                p_list.append(p)                    
        show(column(p_list))
    else:   
        rnd_num=str(int(np.round(1000*np.random.random_sample())))
        output_file("Loadings X Space_"+rnd_num+".html",title='X Loadings PCA') 
        for i in list(np.arange(A)):
            p = figure(x_range=XVar, plot_height=250, title="X Space Loadings "+lv_labels[i],
                    tools="save,box_zoom,pan,reset")
            p.vbar(x=XVar, top=mvmobj['P'][:,i].tolist(), width=0.5)
            p.xgrid.grid_line_color = None
            p.yaxis.axis_label = 'P ['+str(i+1)+']'
            hline = Span(location=0, dimension='width', line_color='black', line_width=2)
            p.renderers.extend([hline])
            if i==0:
                p_list=[p]
            else:
                p_list.append(p)
        show(column(p_list))
    return    

def loadings_map(mvmobj,dims):
    """
    Scatter plot overlaying X and Y loadings 
    by Salvador Garcia-Munoz 
    (sgarciam@ic.ac.uk ,salvadorgarciamunoz@gmail.com)
    
    mvmobj: A model created with phi.pca or phi.pls
    dims: what latent spaces to plot in x and y axes e.g. dims=[1,2]
    """
    A= mvmobj['T'].shape[1]
    num_varX=mvmobj['P'].shape[0]    
    if 'Q' in mvmobj:
        lv_prefix='LV #'     
        lv_labels = []   
        for a in list(np.arange(A)+1):
            lv_labels.append(lv_prefix+str(a))    
        if 'varidX' in mvmobj:
            XVar=mvmobj['varidX']
        else:
            XVar = []
            for n in list(np.arange(num_varX)+1):
                XVar.append('XVar #'+str(n))               
        num_varY=mvmobj['Q'].shape[0]
        if 'varidY' in mvmobj:
            YVar=mvmobj['varidY']
        else:
            YVar = []
            for n in list(np.arange(num_varY)+1):
                YVar.append('YVar #'+str(n))               
    
        rnd_num=str(int(np.round(1000*np.random.random_sample())))
        output_file("Loadings Map"+rnd_num+".html",title='Loadings Map')
       
    
        x_ws = mvmobj['Ws'][:,dims[0]-1]
        x_ws = x_ws/np.max(np.abs(x_ws))
        y_ws = mvmobj['Ws'][:,dims[1]-1]
        y_ws = y_ws/np.max(np.abs(y_ws))
        
        x_q = mvmobj['Q'][:,dims[0]-1]
        x_q = x_q/np.max(np.abs(x_q))   
        y_q = mvmobj['Q'][:,dims[1]-1]
        y_q = y_q/np.max(np.abs(y_q))
        
        
        TOOLS = "save,wheel_zoom,box_zoom,pan,reset,box_select,lasso_select"
        TOOLTIPS = [
                ("index", "$index"),
                ("(x,y)", "($x, $y)"),
                ("Variable:","@names")
                ]
    
        source1 = ColumnDataSource(data=dict(x=x_ws, y=y_ws,names=XVar))  
        source2 = ColumnDataSource(data=dict(x=x_q, y=y_q,names=YVar)) 
        p = figure(tools=TOOLS, tooltips=TOOLTIPS,plot_width=600, plot_height=600, title="Loadings Map LV["+str(dims[0])+"] - LV["+str(dims[1])+"]",
                                                                                                          x_range=(-1.5,1.5),y_range=(-1.5,1.5))
        p.circle('x', 'y', source=source1,size=10,color='darkblue')
        p.circle('x', 'y', source=source2,size=10,color='red')
        p.xaxis.axis_label = lv_labels [dims[0]-1]
        p.yaxis.axis_label = lv_labels [dims[1]-1]
        
        labelsX = LabelSet(x='x', y='y', text='names', level='glyph',x_offset=5, y_offset=5, source=source1, render_mode='canvas')
        labelsY = LabelSet(x='x', y='y', text='names', level='glyph',x_offset=5, y_offset=5, source=source2, render_mode='canvas')
        p.add_layout(labelsX)
        p.add_layout(labelsY)

        vline = Span(location=0, dimension='height', line_color='black', line_width=2)
        # Horizontal line
        hline = Span(location=0, dimension='width', line_color='black', line_width=2)
        p.renderers.extend([vline, hline])
        show(p)    
    return  

def weighted_loadings(mvmobj):
    """
    Column plots of loadings weighted by r2x/r2y correspondingly
    by Salvador Garcia-Munoz 
    (sgarciam@ic.ac.uk ,salvadorgarciamunoz@gmail.com)
    
    mvmobj: A model created with phi.pca or phi.pls
    
    """
    A= mvmobj['T'].shape[1]
    num_varX=mvmobj['P'].shape[0]    
    if 'Q' in mvmobj:
        is_pls=True
        lv_prefix='LV #'
    else:
        is_pls=False
        lv_prefix='PC #'       
    lv_labels = []   
    for a in list(np.arange(A)+1):
        lv_labels.append(lv_prefix+str(a))    
    if 'varidX' in mvmobj:
        X_loading_dict = {'XVar': mvmobj['varidX']}
        XVar=mvmobj['varidX']
    else:
        XVar = []
        for n in list(np.arange(num_varX)+1):
            XVar.append('XVar #'+str(n))               
        X_loading_dict = {'XVar': XVar}
    if 'Q' in mvmobj:
        for i in list(np.arange(A)):
            X_loading_dict.update({lv_labels[i] : mvmobj['Ws'][:,i].tolist()})
            
        num_varY=mvmobj['Q'].shape[0]
        if 'varidY' in mvmobj:
            Q_dict = {'YVar': mvmobj['varidY']}
            YVar=mvmobj['varidY']
        else:
            YVar = []
            for n in list(np.arange(num_varY)+1):
                YVar.append('YVar #'+str(n))               
            Q_dict = {'YVar': YVar}
        for i in list(np.arange(A)):
            Q_dict.update({lv_labels[i] : mvmobj['Q'][:,i].tolist()})
    else:
        for i in list(np.arange(A)):
            X_loading_dict.update({lv_labels[i] : mvmobj['P'][:,i].tolist()})
    if is_pls:
        rnd_num=str(int(np.round(1000*np.random.random_sample())))
        output_file("Loadings X Space_"+rnd_num+".html",title='X Weighted Loadings PLS')
        for i in list(np.arange(A)):
            p = figure(x_range=XVar, plot_height=250, title="X Space Weighted Loadings "+lv_labels[i],
                     tools="save,box_zoom,pan,reset")
            p.vbar(x=XVar, top=(mvmobj['r2xpv'][:,i] * mvmobj['Ws'][:,i]).tolist(), width=0.5)
            p.xgrid.grid_line_color = None
            p.yaxis.axis_label = 'W* ['+str(i+1)+']'
            hline = Span(location=0, dimension='width', line_color='black', line_width=2)
            p.renderers.extend([hline])
            if i==0:
                p_list=[p]
            else:
                p_list.append(p)
        show(column(p_list)) 
        rnd_num=str(int(np.round(1000*np.random.random_sample())))
        output_file("Loadings Y Space_"+rnd_num+".html",title='Y Weighted Loadings PLS')
        for i in list(np.arange(A)):
            p = figure(x_range=YVar, plot_height=250, title="Y Space Weighted Loadings "+lv_labels[i],
                     tools="save,box_zoom,pan,reset")
            p.vbar(x=YVar, top=(mvmobj['r2ypv'][:,i] * mvmobj['Q'][:,i]).tolist(), width=0.5)
            p.xgrid.grid_line_color = None
            p.yaxis.axis_label = 'Q ['+str(i+1)+']'
            hline = Span(location=0, dimension='width', line_color='black', line_width=2)
            p.renderers.extend([hline])
            if i==0:
                p_list=[p]
            else:
                p_list.append(p)                    
        show(column(p_list))
    else:   
        rnd_num=str(int(np.round(1000*np.random.random_sample())))
        output_file("Loadings X Space_"+rnd_num+".html",title='X Weighted Loadings PCA') 
        for i in list(np.arange(A)):
            p = figure(x_range=XVar, plot_height=250, title="X Space Weighted Loadings "+lv_labels[i],
                     tools="save,box_zoom,pan,reset")
            p.vbar(x=XVar, top=(mvmobj['r2xpv'][:,i] * mvmobj['P'][:,i]).tolist(), width=0.5)
            p.xgrid.grid_line_color = None
            p.yaxis.axis_label = 'P ['+str(i+1)+']'
            hline = Span(location=0, dimension='width', line_color='black', line_width=2)
            p.renderers.extend([hline])
            if i==0:
                p_list=[p]
            else:
                p_list.append(p)
        show(column(p_list))
    return  
 
def vip(mvmobj):
    """
    Very Important to the Projection (VIP) plot
        by Salvador Garcia-Munoz 
    (sgarciam@ic.ac.uk ,salvadorgarciamunoz@gmail.com)
    
    mvmobj: A model created with phi.pls
    """
    
    if 'Q' in mvmobj:  
        rnd_num=str(int(np.round(1000*np.random.random_sample())))
        output_file("VIP_"+rnd_num+".html",title='VIP Coefficient') 
        num_varX=mvmobj['P'].shape[0]            
        if 'varidX' in mvmobj:
            XVar=mvmobj['varidX']
        else:
            XVar = []
            for n in list(np.arange(num_varX)+1):
                XVar.append('XVar #'+str(n))               
        
        vip=np.sum(np.abs(mvmobj['Ws'] * np.tile(mvmobj['r2y'],(mvmobj['Ws'].shape[0],1)) ),axis=1)
        vip=np.reshape(vip,(len(vip),-1))
        sort_indx=np.argsort(-vip,axis=0)
        vip=vip[sort_indx]
        sorted_XVar=[]
        for i in sort_indx[:,0]:
            sorted_XVar.append(XVar[i])  
        p = figure(x_range=sorted_XVar, plot_height=250, title="VIP",
            tools="save,box_zoom,pan,reset")
        p.vbar(x=sorted_XVar, top=vip.tolist(), width=0.5)
        p.xgrid.grid_line_color = None
        p.yaxis.axis_label = 'Very Important to the Projection'
        show(p)
    return    

def score_scatter(mvmobj,xydim,*,CLASSID=False,colorby=False,Xnew=False,add_ci=False):
    '''
    Score scatter plot
    by Salvador Garcia-Munoz 
    (sgarciam@ic.ac.uk ,salvadorgarciamunoz@gmail.com)
    
    mvmobj : PLS or PCA object from phyphi
    xydim  : LV to plot on x and y axes. eg [1,2] will plot t1 vs t2
    CLASSID: Pandas DataFrame with CLASSIDS
    colorby: Category (one of the CLASSIDS) to color by
    Xnew   : New data for which to make the score plot this routine evaluates and plots
    '''
    
    if isinstance(Xnew,bool):
        if 'obsidX' in mvmobj:
            ObsID_=mvmobj['obsidX']
        else:
            ObsID_ = []
            for n in list(np.arange(mvmobj['T'].shape[0])+1):
                ObsID_.append('Obs #'+str(n))  
        T_matrix=mvmobj['T']
    else:
        if isinstance(Xnew,np.ndarray):
            X_=Xnew.copy()
            ObsID_ = []
            for n in list(np.arange(Xnew.shape[0])+1):
                ObsID_.append('Obs #'+str(n))  
        elif isinstance(Xnew,pd.DataFrame):
            X_=np.array(Xnew.values[:,1:]).astype(float)
            ObsID_ = Xnew.values[:,0].astype(str)
            ObsID_ = ObsID_.tolist()
            
        if 'Q' in mvmobj:  
            xpred=phi.pls_pred(X_,mvmobj)
        else:
            xpred=phi.pca_pred(X_,mvmobj)
        T_matrix=xpred['Tnew']

    
    if isinstance(CLASSID,np.bool): # No CLASSIDS
        rnd_num=str(int(np.round(1000*np.random.random_sample())))
        output_file("Score_Scatter_"+rnd_num+".html",title='Score Scatter t['+str(xydim[0])+'] - t['+str(xydim[1])+ ']')

        x_=T_matrix[:,[xydim[0]-1]]
        y_=T_matrix[:,[xydim[1]-1]]

           
        source = ColumnDataSource(data=dict(x=x_, y=y_,ObsID=ObsID_))
        TOOLS = "save,wheel_zoom,box_zoom,pan,reset,box_select,lasso_select"
        TOOLTIPS = [
                ("index", "$index"),
                ("(x,y)", "($x, $y)"),
                ("Obs: ","@ObsID")
                ]
        
        p = figure(tools=TOOLS, tooltips=TOOLTIPS,plot_width=600, plot_height=600, title='Score Scatter t['+str(xydim[0])+'] - t['+str(xydim[1])+ ']')
        p.circle('x', 'y', source=source,size=7)
        if add_ci:
            T_aux1=mvmobj['T'][:,[xydim[0]-1]]
            T_aux2=mvmobj['T'][:,[xydim[1]-1]]
            T_aux = np.hstack((T_aux1,T_aux2))
            st=(T_aux.T @ T_aux)/T_aux.shape[0]
            [xd95,xd99,yd95p,yd95n,yd99p,yd99n]=phi.scores_conf_int_calc(st,mvmobj['T'].shape[0])
            p.line(xd95,yd95p,line_color="gold",line_dash='dashed')
            p.line(xd95,yd95n,line_color="gold",line_dash='dashed')
            p.line(xd99,yd99p,line_color="red",line_dash='dashed')
            p.line(xd99,yd99n,line_color="red",line_dash='dashed')
            

        p.xaxis.axis_label = 't ['+str(xydim[0])+']'
        p.yaxis.axis_label = 't ['+str(xydim[1])+']'
        # Vertical line
        vline = Span(location=0, dimension='height', line_color='black', line_width=2)
        # Horizontal line
        hline = Span(location=0, dimension='width', line_color='black', line_width=2)
        p.renderers.extend([vline, hline])
        show(p)      
    else: # YES CLASSIDS
        Classes_=np.unique(CLASSID[colorby]).tolist()
        A=len(Classes_)
        colormap =cm.get_cmap("rainbow")
        different_colors=A
        color_mapping=colormap(np.linspace(0,1,different_colors),1,True)
        bokeh_palette=["#%02x%02x%02x" % (r, g, b) for r, g, b in color_mapping[:,0:3]]  
        rnd_num=str(int(np.round(1000*np.random.random_sample())))               
        output_file("Score_Scatter_"+rnd_num+".html",title='Score Scatter t['+str(xydim[0])+'] - t['+str(xydim[1])+ ']') 
        x_=T_matrix[:,[xydim[0]-1]]
        y_=T_matrix[:,[xydim[1]-1]]          
        
        TOOLS = "save,wheel_zoom,box_zoom,pan,reset,box_select,lasso_select"
        TOOLTIPS = [
                ("index", "$index"),
                ("(x,y)", "($x, $y)"),
                ("Obs: ","@ObsID"),
                ("Class:","@Class")
                ]        
        classid_=list(CLASSID[colorby])
        
        
        p = figure(tools=TOOLS, tooltips=TOOLTIPS,plot_width=600, plot_height=600, title='Score Scatter t['+str(xydim[0])+'] - t['+str(xydim[1])+ ']')

        for classid_in_turn in Classes_:
            x_aux=[]
            y_aux=[]
            obsid_aux=[]
            classid_aux=[]
            for i in list(range(len(ObsID_))):
                if classid_[i]==classid_in_turn:
                    x_aux.append(x_[i][0])
                    y_aux.append(y_[i][0])
                    obsid_aux.append(ObsID_[i])
                    classid_aux.append(classid_in_turn)
            source = ColumnDataSource(data=dict(x=x_aux, y=y_aux,ObsID=obsid_aux,Class=classid_aux))        
            color_=bokeh_palette[Classes_.index(classid_in_turn)]
            p.circle('x','y',source=source,color=color_,legend=classid_in_turn)
        if add_ci:
            T_aux1=mvmobj['T'][:,[xydim[0]-1]]
            T_aux2=mvmobj['T'][:,[xydim[1]-1]]
            T_aux = np.hstack((T_aux1,T_aux2))
            st=(T_aux.T @ T_aux)/T_aux.shape[0]
            [xd95,xd99,yd95p,yd95n,yd99p,yd99n]=phi.scores_conf_int_calc(st,mvmobj['T'].shape[0])
            p.line(xd95,yd95p,line_color="gold",line_dash='dashed')
            p.line(xd95,yd95n,line_color="gold",line_dash='dashed')
            p.line(xd99,yd99p,line_color="red",line_dash='dashed')
            p.line(xd99,yd99n,line_color="red",line_dash='dashed') 
        p.xaxis.axis_label = 't ['+str(xydim[0])+']'
        p.yaxis.axis_label = 't ['+str(xydim[1])+']'
        # Vertical line
        vline = Span(location=0, dimension='height', line_color='black', line_width=2)
        # Horizontal line
        hline = Span(location=0, dimension='width', line_color='black', line_width=2)
        p.renderers.extend([vline, hline])
        show(p)
    return    

def diagnostics(mvmobj,*,Xnew=False,Ynew=False,score_plot_xydim=False):
    """
    Plot calculated Hotelling's T2 and SPE
    by Salvador Garcia-Munoz 
    (sgarciam@ic.ac.uk ,salvadorgarciamunoz@gmail.com)
    
    mvmobj: A model created with phi.pca or phi.pls
    
    Xnew/Ynew:     Data used to calculate diagnostics[numpy arrays or pandas dataframes] 
    
    optional:
        
    score_plot_xydim: will add a score scatter plot at the bottom 
                      if sent with a list of [dimx, dimy] where dimx/dimy 
                      are integers and refer to the latent space to plot
                      in the x and y axes of the scatter plot. e.g. [1,2] will
                      add a t1-t2 plot 
    
    """
    
    if isinstance(score_plot_xydim,np.bool):
        add_score_plot = False
    else:
        add_score_plot = True
        
    if isinstance(Xnew,np.bool): #No Xnew was given need to plot all from model
        if 'obsidX' in mvmobj:
            ObsID_=mvmobj['obsidX']
        else:
            ObsID_ = []
            for n in list(np.arange(mvmobj['T'].shape[0])+1):
                ObsID_.append('Obs #'+str(n))  
                              
        Obs_num = np.arange(mvmobj['T'].shape[0])+1
        
        if add_score_plot and not(isinstance(score_plot_xydim,np.bool)):
            t_x  = mvmobj['T'][:,[score_plot_xydim[0]-1]]
            t_y  = mvmobj['T'][:,[score_plot_xydim[1]-1]]
        else:
            add_score_plot = False
        
        if not(add_score_plot):
            if 'Q' in mvmobj:
                spey_=1
                source = ColumnDataSource(data=dict(x=Obs_num, ObsID=ObsID_,t2=mvmobj['T2'],spex=mvmobj['speX'],spey=mvmobj['speY']))  
            else:
                source = ColumnDataSource(data=dict(x=Obs_num, ObsID=ObsID_,t2=mvmobj['T2'],spex=mvmobj['speX'])) 
        else:
            if 'Q' in mvmobj:
                spey_=1
                source = ColumnDataSource(data=dict(x=Obs_num, ObsID=ObsID_,t2=mvmobj['T2'],spex=mvmobj['speX'],spey=mvmobj['speY'],tx=t_x,ty=t_y))  
            else:
                source = ColumnDataSource(data=dict(x=Obs_num, ObsID=ObsID_,t2=mvmobj['T2'],spex=mvmobj['speX'],tx=t_x,ty=t_y))
    else: #Xnew was given
        if isinstance(Xnew,np.ndarray):
            ObsID_ = []
            for n in list(np.arange(Xnew.shape[0])+1):
                ObsID_.append('Obs #'+str(n))  
        elif isinstance(Xnew,pd.DataFrame):
            X_=np.array(Xnew.values[:,1:]).astype(float)
            ObsID_ = Xnew.values[:,0].astype(str)
            ObsID_ = ObsID_.tolist()
            
        Obs_num = np.arange(mvmobj['T'].shape[0])+1
        
        if add_score_plot and not(isinstance(score_plot_xydim,np.bool)):
            if 'Q' in mvmobj:  
                xpred=phi.pls_pred(X_,mvmobj)
            else:
                xpred=phi.pca_pred(X_,mvmobj)
            T_matrix=xpred['Tnew']
            t_x  = T_matrix[:,[score_plot_xydim[0]-1]]
            t_y  = T_matrix[:,[score_plot_xydim[1]-1]]
        else:
            add_score_plot = False
        
        t2_ = phi.hott2(mvmobj,Xnew=Xnew)
        
        if 'Q' in mvmobj and not(isinstance(Ynew,np.bool)):
            spex_,spey_ = phi.spe(mvmobj,Xnew,Ynew=Ynew)
        else:
            spex_ = phi.spe(mvmobj,Xnew)
            spey_ = False
        if not(add_score_plot):
            if 'Q' in mvmobj and not(isinstance(Ynew,np.bool)):
                source = ColumnDataSource(data=dict(x=Obs_num, ObsID=ObsID_,t2=t2_,spex=spex_,spey=spey_))  
            else:
                source = ColumnDataSource(data=dict(x=Obs_num, ObsID=ObsID_,t2=t2_,spex=spex_)) 
        else:
            if 'Q' in mvmobj and not(isinstance(Ynew,np.bool)):
                source = ColumnDataSource(data=dict(x=Obs_num, ObsID=ObsID_,t2=t2_,spex=spex_,spey=spey_,tx=t_x,ty=t_y))  
            else:
                source = ColumnDataSource(data=dict(x=Obs_num, ObsID=ObsID_,t2=t2_,spex=spex_,tx=t_x,ty=t_y))
    TOOLS = "save,wheel_zoom,box_zoom,reset,lasso_select"
    TOOLTIPS = [
            ("index", "$index"),
            ("(x,y)", "($x, $y)"),
            ("Obs: ","@ObsID")
            ]
    
    rnd_num=str(int(np.round(1000*np.random.random_sample())))               
    output_file("Diagnostics"+rnd_num+".html",title='Diagnostics') 
    p = figure(tools=TOOLS, tooltips=TOOLTIPS, plot_height=400, title="Hotelling's T2")
    p.circle('x','t2',source=source)
    p.line([0,Obs_num[-1]],[mvmobj['T2_lim95'],mvmobj['T2_lim95']],line_color='gold')
    p.line([0,Obs_num[-1]],[mvmobj['T2_lim99'],mvmobj['T2_lim99']],line_color='red')
    p.xaxis.axis_label = 'Observation sequence'
    p.yaxis.axis_label = "HT2"
    p_list=[p]
    
    p = figure(tools=TOOLS, tooltips=TOOLTIPS, plot_height=400, title='SPE X')
    p.circle('x','spex',source=source)
    p.line([0,Obs_num[-1]],[mvmobj['speX_lim95'],mvmobj['speX_lim95']],line_color='gold')
    p.line([0,Obs_num[-1]],[mvmobj['speX_lim99'],mvmobj['speX_lim99']],line_color='red')
    p.xaxis.axis_label = 'Observation sequence'
    p.yaxis.axis_label = 'SPE X-Space'
    p_list.append(p)
    
    p = figure(tools=TOOLS, tooltips=TOOLTIPS, plot_height=400, title='Outlier Map')
    p.circle('t2','spex',source=source)
    
    vline = Span(location=mvmobj['T2_lim99'], dimension='height', line_color='red', line_width=1)
    hline = Span(location=mvmobj['speX_lim99'], dimension='width', line_color='red', line_width=1)
    p.renderers.extend([vline, hline])
    
    p.xaxis.axis_label = "Hotelling's T2"
    p.yaxis.axis_label = 'SPE X-Space'
    p_list.append(p)
    
    
    if 'Q' in mvmobj and not(isinstance(spey_,np.bool)):
        p = figure(tools=TOOLS, tooltips=TOOLTIPS, plot_height=400, title='SPE Y')
        p.circle('x','spey',source=source)
        p.line([0,Obs_num[-1]],[mvmobj['speY_lim95'],mvmobj['speY_lim95']],line_color='gold')
        p.line([0,Obs_num[-1]],[mvmobj['speY_lim99'],mvmobj['speY_lim99']],line_color='red')
        p.xaxis.axis_label = 'Observation sequence'
        p.yaxis.axis_label = 'SPE Y-Space'
        p_list.append(p)
    if add_score_plot:
        p = figure(tools=TOOLS, tooltips=TOOLTIPS, plot_height=400, title='Score Scatter')
        p.circle('tx', 'ty', source=source,size=7)
        
        T_aux1=mvmobj['T'][:,[score_plot_xydim[0]-1]]
        T_aux2=mvmobj['T'][:,[score_plot_xydim[1]-1]]
        T_aux = np.hstack((T_aux1,T_aux2))
        st=(T_aux.T @ T_aux)/T_aux.shape[0]
        [xd95,xd99,yd95p,yd95n,yd99p,yd99n]=phi.scores_conf_int_calc(st,mvmobj['T'].shape[0])
        p.line(xd95,yd95p,line_color="gold",line_dash='dashed')
        p.line(xd95,yd95n,line_color="gold",line_dash='dashed')
        p.line(xd99,yd99p,line_color="red",line_dash='dashed')
        p.line(xd99,yd99n,line_color="red",line_dash='dashed') 
        p.xaxis.axis_label = 't ['+str(score_plot_xydim[0])+']'
        p.yaxis.axis_label = 't ['+str(score_plot_xydim[1])+']'
        # Vertical line
        vline = Span(location=0, dimension='height', line_color='black', line_width=2)
        # Horizontal line
        hline = Span(location=0, dimension='width', line_color='black', line_width=2)
        p.renderers.extend([vline, hline])
        #Do another p.figure
        p_list.append(p)
    
    show(column(p_list)) 
    return

def predvsobs(mvmobj,X,Y,*,CLASSID=False,colorby=False,x_space=False):
    """
    Plot observed vs predicted values
    by Salvador Garcia-Munoz 
    (sgarciam@ic.ac.uk ,salvadorgarciamunoz@gmail.com)
    
    mvmobj: A model created with phi.pca or phi.pls
    
    X/Y:     Data [numpy arrays or pandas dataframes] 
    
    optional:
    CLASSID: Pandas Data Frame with classifiers per observation, each column is a class
    
    colorby: one of the classes in CLASSID to color by
    
    x_space: = 'False' will skip plotting the obs. vs pred for X *default*
               'True' will also plot obs vs pred for X
  
    
    """
    num_varX=mvmobj['P'].shape[0]
    
    if isinstance(X,np.ndarray):
        X_=X.copy()
        ObsID_ = []
        for n in list(np.arange(X.shape[0])+1):
            ObsID_.append('Obs #'+str(n))  
        XVarID_ = []
        for n in list(np.arange(X.shape[1])+1):
            XVarID_.append('Var #'+str(n))  
    elif isinstance(X,pd.DataFrame):
        X_=np.array(X.values[:,1:]).astype(float)
        ObsID_ = X.values[:,0].astype(str)
        ObsID_ = ObsID_.tolist()

    if 'varidX' in mvmobj:
        XVar=mvmobj['varidX']
    else:
        XVar = []
        for n in list(np.arange(num_varX)+1):
            XVar.append('XVar #'+str(n))    
                        
    if 'Q' in mvmobj:
        num_varY=mvmobj['Q'].shape[0]
        if 'varidY' in mvmobj:
            YVar=mvmobj['varidY']
        else:
            YVar = []
            for n in list(np.arange(num_varY)+1):
                YVar.append('YVar #'+str(n))
                        

    if isinstance(Y,np.ndarray):
        Y_=Y.copy()
    elif isinstance(Y,pd.DataFrame):
        Y_=np.array(Y.values[:,1:]).astype(float)

            
    if 'Q' in mvmobj:  
        pred=phi.pls_pred(X_,mvmobj)
        yhat=pred['Yhat']
        if x_space:
            xhat=pred['Xhat']
        else:
            xhat=False
    else:
        x_space=True
        pred=phi.pca_pred(X_,mvmobj)
        xhat=pred['Xhat']
        yhat=False
        
    TOOLS = "save,wheel_zoom,box_zoom,pan,reset,box_select,lasso_select"
    TOOLTIPS = [
                ("index", "$index"),
                ("(x,y)", "($x, $y)"),
                ("Obs: ","@ObsID")
                ]
    
    if isinstance(CLASSID,np.bool): # No CLASSIDS
        rnd_num=str(int(np.round(1000*np.random.random_sample())))
        output_file("ObsvsPred_"+rnd_num+".html",title='ObsvsPred')
        plot_counter=0
        
        if not(isinstance(yhat,np.bool)): #skip if PCA model sent
            for i in list(range(Y_.shape[1])):
                x_ = Y_[:,i]
                y_ = yhat[:,i]           
                source = ColumnDataSource(data=dict(x=x_, y=y_,ObsID=ObsID_))
                p = figure(tools=TOOLS, tooltips=TOOLTIPS,plot_width=600, plot_height=600, title=YVar[i])
                p.circle('x', 'y', source=source,size=7,color='darkblue')
                p.line([np.nanmin(x_),np.nanmax(x_)],[np.nanmin(y_),np.nanmax(y_)],line_color='cyan',line_dash='dashed')
                p.xaxis.axis_label ='Observed'
                p.yaxis.axis_label ='Predicted'
                if plot_counter==0:
                    p_list=[p]
                else:
                    p_list.append(p)
                plot_counter = plot_counter+1
                
        if x_space: #
            for i in list(range(X_.shape[1])):
                x_ = X_[:,i]
                y_ = xhat[:,i]           
                source = ColumnDataSource(data=dict(x=x_, y=y_,ObsID=ObsID_))
                p = figure(tools=TOOLS, tooltips=TOOLTIPS,plot_width=600, plot_height=600, title=XVar[i])
                p.circle('x', 'y', source=source,size=7,color='darkblue')
                p.line([np.nanmin(x_),np.nanmax(x_)],[np.nanmin(y_),np.nanmax(y_)],line_color='cyan',line_dash='dashed')
                p.xaxis.axis_label ='Observed'
                p.yaxis.axis_label ='Predicted'
                if plot_counter==0:
                    p_list=[p]
                else:
                    p_list.append(p)
                plot_counter = plot_counter+1
        show(column(p_list))
        
    else: # YES CLASSIDS
        Classes_=np.unique(CLASSID[colorby]).tolist()
        different_colors=len(Classes_)
        colormap =cm.get_cmap("rainbow")
        color_mapping=colormap(np.linspace(0,1,different_colors),1,True)
        bokeh_palette=["#%02x%02x%02x" % (r, g, b) for r, g, b in color_mapping[:,0:3]]  
        rnd_num=str(int(np.round(1000*np.random.random_sample())))
        output_file("ObsvsPred_"+rnd_num+".html",title='ObsvsPred')      
        classid_=list(CLASSID[colorby])
        
        plot_counter=0
        
        if not(isinstance(yhat,np.bool)): #skip if PCA model sent
            for i in list(range(Y_.shape[1])):
                x_ = Y_[:,i]
                y_ = yhat[:,i]
                p = figure(tools=TOOLS, tooltips=TOOLTIPS,plot_width=600, plot_height=600, title=YVar[i])
                for classid_in_turn in Classes_:
                    x_aux=[]
                    y_aux=[]
                    obsid_aux=[]
                    classid_aux=[]
                    for i in list(range(len(ObsID_))):
                        if classid_[i]==classid_in_turn and not(np.isnan(x_[i])):
                            x_aux.append(x_[i])
                            y_aux.append(y_[i])
                            obsid_aux.append(ObsID_[i])
                            classid_aux.append(classid_in_turn)
                    source = ColumnDataSource(data=dict(x=x_aux, y=y_aux,ObsID=obsid_aux,Class=classid_aux))        
                    color_=bokeh_palette[Classes_.index(classid_in_turn)]
                    p.circle('x','y',source=source,color=color_,legend=classid_in_turn)
                    p.line([np.nanmin(x_),np.nanmax(x_)],[np.nanmin(y_),np.nanmax(y_)],line_color='cyan',line_dash='dashed')
                p.xaxis.axis_label ='Observed'
                p.yaxis.axis_label ='Predicted'
                
                if plot_counter==0:
                    p_list=[p]
                    plot_counter = plot_counter+1
                else:
                    p_list.append(p)
              
        if x_space: #
            for i in list(range(X_.shape[1])):
                x_ = X_[:,i]
                y_ = xhat[:,i]
                p = figure(tools=TOOLS, tooltips=TOOLTIPS,plot_width=600, plot_height=600, title=XVar[i])
                for classid_in_turn in Classes_:
                    x_aux=[]
                    y_aux=[]
                    obsid_aux=[]
                    classid_aux=[]
                    for i in list(range(len(ObsID_))):
                        if classid_[i]==classid_in_turn and not(np.isnan(x_[i])):
                            x_aux.append(x_[i])
                            y_aux.append(y_[i])
                            obsid_aux.append(ObsID_[i])
                            classid_aux.append(classid_in_turn)
                    source = ColumnDataSource(data=dict(x=x_aux, y=y_aux,ObsID=obsid_aux,Class=classid_aux))        
                    color_=bokeh_palette[Classes_.index(classid_in_turn)]
                    p.circle('x','y',source=source,color=color_,legend=classid_in_turn)
                    p.line([np.nanmin(x_),np.nanmax(x_)],[np.nanmin(y_),np.nanmax(y_)],line_color='cyan',line_dash='dashed')
                p.xaxis.axis_label ='Observed'
                p.yaxis.axis_label ='Predicted'
                if plot_counter==0:
                    p_list=[p]
                    plot_counter = plot_counter+1
                else:
                    p_list.append(p)
        show(column(p_list))
    return    

def contributions_plot(mvmobj,X,cont_type,*,Y=False,from_obs=False,to_obs=False,lv_space=False):
    """
    Calculate contributions to diagnostics
    by Salvador Garcia-Munoz 
    (sgarciam@ic.ac.uk ,salvadorgarciamunoz@gmail.com)
    
    mvmobj : A dictionary created by phi.pls or phi.pca
    
    X/Y:     Data [numpy arrays or pandas dataframes] - Y space is optional
    
    cont_type: 'ht2'
               'spe'
               'scores'
               
    from_obs: Scalar or list of scalars with observation(s) number(s) | first element is #0
              - OR -
              Strings or list of strings with observation(s) name(s) [if X/Y are pandas data frames]
              Used to off set calculations for scores or ht2
              "False' will calculate with respect to origin *default if not sent*
              
    to_obs: Scalar or list of scalars with observation(s) number(s)| first element is #0
              - OR -
            Strings or list of strings with observation(s) name(s) [if X/Y are pandas data frames]
            To calculate contributions for
            
            *Note: from_obs is ignored when cont_type='spe'*
            
    lv_space: Latent spaces over which to do the calculations [applicable to 'ht2' and 'scores']
    """
    good_to_go=True
    if isinstance(X,pd.DataFrame):
        ObsID=X.values[:,0].tolist()
        if isinstance(to_obs,str):
            to_obs_=ObsID.index(to_obs)
        elif isinstance(to_obs,int):
            to_obs_=to_obs
        elif isinstance(to_obs,list):
            if isinstance(to_obs[0],str):
                to_obs_=[]
                for o in to_obs:
                    to_obs_.append(ObsID.index(o))
            elif isinstance(to_obs[0],int):
                to_obs_=to_obs.copy()
        elif isinstance(to_obs,np.bool):
            good_to_go=False
        if not(isinstance(from_obs,np.bool)):
            if isinstance(from_obs,str):
                from_obs_=ObsID.index(from_obs)
            elif isinstance(from_obs,int):
                from_obs_=from_obs.copy()
            elif isinstance(from_obs,list):
                if isinstance(from_obs[0],str):
                    from_obs_=[]
                    for o in from_obs:
                        from_obs_.append(ObsID.index(o))
                elif isinstance(from_obs[0],int):
                    from_obs_=from_obs.copy()
        else:
            from_obs_=False
    else:
        if isinstance(to_obs,int) or isinstance(to_obs,list):
            to_obs_=to_obs.copy()
        else:
            good_to_go=False    
    if cont_type=='scores' and not(isinstance(Y,np.bool)):
        Y=False
        
    if isinstance(Y,np.bool) and good_to_go:
        Xconts=phi.contributions(mvmobj,X,cont_type,Y=False,from_obs=from_obs_,to_obs=to_obs_,lv_space=lv_space)
        Yconts=False
    elif not(isinstance(Y,np.bool)) and good_to_go and ('Q' in mvmobj) and cont_type=='spe':
        Xconts,Yconts=phi.contributions(mvmobj,X,cont_type,Y=Y,from_obs=from_obs_,to_obs=to_obs_,lv_space=lv_space)
    
    if 'varidX' in mvmobj:
        XVar=mvmobj['varidX']
    else:
        XVar = []
        for n in list(np.arange(mvmobj['P'].shape[0])+1):
            XVar.append('XVar #'+str(n))               

    rnd_num=str(int(np.round(1000*np.random.random_sample())))
    output_file("Contributions"+rnd_num+".html",title='Contributions')
    if isinstance(from_obs,list):
        from_txt=", ".join(map(str, from_obs))
        from_txt=" from obs "+from_txt
    else:
        from_txt=""
    if isinstance(to_obs,list):
        to_txt=", ".join(map(str, to_obs))
        to_txt=" to obs "+to_txt
    else:
        to_txt=""
    
    p = figure(x_range=XVar, plot_height=300,plot_width=500, title="Contributions Plot"+from_txt+to_txt,
                    tools="save,box_zoom,pan,reset")
    p.vbar(x=XVar, top=Xconts[0].tolist(), width=0.5)
    p.xgrid.grid_line_color = None
    p.yaxis.axis_label = 'Contributions to '+cont_type
    hline = Span(location=0, dimension='width', line_color='black', line_width=2)
    p.renderers.extend([hline])
    p_list=[p]
    
    if not(isinstance(Yconts,np.bool)):
        if 'varidY' in mvmobj:
            YVar=mvmobj['varidY']
        else:
            YVar = []
            for n in list(np.arange(mvmobj['Q'].shape[0])+1):
                YVar.append('YVar #'+str(n))               
        
        p = figure(x_range=YVar, plot_height=300,plot_width=500, title="Contributions Plot",
                    tools="save,box_zoom,pan,reset")
        p.vbar(x=YVar, top=Yconts[0].tolist(), width=0.5)
        p.xgrid.grid_line_color = None
        p.yaxis.axis_label = 'Contributions to '+cont_type
        hline = Span(location=0, dimension='width', line_color='black', line_width=2)
        p.renderers.extend([hline])
        p_list.append(p)
        
    show(column(p_list))  
    return

def plot_spectra(X,*,plot_title='Main Title',tab_title='Tab Title',xaxis_label='X- axis',yaxis_label='Y- axis'): 
    rnd_num=str(int(np.round(1000*np.random.random_sample())))                
    output_file("Spectra"+rnd_num+".html",title=tab_title)
    x=X.columns[1:].tolist()
    x=np.array(x)
    x=np.reshape(x,(1,-1))
    y=X.values[:,1:].astype(float)
    p = figure(title=plot_title)
    p.xaxis.axis_label = xaxis_label
    p.yaxis.axis_label = yaxis_label
    p.multi_line(x.tolist()*y.shape[0],y.tolist())
    show(p)
    return
    
    