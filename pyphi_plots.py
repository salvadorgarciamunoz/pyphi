#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 12:01:54 2019

@author: Sal Garcia <sgarciam@ic.ac.uk> <salvadorgarciamunoz@gmail.com>
"""
import numpy as np
from bokeh.io import show, output_file
from bokeh.plotting import figure
from bokeh.palettes import GnBu3
from bokeh.layouts import column
from bokeh.models import ColumnDataSource
from bokeh.models import Span

import matplotlib.cm as cm

def r2pv(mvmobj):
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
        output_file("r2xypv.html",title="R2XYPV") 
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
        output_file("r2xpv.html",title='R2XPV') 
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
        output_file("Loadings X Space.html",title='X Loadings PLS')
        for i in list(np.arange(A)):
            p = figure(x_range=XVar, plot_height=250, title="X Space Loadings "+lv_labels[i],
                    tools="save,box_zoom,pan,reset")
            p.vbar(x=XVar, top=mvmobj['Ws'][:,i].tolist(), width=0.5)
            p.xgrid.grid_line_color = None
            p.yaxis.axis_label = 'W* ['+str(i+1)+']'
            if i==0:
                p_list=[p]
            else:
                p_list.append(p)
        show(column(p_list))        
        output_file("Loadings Y Space.html",title='Y Loadings PLS')
        for i in list(np.arange(A)):
            p = figure(x_range=YVar, plot_height=250, title="Y Space Loadings "+lv_labels[i],
                    tools="save,box_zoom,pan,reset")
            p.vbar(x=YVar, top=mvmobj['Q'][:,i].tolist(), width=0.5)
            p.xgrid.grid_line_color = None
            p.yaxis.axis_label = 'Q ['+str(i+1)+']'
            if i==0:
                p_list=[p]
            else:
                p_list.append(p)                    
        show(column(p_list))
    else:   
        output_file("Loadings X Space.html",title='X Loadings PCA') 
        for i in list(np.arange(A)):
            p = figure(x_range=XVar, plot_height=250, title="X Space Loadings "+lv_labels[i],
                    tools="save,box_zoom,pan,reset")
            p.vbar(x=XVar, top=mvmobj['P'][:,i].tolist(), width=0.5)
            p.xgrid.grid_line_color = None
            p.yaxis.axis_label = 'P ['+str(i+1)+']'
            if i==0:
                p_list=[p]
            else:
                p_list.append(p)
        show(column(p_list))
    return    

def weighted_loadings(mvmobj):
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
        output_file("Loadings X Space.html",title='X Weighted Loadings PLS')
        for i in list(np.arange(A)):
            p = figure(x_range=XVar, plot_height=250, title="X Space Weighted Loadings "+lv_labels[i],
                     tools="save,box_zoom,pan,reset")
            p.vbar(x=XVar, top=(mvmobj['r2xpv'][:,i] * mvmobj['Ws'][:,i]).tolist(), width=0.5)
            p.xgrid.grid_line_color = None
            p.yaxis.axis_label = 'W* ['+str(i+1)+']'
            if i==0:
                p_list=[p]
            else:
                p_list.append(p)
        show(column(p_list))        
        output_file("Loadings Y Space.html",title='Y Weighted Loadings PLS')
        for i in list(np.arange(A)):
            p = figure(x_range=YVar, plot_height=250, title="Y Space Weighted Loadings "+lv_labels[i],
                     tools="save,box_zoom,pan,reset")
            p.vbar(x=YVar, top=(mvmobj['r2ypv'][:,i] * mvmobj['Q'][:,i]).tolist(), width=0.5)
            p.xgrid.grid_line_color = None
            p.yaxis.axis_label = 'Q ['+str(i+1)+']'
            if i==0:
                p_list=[p]
            else:
                p_list.append(p)                    
        show(column(p_list))
    else:   
        output_file("Loadings X Space.html",title='X Weighted Loadings PCA') 
        for i in list(np.arange(A)):
            p = figure(x_range=XVar, plot_height=250, title="X Space Weighted Loadings "+lv_labels[i],
                     tools="save,box_zoom,pan,reset")
            p.vbar(x=XVar, top=(mvmobj['r2xpv'][:,i] * mvmobj['P'][:,i]).tolist(), width=0.5)
            p.xgrid.grid_line_color = None
            p.yaxis.axis_label = 'P ['+str(i+1)+']'
            if i==0:
                p_list=[p]
            else:
                p_list.append(p)
        show(column(p_list))
    return  
 
def vip(mvmobj):
    if 'Q' in mvmobj:  
        output_file("VIP.html",title='VIP Coefficient') 
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

def score_scatter(mvmobj,xydim,*,CLASSID=False,colorby=False,Xnew=False):
    '''
    mvmobj : PLS or PCA object from phyphi
    xydim  : LV to plot on x and y axes. eg [1,2] will plot t1 vs t2
    CLASSID: Pandas DataFrame with CLASSIDS
    colorby: Category (one of the CLASSIDS) to color by
    Xnew   : New data for which to make the score plot this routine evaluates and plots
    '''
    
    if 'obsidX' in mvmobj:
        ObsID_=mvmobj['obsidX']
    else:
        ObsID_ = []
        for n in list(np.arange(mvmobj['T'].shape[0])+1):
            ObsID_.append('Obs #'+str(n))  
 
    if isinstance(CLASSID,np.bool): # No CLASSIDS
        output_file("Score_Scatter.html",title='Score Scatter t['+str(xydim[0])+'] - t['+str(xydim[1])+ ']')
        x_=mvmobj['T'][:,[xydim[0]-1]]
        y_=mvmobj['T'][:,[xydim[1]-1]]
           
        source = ColumnDataSource(data=dict(x=x_, y=y_,ObsID=ObsID_))
        TOOLS = "save,box_zoom,pan,reset,box_select,lasso_select"
        TOOLTIPS = [
                ("index", "$index"),
                ("(x,y)", "($x, $y)"),
                ("Obs: ","@ObsID")
                ]
        
        p = figure(tools=TOOLS, tooltips=TOOLTIPS,plot_width=600, plot_height=600, title='Score Scatter t['+str(xydim[0])+'] - t['+str(xydim[1])+ ']')
        p.circle('x', 'y', source=source,size=7)

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
                       
        output_file("Score_Scatter.html",title='Score Scatter t['+str(xydim[0])+'] - t['+str(xydim[1])+ ']')
        x_=mvmobj['T'][:,[xydim[0]-1]]
        y_=mvmobj['T'][:,[xydim[1]-1]]        
        TOOLS = "save,box_zoom,pan,reset,box_select,lasso_select"
        TOOLTIPS = [
                ("index", "$index"),
                ("(x,y)", "($x, $y)"),
                ("Obs: ","@ObsID"),
                ("Class:","@Class")
                ]        
        classid_=list(CLASSID[colorby])
        color_list_=[]
        for i in list(range(len(ObsID_))):
            color_list_.append(bokeh_palette[Classes_.index(classid_[i])])
        source = ColumnDataSource(data=dict(x=x_, y=y_,ObsID=ObsID_,color_list=color_list_,Class=list(CLASSID[colorby])))
        
        p = figure(tools=TOOLS, tooltips=TOOLTIPS,plot_width=600, plot_height=600, title='Score Scatter t['+str(xydim[0])+'] - t['+str(xydim[1])+ ']')

        p.circle('x', 'y', source=source,size=7,color='color_list')

        p.xaxis.axis_label = 't ['+str(xydim[0])+']'
        p.yaxis.axis_label = 't ['+str(xydim[1])+']'
        # Vertical line
        vline = Span(location=0, dimension='height', line_color='black', line_width=2)
        # Horizontal line
        hline = Span(location=0, dimension='width', line_color='black', line_width=2)
        p.renderers.extend([vline, hline])
        show(p)
    return    
    