#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plots for pyPhi

@author: Sal Garcia <sgarciam@ic.ac.uk> <salvadorgarciamunoz@gmail.com>
Addition Apr 15th        lineplot can make yaxis in log scale
Addition Apr 1st         Improved generic scatter plot and added generic line plot
Addition Mar 27 2025     Contribution plots adjusted to work with Multi_block models
Addition Mar 26 2025     Changed the rotation in Xlabels from 45deg to vertical
Addition Feb 24 2025     Added a generic scatter plot
Addition on Jan 20 2025  Added barplot and lineplot
Addition on Apr 29 2024  Made it compatible with Bokeh 3.4.1 replacing "circle" with "scatter"
Addition on Feb 24 2024  Replaced the randon number in the file names with a time string.
Addition on Feb 21 2024  Added ability to make score plots with a gradient color
                         of nbins based on a numerical value in the classids 
Addition on Jan 18 2024  Added flag to score_scatter to include model scores in plot
                         replaced phi.unique -> np.unique
                         Updated call to maptplotlib colormap to keep it compatible
                         
Addition on Sep 26 2023  All plots are now viewable offline (e.g. in airplane mode)
Addition on May 1 2023   corrected description of mb_vip
Addition on Apr 25 2023  added markersize to score_scatter
Addition on Apr 23 2023  also added the text_alpha flag to loadings map for PCA models
Addition on Apr 22 2023  added tooltips to contribution plots and VIP
                         implemented multiple columns in score scatter (yay!)

Addition on Apr 17 2023  added tpls to the supported models in all loadings, vip, r2pv 
                         and score_scatter plots
Addition on Apr 15 2023, made all loadings, vip, r2pv and score_scatter compatible with
                         lpls and jrpls models
Addition on April 9 2023,  added legends and pan tools to r2pv (yay!)
Addition on April 8 2023, fixed predvsobs to take MB data

Release Nov 15 2021
    * Added "xgrid" flag to all plots using bar plots (loadings, weighted loadings, contributions) to add the Xgrid lines to the plot

Release Jan 15, 2021
    * Added mb_blockweights plot for MBPSL models
    
Release Date: March 30 2020
    * Fixed small syntax change for Bohek to stay compatible

Release Date: Aug 22 2019

What was done:
    
    * This header is now included to track high level changes 
    
"""
import numpy as np
from bokeh.io import show, output_file
from bokeh.plotting import figure
from bokeh.layouts import column
from bokeh.models import ColumnDataSource,LabelSet,Span,Legend
import pyphi as phi
import pandas as pd
from datetime import datetime
import math

#import matplotlib.cm as cm
import matplotlib

def timestr():
    now=datetime.now()
    return now.strftime("%Y%m%d%H%M%S%f")
    

def r2pv(mvm_obj,*,plotwidth=600,plotheight=400,addtitle='',material=False,zspace=False):
    """R2 per variable per component plots
    
    r2pv(mvm_obj,*,plotwidth=600,plotheight=400,addtitle='',material=False,zspace=False)
    
    Args:
        mvm_obj: A model created with phi.pca or phi.pls
    
    Other Parameters:
        material: To obtain the plot for the properties of a specific material
                  When doing this for a JRPLS or TPLS mode
        zspace:  If True will display the r2pv for the process space in a TPLS model
        
    by Salvador Garcia-Munoz 
    (sgarciam@ic.ac.uk ,salvadorgarciamunoz@gmail.com)
    """
    mvmobj=mvm_obj.copy()
    A= mvmobj['T'].shape[1]
    yaxlbl='X'
    if (mvmobj['type']=='lpls') or (mvmobj['type']=='jrpls') or (mvmobj['type']=='tpls'):
        if ((mvmobj['type']=='jrpls') or (mvmobj['type']=='tpls')) and not(isinstance(material, bool) ):
            mvmobj['r2xpv']=mvmobj['r2xpvi'][mvmobj['materials'].index(material)]
            mvmobj['varidX']=mvmobj['varidXi'][mvmobj['materials'].index(material) ]
        elif (mvmobj['type']=='tpls') and zspace :
            mvmobj['r2xpv']=mvmobj['r2zpv']
            mvmobj['varidX']=mvmobj['varidZ']
            yaxlbl='Z'
    else:
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
        #rnd_num=str(int(np.round(1000*np.random.random_sample())))
        rnd_num=timestr()
        output_file("r2xypv_"+rnd_num+".html",title="R2"+ yaxlbl+ "YPV",mode='inline') 
        #colormap =cm.get_cmap("rainbow")
        colormap = matplotlib.colormaps['rainbow']
        different_colors=A
        color_mapping=colormap(np.linspace(0,1,different_colors),1,True)
        bokeh_palette=["#%02x%02x%02x" % (r, g, b) for r, g, b in color_mapping[:,0:3]]  
            
        px = figure(x_range=XVar, title="R2"+ yaxlbl+" Per Variable "+addtitle,
             tools="save,box_zoom,xpan,hover,reset", tooltips="$name @XVar: @$name",width=plotwidth,height=plotheight)
        
        v=px.vbar_stack(lv_labels, x='XVar', width=0.9,color=bokeh_palette,source=r2pvX_dict)
        px.y_range.range_padding = 0.1
        px.ygrid.grid_line_color = None
        px.xgrid.grid_line_color = None
        px.axis.minor_tick_line_color = None
        px.outline_line_color = None
        px.yaxis.axis_label = 'R2'+ yaxlbl
        px.xaxis.major_label_orientation = math.pi/2

        
        legend = Legend(items=[(x, [v[i]]) for i, x in enumerate(lv_labels)], location=(0, 0))
        px.add_layout(legend, 'right')
        py = figure(x_range=YVar, height=plotheight, title="R2Y Per Variable "+addtitle,
            tools="save,box_zoom,xpan,hover,reset", tooltips="$name @YVar: @$name",width=plotwidth)
        
        v=py.vbar_stack(lv_labels, x='YVar', width=0.9,color=bokeh_palette,source=r2pvY_dict)
        py.y_range.range_padding = 0.1
        py.ygrid.grid_line_color = None
        py.axis.minor_tick_line_color = None
        py.xgrid.grid_line_color = None
        py.outline_line_color = None
        py.yaxis.axis_label = 'R2Y'
        py.xaxis.major_label_orientation = math.pi/2
        legend = Legend(items=[(x, [v[i]]) for i, x in enumerate(lv_labels)], location=(0, 0))
        py.add_layout(legend, 'right')
        show(column(px,py))
        
        
    else:   
        #rnd_num=str(int(np.round(1000*np.random.random_sample())))
        rnd_num=timestr()
        output_file("r2xpv_"+rnd_num+".html",title='R2XPV',mode='inline') 
        #colormap =cm.get_cmap("rainbow")
        colormap = matplotlib.colormaps['rainbow']
        different_colors=A
        color_mapping=colormap(np.linspace(0,1,different_colors),1,True)
        bokeh_palette=["#%02x%02x%02x" % (r, g, b) for r, g, b in color_mapping[:,0:3]]  
               
        p = figure(x_range=XVar, title="R2X Per Variable "+addtitle,
             tools="save,box_zoom,xpan,hover,reset", tooltips="$name @XVar: @$name",width=plotwidth,height=plotheight)
        
        v=p.vbar_stack(lv_labels, x='XVar', width=0.9,color=bokeh_palette,source=r2pvX_dict)
        legend = Legend(items=[(x, [v[i]]) for i, x in enumerate(lv_labels)], location=(0, 0))
        p.y_range.range_padding = 0.1
        p.ygrid.grid_line_color = None
        p.axis.minor_tick_line_color = None
        p.outline_line_color = None
        p.yaxis.axis_label = 'R2X'
        p.xaxis.major_label_orientation = math.pi/2
        p.add_layout(legend, 'right')
        show(p)
    return
    
def loadings(mvm_obj,*,plotwidth=600,xgrid=False,addtitle='',material=False,zspace=False):
    """Column plots of loadings
    
    loadings(mvm_obj,*,plotwidth=600,xgrid=False,addtitle='',material=False,zspace=False)
    
    Args:
        mvm_obj: A PCA,PLS, LPLS, JPLS or TPLS model 
        
    Other Parameters:
        material: To obtain the plot for the properties of a specific material
                  When doing this for a JRPLS or TPLS mode     
        zspace:  If True will display the plot for the process space in a TPLS model

    by Salvador Garcia-Munoz 
    (sgarciam@ic.ac.uk ,salvadorgarciamunoz@gmail.com)
    """
    mvmobj=mvm_obj.copy()
    space_lbl='X'
    A= mvmobj['T'].shape[1]
    if (mvmobj['type']=='lpls') or (mvmobj['type']=='jrpls') or (mvmobj['type']=='tpls'):
        loading_lbl='S*'
        if (mvmobj['type']=='lpls'):
            mvmobj['Ws']=mvmobj['Ss']
        if isinstance(material, bool) and not(zspace):
            mvmobj['Ws']=mvmobj['Ss']
        if ((mvmobj['type']=='jrpls') or (mvmobj['type']=='tpls') ) and not(isinstance(material, bool) ):
            mvmobj['Ws']=mvmobj['Ssi'][mvmobj['materials'].index(material)]
            mvmobj['varidX']=mvmobj['varidXi'][mvmobj['materials'].index(material) ]
            
        elif (mvmobj['type']=='tpls') and zspace :
            mvmobj['varidX']=mvmobj['varidZ']
            loading_lbl='Wz*'
            space_lbl='Z'
        
    else:
        num_varX=mvmobj['P'].shape[0]    
        loading_lbl='W*'
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
            
    TOOLS = "save,wheel_zoom,box_zoom,pan,reset,box_select,lasso_select"
    TOOLTIPS = [
                ("Variable:","@names")
                ]
      
    if is_pls:
        #rnd_num=str(int(np.round(1000*np.random.random_sample())))
        rnd_num=timestr()
        output_file("Loadings "+space_lbl+" Space_"+rnd_num+".html",title=space_lbl+' Loadings PLS',mode='inline')
        for i in list(np.arange(A)):
            p = figure(x_range=XVar, title=space_lbl+" Space Loadings "+lv_labels[i]+addtitle,
                    tools=TOOLS,tooltips=TOOLTIPS,width=plotwidth)
            source1 = ColumnDataSource(data=dict(x_=XVar, y_=mvmobj['Ws'][:,i].tolist(),names=XVar)) 
            
            #p.vbar(x=XVar, top=mvmobj['Ws'][:,i].tolist(), width=0.5)
            p.vbar(x='x_', top='y_', source=source1,width=0.5)
            p.ygrid.grid_line_color = None    
            if xgrid:
                p.xgrid.grid_line_color = 'lightgray'
                
            else:
                p.xgrid.grid_line_color = None    
                
            p.yaxis.axis_label = loading_lbl+' ['+str(i+1)+']'
            hline = Span(location=0, dimension='width', line_color='black', line_width=2)
            p.renderers.extend([hline])
            p.xaxis.major_label_orientation = math.pi/2
            if i==0:
                p_list=[p]
            else:
                p_list.append(p)
        show(column(p_list))    
        #rnd_num=str(int(np.round(1000*np.random.random_sample())))
        rnd_num=timestr()
        output_file("Loadings Y Space_"+rnd_num+".html",title='Y Loadings PLS',mode='inline')
        for i in list(np.arange(A)):
            p = figure(x_range=YVar, title="Y Space Loadings "+lv_labels[i]+addtitle,
                    tools="save,box_zoom,pan,reset",tooltips=TOOLTIPS,width=plotwidth)
            
            source1 = ColumnDataSource(data=dict(x_=YVar, y_=mvmobj['Q'][:,i].tolist(),names=YVar)) 
            #p.vbar(x=YVar, top=mvmobj['Q'][:,i].tolist(), width=0.5)
            p.vbar(x='x_', top='y_', source=source1,width=0.5)
            p.ygrid.grid_line_color = None    
            if xgrid:
                p.xgrid.grid_line_color = 'lightgray'
            else:
                p.xgrid.grid_line_color = None    
            p.yaxis.axis_label = 'Q ['+str(i+1)+']'
            hline = Span(location=0, dimension='width', line_color='black', line_width=2)
            p.renderers.extend([hline])
            p.xaxis.major_label_orientation = math.pi/2
            if i==0:
                p_list=[p]
            else:
                p_list.append(p)                    
        show(column(p_list))
    else:   
        #rnd_num=str(int(np.round(1000*np.random.random_sample())))
        rnd_num=timestr()
        output_file("Loadings X Space_"+rnd_num+".html",title='X Loadings PCA',mode='inline') 
        for i in list(np.arange(A)):
            source1 = ColumnDataSource(data=dict(x_=XVar, y_=mvmobj['P'][:,i].tolist(),names=XVar))  
            
            p = figure(x_range=XVar, title="X Space Loadings "+lv_labels[i]+addtitle,
                    tools=TOOLS,tooltips=TOOLTIPS,width=plotwidth)
            
            #p.vbar(x=XVar, top=mvmobj['P'][:,i].tolist(), width=0.5)
            
            p.vbar(x='x_', top='y_', source=source1,width=0.5)
            if xgrid:
                p.xgrid.grid_line_color = 'lightgray'
            else:
                p.xgrid.grid_line_color = None    
            p.yaxis.axis_label = 'P ['+str(i+1)+']'
            hline = Span(location=0, dimension='width', line_color='black', line_width=2)
            p.renderers.extend([hline])
            p.xaxis.major_label_orientation = math.pi/2
            if i==0:
                p_list=[p]
            else:
                p_list.append(p)
        show(column(p_list))
    return    

def loadings_map(mvm_obj,dims,*,plotwidth=600,addtitle='',material=False,zspace=False,textalpha=0.75):
    """Scatter plot overlaying X and Y loadings 
    
    loadings_map(mvm_obj,dims,*,plotwidth=600,addtitle='',material=False,zspace=False,textalpha=0.75)
    
    Args:
        mvm_obj: A PCA,PLS, LPLS, JPLS or TPLS model 
        dims: Dimensions to plot in x and y axis (e.g. [1,2])
        
    Other Parameters:
        material: To obtain the plot for the properties of a specific material
          When doing this for a JRPLS or TPLS mode     
          zspace:  If True will display the plot for the process space in a TPLS model
    by Salvador Garcia-Munoz 
    (sgarciam@ic.ac.uk ,salvadorgarciamunoz@gmail.com)
    
    """
    mvmobj=mvm_obj.copy()
    A= mvmobj['T'].shape[1]
    if (mvmobj['type']=='lpls') or (mvmobj['type']=='jrpls') or (mvmobj['type']=='tpls'):
     
        if (mvmobj['type']=='lpls'):
            mvmobj['Ws']=mvmobj['Ss']
        if isinstance(material, bool) and not(zspace):
            mvmobj['Ws']=mvmobj['Ss']
        if ((mvmobj['type']=='jrpls') or (mvmobj['type']=='tpls')) and not(isinstance(material, bool) ):
            mvmobj['Ws']=mvmobj['Ssi'][mvmobj['materials'].index(material)]
            mvmobj['varidX']=mvmobj['varidXi'][mvmobj['materials'].index(material) ]
        elif (mvmobj['type']=='tpls') and zspace :
            mvmobj['varidX']=mvmobj['varidZ']
            
    else:
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
    
        #rnd_num=str(int(np.round(1000*np.random.random_sample())))
        rnd_num=timestr()
        output_file("Loadings Map"+rnd_num+".html",title='Loadings Map',mode='inline')
       
    
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
        p = figure(tools=TOOLS, tooltips=TOOLTIPS,width=plotwidth, title="Loadings Map LV["+str(dims[0])+"] - LV["+str(dims[1])+"] "+addtitle,
                                                                                                          x_range=(-1.5,1.5),y_range=(-1.5,1.5))
        #p.circle('x', 'y', source=source1,size=10,color='darkblue')
        #p.circle('x', 'y', source=source2,size=10,color='red')
        
        p.scatter(x='x',y= 'y', source=source1,size=10,color='darkblue')
        p.scatter('x', 'y', source=source2,size=10,color='red')
        
        p.xaxis.axis_label = lv_labels [dims[0]-1]
        p.yaxis.axis_label = lv_labels [dims[1]-1]
        
        labelsX = LabelSet(x='x', y='y', text='names', 
                           level='glyph',x_offset=5, y_offset=5, 
                           source=source1,text_color='darkgray',
                           text_alpha=textalpha )
        labelsY = LabelSet(x='x', y='y', text='names', 
                           level='glyph',x_offset=5, y_offset=5, 
                           source=source2,text_color='darkgray',
                           text_alpha=textalpha )
        p.add_layout(labelsX)
        p.add_layout(labelsY)

        vline = Span(location=0, dimension='height', line_color='black', line_width=2)
        # Horizontal line
        hline = Span(location=0, dimension='width', line_color='black', line_width=2)
        p.renderers.extend([vline, hline])
        show(p)    
    else:
        lv_prefix='PC #'     
        lv_labels = []   
        for a in list(np.arange(A)+1):
            lv_labels.append(lv_prefix+str(a))    
        if 'varidX' in mvmobj:
            XVar=mvmobj['varidX']
        else:
            XVar = []
            for n in list(np.arange(num_varX)+1):
                XVar.append('XVar #'+str(n))                   
        #rnd_num=str(int(np.round(1000*np.random.random_sample())))
        rnd_num=timestr()
        output_file("Loadings Map"+rnd_num+".html",title='Loadings Map',mode='inline')    
        x_p = mvmobj['P'][:,dims[0]-1]
        y_p = mvmobj['P'][:,dims[1]-1]                        
        TOOLS = "save,wheel_zoom,box_zoom,pan,reset,box_select,lasso_select"
        TOOLTIPS = [
                ("index", "$index"),
                ("(x,y)", "($x, $y)"),
                ("Variable:","@names")
                ]
    
        source1 = ColumnDataSource(data=dict(x=x_p, y=y_p,names=XVar))  
        p = figure(tools=TOOLS, tooltips=TOOLTIPS,width=plotwidth, title="Loadings Map PC["+str(dims[0])+"] - PC["+str(dims[1])+"] "+addtitle,                                                                                                         x_range=(-1.5,1.5),y_range=(-1.5,1.5))
        #p.circle('x', 'y', source=source1,size=10,color='darkblue')
        p.scatter(x='x',y='y', source=source1,size=10,color='darkblue')
        p.xaxis.axis_label = lv_labels [dims[0]-1]
        p.yaxis.axis_label = lv_labels [dims[1]-1]        
        labelsX = LabelSet(x='x', y='y', text='names', level='glyph',x_offset=5, y_offset=5, source=source1, 
                          text_color='darkgray',text_alpha=textalpha)
        p.add_layout(labelsX)
        vline = Span(location=0, dimension='height', line_color='black', line_width=2)
        # Horizontal line
        hline = Span(location=0, dimension='width', line_color='black', line_width=2)
        p.renderers.extend([vline, hline])
        show(p)            
    return  

def weighted_loadings(mvm_obj,*,plotwidth=600,xgrid=False,addtitle='',material=False,zspace=False):
    """Column plots of loadings weighted by r2x/r2y correspondingly
    
    weighted_loadings(mvm_obj,*,plotwidth=600,xgrid=False,addtitle='',material=False,zspace=False):
    
    Args:
        mvm_obj: A PCA,PLS, LPLS, JPLS or TPLS model 
        
    Other Parameters:
        material: To obtain the plot for the properties of a specific material
                  When doing this for a JRPLS or TPLS mode     
        zspace:  If True will display the plot for the process space in a TPLS model

    by Salvador Garcia-Munoz 
    (sgarciam@ic.ac.uk ,salvadorgarciamunoz@gmail.com)
    """
    mvmobj=mvm_obj.copy()
    A= mvmobj['T'].shape[1]
    space_lbl='X'
    if (mvmobj['type']=='lpls') or (mvmobj['type']=='jrpls') or (mvmobj['type']=='tpls'):
        loading_lbl='S*'
        if (mvmobj['type']=='lpls'):
            mvmobj['Ws']=mvmobj['Ss']
        if isinstance(material, bool) and not(zspace):
            mvmobj['Ws']=mvmobj['Ss']
        if ((mvmobj['type']=='jrpls')  or (mvmobj['type']=='tpls')) and not(isinstance(material, bool) ):
            mvmobj['Ws']=mvmobj['Ssi'][mvmobj['materials'].index(material)]
            mvmobj['varidX']=mvmobj['varidXi'][mvmobj['materials'].index(material) ]
            mvmobj['r2xpv']=mvmobj['r2xpvi'][mvmobj['materials'].index(material) ]
        elif (mvmobj['type']=='tpls') and zspace:
            mvmobj['varidX']=mvmobj['varidZ']
            mvmobj['r2xpv']=mvmobj['r2zpv']
            loading_lbl='Wz*'
            space_lbl='Z'
    else:
        num_varX=mvmobj['P'].shape[0]     
        loading_lbl='W*'
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
            
    TOOLS = "save,wheel_zoom,box_zoom,pan,reset,box_select,lasso_select"
    TOOLTIPS = [
                ("Variable:","@names")
                ]
    
    if is_pls:
        #rnd_num=str(int(np.round(1000*np.random.random_sample())))
        rnd_num=timestr()
        output_file("Loadings "+space_lbl+" Space_"+rnd_num+".html",title=space_lbl+' Weighted Loadings PLS',mode='inline')
        for i in list(np.arange(A)):
            p = figure(x_range=XVar, title=space_lbl+" Space Weighted Loadings "+lv_labels[i]+addtitle,
                     tools=TOOLS,tooltips=TOOLTIPS,width=plotwidth)
            source1 = ColumnDataSource(data=dict(x_=XVar, y_=(mvmobj['r2xpv'][:,i] * mvmobj['Ws'][:,i]).tolist(),names=XVar)) 
             
            #p.vbar(x=XVar, top=(mvmobj['r2xpv'][:,i] * mvmobj['Ws'][:,i]).tolist(), width=0.5)
            p.vbar(x='x_', top='y_', source=source1,width=0.5)
            p.ygrid.grid_line_color = None    
            if xgrid:
                p.xgrid.grid_line_color = 'lightgray'
            else:
                p.xgrid.grid_line_color = None    

            p.yaxis.axis_label = loading_lbl+' x R2'+space_lbl+' ['+str(i+1)+']'
            hline = Span(location=0, dimension='width', line_color='black', line_width=2)
            p.renderers.extend([hline])
            p.xaxis.major_label_orientation = math.pi/2
            if i==0:
                p_list=[p]
            else:
                p_list.append(p)
        show(column(p_list)) 
        #rnd_num=str(int(np.round(1000*np.random.random_sample())))
        rnd_num=timestr()
        output_file("Loadings Y Space_"+rnd_num+".html",title='Y Weighted Loadings PLS',mode='inline')
        for i in list(np.arange(A)):
            p = figure(x_range=YVar, title="Y Space Weighted Loadings "+lv_labels[i]+addtitle,
                     tools=TOOLS,tooltips=TOOLTIPS,width=plotwidth)
            source1 = ColumnDataSource(data=dict(x_=YVar, y_=(mvmobj['r2ypv'][:,i] * mvmobj['Q'][:,i]).tolist(),names=YVar)) 
            
            #p.vbar(x=YVar, top=(mvmobj['r2ypv'][:,i] * mvmobj['Q'][:,i]).tolist(), width=0.5)
            p.vbar(x='x_', top='y_', source=source1,width=0.5)
            p.ygrid.grid_line_color = None    
            if xgrid:
                p.xgrid.grid_line_color = 'lightgray'
            else:
                p.xgrid.grid_line_color = None    
            p.yaxis.axis_label = 'Q x R2Y ['+str(i+1)+']'
            hline = Span(location=0, dimension='width', line_color='black', line_width=2)
            p.renderers.extend([hline])
            p.xaxis.major_label_orientation = math.pi/2
            if i==0:
                p_list=[p]
            else:
                p_list.append(p)                    
        show(column(p_list))
    else:   
        #rnd_num=str(int(np.round(1000*np.random.random_sample())))
        rnd_num=timestr()
        output_file("Loadings X Space_"+rnd_num+".html",title='X Weighted Loadings PCA',mode='inline') 
        for i in list(np.arange(A)):
            p = figure(x_range=XVar, title="X Space Weighted Loadings "+lv_labels[i]+addtitle,
                     tools=TOOLS,tooltips=TOOLTIPS,width=plotwidth)
            source1 = ColumnDataSource(data=dict(x_=XVar, y_=(mvmobj['r2xpv'][:,i] * mvmobj['P'][:,i]).tolist(),names=XVar)) 
            
            #p.vbar(x=XVar, top=(mvmobj['r2xpv'][:,i] * mvmobj['P'][:,i]).tolist(), width=0.5)
            p.vbar(x='x_', top='y_', source=source1,width=0.5)
            p.ygrid.grid_line_color = None    
            if xgrid:
                p.xgrid.grid_line_color = 'lightgray'
            else:
                p.xgrid.grid_line_color = None    

            p.yaxis.axis_label = 'P x R2X['+str(i+1)+']'
            hline = Span(location=0, dimension='width', line_color='black', line_width=2)
            p.renderers.extend([hline])
            p.xaxis.major_label_orientation = math.pi/2
            if i==0:
                p_list=[p]
            else:
                p_list.append(p)
        show(column(p_list))
    return  
 
def vip(mvm_obj,*,plotwidth=600,material=False,zspace=False,addtitle=''):
    """Very Important to the Projection (VIP) plot
    
    vip(mvm_obj,*,plotwidth=600,material=False,zspace=False,addtitle='')
    
    Args:
        mvm_obj: A PLS, LPLS, JPLS or TPLS model 
        
    Other Parameters:
        material: To obtain the plot for the properties of a specific material
                  When doing this for a JRPLS or TPLS mode     
        zspace:  If True will display the vip for the process space in a TPLS model

        by Salvador Garcia-Munoz 
    (sgarciam@ic.ac.uk ,salvadorgarciamunoz@gmail.com)
    
    mvm_obj: A model created with phi.pls
    """
    mvmobj=mvm_obj.copy()
    if 'Q' in mvmobj:  
        if (mvmobj['type']=='lpls') or (mvmobj['type']=='jrpls') or (mvmobj['type']=='tpls'):
            if (mvmobj['type']=='lpls'):
                mvmobj['Ws']=mvmobj['Ss']
            if isinstance(material, bool) and not(zspace):
                mvmobj['Ws']=mvmobj['Ss']
            if ((mvmobj['type']=='jrpls') or (mvmobj['type']=='tpls')) and not(isinstance(material, bool) ):
                mvmobj['Ws']=mvmobj['Ssi'][mvmobj['materials'].index(material)]
                mvmobj['varidX']=mvmobj['varidXi'][mvmobj['materials'].index(material) ]
            elif (mvmobj['type']=='tpls') and zspace:
                mvmobj['varidX']=mvmobj['varidZ']
            
        else:
            num_varX=mvmobj['P'].shape[0] 
        #rnd_num=str(int(np.round(1000*np.random.random_sample())))
        rnd_num=timestr()
        output_file("VIP_"+rnd_num+".html",title='VIP Coefficient',mode='inline') 
                   
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
            
        TOOLTIPS = [
                    ("Variable","@names")
                    ]
        
        p = figure(x_range=sorted_XVar, title="VIP "+addtitle,
            tools="save,box_zoom,pan,reset",tooltips=TOOLTIPS,width=plotwidth)
        source1 = ColumnDataSource(data=dict(x_=sorted_XVar, y_=vip.tolist(),names=sorted_XVar)) 
        #p.vbar(x=sorted_XVar, top=vip.tolist(), width=0.5)
        p.vbar(x='x_', top='y_', source=source1,width=0.5)
        p.xgrid.grid_line_color = None
        p.yaxis.axis_label = 'Very Important to the Projection'
        p.xaxis.major_label_orientation = math.pi/2
        show(p)
    return    

def _create_classid_(df,column,*,nbins=5):
    '''
    Internal routine to create a CLASSID dataframe from values in a column
    '''
    
    nan_map=np.isnan(df[column].values )
    hist,bin_edges=np.histogram(df[column].values[np.logical_not(np.isnan(df[column].values))],bins=nbins )
    range_list=[]
    for i,e in enumerate(bin_edges[:-1]):
        range_list.append(str(np.round(bin_edges[i],3))+' to '+ str(np.round(bin_edges[i+1],3)))
    #range_list.append('NaN')
    bin_edges_=bin_edges.copy()
    bin_edges_[-1]=bin_edges_[-1]+0.1
    membership_=np.digitize(df[column].values[np.logical_not(np.isnan(df[column].values))],bin_edges_)
    membership=[]
    #for m in membership_:
    non_nan_counter=0
    for isnan in nan_map:
        if not(isnan):
            m=membership_[non_nan_counter]
            membership.append(range_list[m-1])
            non_nan_counter=non_nan_counter+1
        else:
            membership.append('Missing Value')
            
    classid_df=df[df.columns[0]].to_frame()
    classid_df.insert(1,column,membership)
    return classid_df

def score_scatter(mvm_obj,xydim,*,CLASSID=False,colorby=False,Xnew=False,
                  add_ci=False,add_labels=False,add_legend=True,legend_cols=1, 
                  addtitle='',plotwidth=600,plotheight=600,
                  rscores=False,material=False,marker_size=7,nbins=False,include_model=False):
    '''Score scatter plot
    
    score_scatter(mvm_obj,xydim,*,CLASSID=False,colorby=False,nbins=False,Xnew=False,
                      add_ci=False,add_labels=False,add_legend=True,legend_cols=1, 
                      addtitle='',plotwidth=600,plotheight=600,
                      rscores=False,material=False,marker_size=7,include_model=False):
    Args:
        mvm_obj     : PLS or PCA object from phyphi
        xydim      : LV to plot on x and y axes. eg [1,2] will plot t1 vs t2
        CLASSID    : Pandas DataFrame with CLASSIDS
        colorby    : Category (one of the CLASSIDS) to color by
        Xnew       : New data for which to make the score plot this routine evaluates and plots
        nbins      : Number of groups to use when color coding by a numeric value
        add_ci     : when = True will add confidence intervals
        add_labels : When = True labels each point with Obs ID
        add_legend : When = True will add a legend with classid labels
        legend_cols: Number of columns for legend
        addtitle   : Additional text to be added to title
        plotwidth  : If omitted, width is 600
        plotheight : If omitted, height is 600
        rscores    : Plot scores for all material space in lpls|jrpls|tpls
        material   : Label for specific material to plot scores for in lpls|jrpls|tpls 
        include_model: Will plot model scores in gray as backgrpound
        
    by Salvador Garcia Munoz (sgarciam@imperial.ac.uk salvadorgarciamunoz@gmail.com)
    '''
    if not(isinstance(nbins, bool)):
         if colorby in CLASSID.columns.to_list():
             classid_by_var = _create_classid_(CLASSID,colorby,nbins=nbins)
             CLASSID = classid_by_var.copy()
             
    mvmobj=mvm_obj.copy()
    if ((mvmobj['type']=='lpls') or  (mvmobj['type']=='jrpls')  or  (mvmobj['type']=='tpls')) and (not(isinstance(Xnew,bool))):    
        Xnew=False
        print('score scatter does not take Xnew for jrpls or lpls for now')
        
    if isinstance(Xnew,bool):
        
        if 'obsidX' in mvmobj:
            ObsID_=mvmobj['obsidX']
        else:
            ObsID_ = []
            for n in list(np.arange(mvmobj['T'].shape[0])+1):
                ObsID_.append('Obs #'+str(n))  
        T_matrix=mvmobj['T']    
        
        if not(rscores):        
            if (mvmobj['type']=='lpls'):
                ObsID_=mvmobj['obsidR']
            if (mvmobj['type']=='jrpls') or (mvmobj['type']=='tpls') :   
                 ObsID_=mvmobj['obsidRi'][0]
        else:
            if (mvmobj['type']=='lpls'):
                ObsID_=mvmobj['obsidX']
                T_matrix=mvmobj['Rscores']
            if (mvmobj['type']=='jrpls') or (mvmobj['type']=='tpls')  : 
                if isinstance(material,bool):
                    allobsids=[y for x in mvmobj['obsidXi'] for y in x]
                    ObsID_=allobsids
                    clssid_obs=[]
                    clssid_class=[]
                    for i,R_ in enumerate(mvmobj['Rscores']):
                        clssid_obs.extend(mvmobj['obsidXi'][i])
                        clssid_class.extend([mvmobj['materials'][i]]*len( mvmobj['obsidXi'][i]))
                        if i==0:
                            allrscores=R_
                        else:
                            allrscores=np.vstack((allrscores,R_))
                    classid=pd.DataFrame(clssid_class,columns=['material'])
                    classid.insert(0,'obs',clssid_obs)
                    CLASSID=classid
                    colorby='material'
                    T_matrix=allrscores
                else:
                    ObsID_ = mvmobj['obsidXi'][mvmobj['materials'].index(material) ]
                    T_matrix = mvmobj['Rscores'][mvmobj['materials'].index(material) ]

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
    
    if include_model:
        if 'obsidX' in mvmobj:
            ObsID__=mvmobj['obsidX'].copy()
        else:
            ObsID__ = []
            for n in list(np.arange(mvmobj['T'].shape[0])+1):
                ObsID__.append('Model Obs #'+str(n))  
        T_matrix_=mvmobj['T'].copy()    
      
        if isinstance(CLASSID,bool): #If there are no classids I need to create one
            source=(['Model']*T_matrix_.shape[0])
            source.extend(['New']*T_matrix.shape[0])
            ObsID__.extend(ObsID_)
            CLASSID=pd.DataFrame.from_dict( {'ObsID':ObsID__,'_Source_':source })
            colorby='_Source_'
        else: #IF there are I need to augment it
            source=['Model']*T_matrix_.shape[0]
            CLASSID_=pd.DataFrame.from_dict( {CLASSID.columns[0]:ObsID__,colorby:source })
            ObsID__.extend(ObsID_)
            CLASSID = pd.concat([CLASSID_,CLASSID])
        ObsID_=ObsID__.copy()    
        T_matrix=np.vstack((T_matrix_,T_matrix ))
    ObsNum_=[]    
    for n in list(range(1,len(ObsID_)+1)):
                ObsNum_.append(str(n))  
    
    if isinstance(CLASSID,bool): # No CLASSIDS
        #rnd_num=str(int(np.round(1000*np.random.random_sample())))
        rnd_num=timestr()
        output_file("Score_Scatter_"+rnd_num+".html",title='Score Scatter t['+str(xydim[0])+'] - t['+str(xydim[1])+ ']',mode='inline')

        x_=T_matrix[:,[xydim[0]-1]]
        y_=T_matrix[:,[xydim[1]-1]]

           
        source = ColumnDataSource(data=dict(x=x_, y=y_,ObsID=ObsID_,ObsNum=ObsNum_))
        TOOLS = "save,wheel_zoom,box_zoom,pan,reset,box_select,lasso_select"
        TOOLTIPS = [
                ("Obs #", "@ObsNum"),
                ("(x,y)", "($x, $y)"),
                ("Obs: ","@ObsID")
                ]
        
        p = figure(tools=TOOLS, tooltips=TOOLTIPS,width=plotwidth,height=plotheight, title='Score Scatter t['+str(xydim[0])+'] - t['+str(xydim[1])+ '] '+addtitle)
        #p.circle('x', 'y', source=source,size=marker_size)
        p.scatter('x', 'y', source=source,size=marker_size)
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
            
        if add_labels:
            labelsX = LabelSet(x='x', y='y', text='ObsID', level='glyph',x_offset=5, y_offset=5, source=source)
            p.add_layout(labelsX)
        if not(rscores):    
            p.xaxis.axis_label = 't ['+str(xydim[0])+']'
            p.yaxis.axis_label = 't ['+str(xydim[1])+']'
        else:
            p.xaxis.axis_label = 'r ['+str(xydim[0])+']'
            p.yaxis.axis_label = 'r ['+str(xydim[1])+']'
        # Vertical line
        vline = Span(location=0, dimension='height', line_color='black', line_width=2)
        # Horizontal line
        hline = Span(location=0, dimension='width', line_color='black', line_width=2)
        p.renderers.extend([vline, hline])
        show(p)      
    else: # YES CLASSIDS   
        Classes_=phi.unique(CLASSID,colorby)
        A=len(Classes_)
        colormap = matplotlib.colormaps['rainbow']
        different_colors=A
        color_mapping=colormap(np.linspace(0,1,different_colors),1,True)
        #Test code to overwrite "Model" Category with light Cyan
        if Classes_[0]=='Model':
            color_mapping=colormap(np.linspace(0,1,different_colors-1),1,True)
            color_mapping=np.vstack((np.array([225,225,225,255]),color_mapping))
            
            
        if not(isinstance(nbins, bool)):
             if colorby in CLASSID.columns.to_list():
                 #Classes_=phi.unique(CLASSID,colorby)
                 Classes_=np.unique(CLASSID[colorby]).tolist()
                 item_o=[]
                 missing_value_class_exists=False
                 for item in Classes_:
                     if item!='Missing Value':
                         item_o.append(float(item[:item.find(' ')]))
                     else:
                         missing_value_class_exists=True
                 idx=np.argsort(item_o)
                 if missing_value_class_exists:
                     idx=np.append(Classes_.index('Missing Value'),idx)
                 Classes_o=[]
                 for i in idx:
                     Classes_o.append(Classes_[i])
                 Classes_=Classes_o.copy()    
                 #print(Classes_)
                 A=len(Classes_)
                 #colormap =cm.get_cmap("rainbow")
                 colormap = matplotlib.colormaps['viridis']
                 different_colors=A
                 color_mapping=colormap(np.linspace(0,1,different_colors),1,True)
                 if (Classes_[0]=='Model') or (Classes_[0]=='Missing Value'):
                     color_mapping=colormap(np.linspace(0,1,different_colors-1),1,True)
                     color_mapping=np.vstack((np.array([225,225,225,255]),color_mapping))

        bokeh_palette=["#%02x%02x%02x" % (r, g, b) for r, g, b in color_mapping[:,0:3]]  
        #rnd_num=str(int(np.round(1000*np.random.random_sample())))  
        rnd_num=timestr()         
        output_file("Score_Scatter_"+rnd_num+".html",title='Score Scatter t['+str(xydim[0])+'] - t['+str(xydim[1])+ ']',mode='inline') 
        x_=T_matrix[:,[xydim[0]-1]]
        y_=T_matrix[:,[xydim[1]-1]]          
        
        TOOLS = "save,wheel_zoom,box_zoom,pan,reset,box_select,lasso_select"
        TOOLTIPS = [
                ("Obs #", "@ObsNum"),
                ("(x,y)", "($x, $y)"),
                ("Obs: ","@ObsID"),
                ("Class:","@Class")
                ]        
#        if not(isinstance(nbins, bool)):
#             if colorby in CLASSID.columns.to_list():
#                 classid_=list(classid_by_var[colorby])
#        else:
        classid_=list(CLASSID[colorby])
        legend_it = []
        
        p = figure(tools=TOOLS, tooltips=TOOLTIPS,toolbar_location="above",width=plotwidth,height=plotheight,title='Score Scatter t['+str(xydim[0])+'] - t['+str(xydim[1])+ '] '+addtitle)

        for classid_in_turn in Classes_:                      
            x_aux       = []
            y_aux       = []
            obsid_aux   = []
            obsnum_aux  = []
            classid_aux = []
            
            for i in list(range(len(ObsID_))):
                
                if classid_[i]==classid_in_turn:
                    x_aux.append(x_[i][0])
                    y_aux.append(y_[i][0])
                    obsid_aux.append(ObsID_[i])
                    obsnum_aux.append(ObsNum_[i])
                    classid_aux.append(classid_in_turn)
            source = ColumnDataSource(data=dict(x=x_aux, y=y_aux,ObsID=obsid_aux,ObsNum=obsnum_aux, Class=classid_aux))        
            color_=bokeh_palette[Classes_.index(classid_in_turn)]
            if add_legend:
                #c = p.circle('x','y',source=source,color=color_,size=marker_size)
                c = p.scatter('x','y',source=source,color=color_,size=marker_size)
                aux_=classid_in_turn
                if isinstance(aux_,(float,int)):
                    aux_=str(aux_)
                #legend_it.append((classid_in_turn, [c]))
                legend_it.append((aux_, [c]))
            else:
                #p.circle('x','y',source=source,color=color_,size=marker_size)
                p.scatter('x','y',source=source,color=color_,size=marker_size)
            if add_labels:
                labelsX = LabelSet(x='x', y='y', text='ObsID', level='glyph',x_offset=5, y_offset=5, source=source)
                p.add_layout(labelsX)
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
        if not(rscores):    
            p.xaxis.axis_label = 't ['+str(xydim[0])+']'
            p.yaxis.axis_label = 't ['+str(xydim[1])+']'
        else:
            p.xaxis.axis_label = 'r ['+str(xydim[0])+']'
            p.yaxis.axis_label = 'r ['+str(xydim[1])+']'
        # Vertical line
        vline = Span(location=0, dimension='height', line_color='black', line_width=2)
        # Horizontal line
        hline = Span(location=0, dimension='width', line_color='black', line_width=2)
        p.renderers.extend([vline, hline])
        if add_legend:
            #legend_cols=1
            ipc=[np.round(len(legend_it)/legend_cols)]*legend_cols              
            ipc[-1]=len(legend_it)-sum(ipc[:-1])
            pastit=0
            for it in ipc:
                leg_ = Legend(
                    items=legend_it[int(0+pastit):int(pastit+it)])
                    #location=(0,15+pastit*5))
                pastit+=it
                p.add_layout(leg_, 'right')
                leg_.click_policy="hide"
            #legend = Legend(items=legend_it, location='top_right')
            #p.add_layout(legend, 'right')
            
            
        show(p)
    return    

def score_line(mvmobj,dim,*,CLASSID=False,colorby=False,Xnew=False,add_ci=False,add_labels=False,add_legend=True,plotline=True,plotwidth=600,plotheight=600):
    '''Score line plot
    
    score_line(mvmobj,dim,*,CLASSID=False,colorby=False,Xnew=False,add_ci=False,add_labels=False,add_legend=True,plotline=True,plotwidth=600,plotheight=600):

    Args:
        mvmobj     : PLS or PCA object from phyphi
        dim        : LV to plot eg "1" will plot t1 vs observation #
        CLASSID    : Pandas DataFrame with CLASSIDS
        colorby    : Category (one of the CLASSIDS) to color by
        Xnew       : New data for which to make the score plot this routine evaluates and plots
        add_ci     : When = True will add confidence intervals
        add_labels : When =True will display Obs ID per point
        plotwidth  : When Omitted is = 600
        plotline   : Adds a conecting line between dots [True by default]
    
    by Salvador Garcia-Munoz 
    (sgarciam@ic.ac.uk ,salvadorgarciamunoz@gmail.com)
    '''
    if not(isinstance(dim,list)):
        if isinstance(dim, int):
            dim=[dim]
  
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

    ObsNum_=[]    
    for n in list(range(1,len(ObsID_)+1)):
        ObsNum_.append('Obs #'+str(n))  
                       
    if isinstance(CLASSID,bool): # No CLASSIDS
        #rnd_num=str(int(np.round(1000*np.random.random_sample())))
        rnd_num=timestr()
        output_file("Score_Line_"+rnd_num+".html",title='Score Line t['+str(dim[0])+ ']',mode='inline')

        y_=T_matrix[:,[dim[0]-1]]
        x_=list(range(1,y_.shape[0]+1))

           
        source = ColumnDataSource(data=dict(x=x_, y=y_,ObsID=ObsID_,ObsNum=ObsNum_))
        TOOLS = "save,wheel_zoom,box_zoom,pan,reset,box_select,lasso_select"
        TOOLTIPS = [
                ("Obs#", "@ObsNum"),
                ("(x,y)", "($x, $y)"),
                ("Obs: ","@ObsID")
                ]
        
        p = figure(tools=TOOLS, tooltips=TOOLTIPS,width=plotwidth,height=plotheight, title='Score Line t['+str(dim[0])+']' )
        #p.circle('x', 'y', source=source,size=7)
        p.scatter('x', 'y', source=source,size=10)
        
        if plotline:
            p.line('x', 'y', source=source)
        if add_ci:
            lim95,lim99=phi.single_score_conf_int(mvmobj['T'][:,[dim[0]-1]])
            p.line(x_, lim95,line_color="gold",line_dash='dashed')
            p.line(x_,-lim95,line_color="gold",line_dash='dashed')
            p.line(x_, lim99,line_color="red",line_dash='dashed')
            p.line(x_,-lim99,line_color="red",line_dash='dashed')
        if add_labels:
            labelsX = LabelSet(x='x', y='y', text='ObsID', level='glyph',x_offset=5, y_offset=5, source=source)
            p.add_layout(labelsX)
        p.xaxis.axis_label = 'Observation'
        p.yaxis.axis_label = 't ['+str(dim[0])+']'
        show(p)      
    else: # YES CLASSIDS
        
        Classes_=phi.unique(CLASSID,colorby)
        colormap = matplotlib.colormaps['rainbow']
        different_colors=len(Classes_)
        color_mapping=colormap(np.linspace(0,1,different_colors),1,True)
        bokeh_palette=["#%02x%02x%02x" % (r, g, b) for r, g, b in color_mapping[:,0:3]]  
        rnd_num=timestr()            
        output_file("Score_Line_"+rnd_num+".html",title='Score Line t['+str(dim[0])+ ']',mode='inline') 

        y_=T_matrix[:,[dim[0]-1]]  
        x_=list(range(1,y_.shape[0]+1))        
        
        TOOLS = "save,wheel_zoom,box_zoom,pan,reset,box_select,lasso_select"
        TOOLTIPS = [
                ("Obs#", "@ObsNum"),
                ("(x,y)", "($x, $y)"),
                ("Obs: ","@ObsID"),
                ("Class:","@Class")
                ]        
        classid_=list(CLASSID[colorby])
        legend_it = []
        
        p = figure(tools=TOOLS, tooltips=TOOLTIPS,toolbar_location="above",width=plotwidth,height=plotheight, title='Score Line t['+str(dim[0])+ ']')

        for classid_in_turn in Classes_:
            x_aux=[]
            y_aux=[]
            obsid_aux=[]
            classid_aux=[]
            obsnum_aux=[]
            for i in list(range(len(ObsID_))):
                if classid_[i]==classid_in_turn:
                    x_aux.append(x_[i])
                    y_aux.append(y_[i][0])
                    obsid_aux.append(ObsID_[i])
                    obsnum_aux.append(ObsNum_[i])
                    classid_aux.append(classid_in_turn)
            source = ColumnDataSource(data=dict(x=x_aux, y=y_aux,ObsID=obsid_aux,ObsNum=obsnum_aux,Class=classid_aux))        
            color_=bokeh_palette[Classes_.index(classid_in_turn)]
            #c=p.circle('x','y',source=source,color=color_)
            c=p.scatter('x','y',source=source,color=color_,size=10)
            
            if plotline:
                c1=p.line('x','y',source=source,color=color_)    
             #added to allow numbers in classids   
            aux_=classid_in_turn
            if isinstance(aux_,(float,int)):
                aux_=str(aux_)
             #        
            if add_legend and plotline:    
              #  legend_it.append((classid_in_turn, [c,c1]))
                legend_it.append((aux_, [c,c1]))
            if add_legend and not(plotline):
              #  legend_it.append((classid_in_turn, [c]))
                legend_it.append((aux_, [c]))

            if add_labels:
                labelsX = LabelSet(x='x', y='y', text='ObsID', level='glyph',x_offset=5, y_offset=5, source=source)
                p.add_layout(labelsX)
                
        if add_ci:
            lim95,lim99=phi.single_score_conf_int(mvmobj['T'][:,[dim[0]-1]])
            p.line(x_, lim95,line_color="gold",line_dash='dashed')
            p.line(x_,-lim95,line_color="gold",line_dash='dashed')
            p.line(x_, lim99,line_color="red",line_dash='dashed')
            p.line(x_,-lim99,line_color="red",line_dash='dashed')   
        p.xaxis.axis_label = 'Observation'
        p.yaxis.axis_label = 't ['+str(dim[0])+']'
        if add_legend:
            legend = Legend(items=legend_it, location='top_right')
            p.add_layout(legend, 'right')
            legend.click_policy="hide"
        show(p)
    return  



def diagnostics(mvmobj,*,Xnew=False,Ynew=False,score_plot_xydim=False,plotwidth=600,ht2_logscale=False,spe_logscale=False):
    """Hotelling's T2 and SPE
    
    diagnostics(mvmobj,*,Xnew=False,Ynew=False,score_plot_xydim=False,plotwidth=600,ht2_logscale=False,spe_logscale=False):
 
    Args:
        mvmobj: A model created with phi.pca or phi.pls
    
    Optional Parameters:
        Xnew/Ynew:     Data used to calculate diagnostics[numpy arrays or pandas dataframes] 

        score_plot_xydim: will add a score scatter plot at the bottom 
                          if sent with a list of [dimx, dimy] where dimx/dimy 
                          are integers and refer to the latent space to plot
                          in the x and y axes of the scatter plot. e.g. [1,2] will
                          add a t1-t2 plot 
                      
    by Salvador Garcia-Munoz 
    (sgarciam@ic.ac.uk ,salvadorgarciamunoz@gmail.com)
    """
    
    if isinstance(score_plot_xydim,bool):
        add_score_plot = False
    else:
        add_score_plot = True
        
    if isinstance(Xnew,bool): #No Xnew was given need to plot all from model
        if 'obsidX' in mvmobj:
            ObsID_=mvmobj['obsidX']
        else:
            ObsID_ = []
            for n in list(np.arange(mvmobj['T'].shape[0])+1):
                ObsID_.append('Obs #'+str(n))  
                              
        Obs_num = np.arange(mvmobj['T'].shape[0])+1
        
        if add_score_plot and not(isinstance(score_plot_xydim,bool)):
            t_x  = mvmobj['T'][:,[score_plot_xydim[0]-1]]
            t_y  = mvmobj['T'][:,[score_plot_xydim[1]-1]]
        else:
            add_score_plot = False
        t2_   = mvmobj['T2']
        spex_ = mvmobj['speX']
        
        if ht2_logscale:
            t2_=np.log10(t2_)
        if spe_logscale:
            spex_= np.log10(spex_)
            
            
        if not(add_score_plot):
            if 'Q' in mvmobj:
                spey_=1
                source = ColumnDataSource(data=dict(x=Obs_num, ObsID=ObsID_,t2=t2_,spex=spex_,spey=mvmobj['speY']))  
            else:
                source = ColumnDataSource(data=dict(x=Obs_num, ObsID=ObsID_,t2=t2_,spex=spex_)) 
        else:
            if 'Q' in mvmobj:
                spey_=1
                source = ColumnDataSource(data=dict(x=Obs_num, ObsID=ObsID_,t2=t2_,spex=spex_,spey=mvmobj['speY'],tx=t_x,ty=t_y))  
            else:
                source = ColumnDataSource(data=dict(x=Obs_num, ObsID=ObsID_,t2=t2_,spex=spex_,tx=t_x,ty=t_y))
    else: #Xnew was given
        if isinstance(Xnew,np.ndarray):
            ObsID_ = []
            for n in list(np.arange(Xnew.shape[0])+1):
                ObsID_.append('Obs #'+str(n))  
        elif isinstance(Xnew,pd.DataFrame):
            X_=np.array(Xnew.values[:,1:]).astype(float)
            ObsID_ = Xnew.values[:,0].astype(str)
            ObsID_ = ObsID_.tolist()
            
        
        
        if add_score_plot and not(isinstance(score_plot_xydim,bool)):
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
        
        Obs_num = np.arange(t2_.shape[0])+1
        
        if 'Q' in mvmobj and not(isinstance(Ynew,bool)):
            spex_,spey_ = phi.spe(mvmobj,Xnew,Ynew=Ynew)
        else:
            spex_ = phi.spe(mvmobj,Xnew)
            spey_ = False
            
        if ht2_logscale:
            t2_=np.log10(t2_)
        if spe_logscale:
            spex_= np.log10(spex_)
        ObsNum_=[]    
        for n in list(range(1,len(ObsID_)+1)):
            ObsNum_.append('Obs #'+str(n))  
                       
                       
        if not(add_score_plot):
            if 'Q' in mvmobj and not(isinstance(Ynew,bool)):
                source = ColumnDataSource(data=dict(x=Obs_num, ObsID=ObsID_,ObsNum=ObsNum_,t2=t2_,spex=spex_,spey=spey_))  
            else:
                source = ColumnDataSource(data=dict(x=Obs_num, ObsID=ObsID_,ObsNum=ObsNum_,t2=t2_,spex=spex_)) 
        else:
            if 'Q' in mvmobj and not(isinstance(Ynew,bool)):
                source = ColumnDataSource(data=dict(x=Obs_num, ObsID=ObsID_,ObsNum=ObsNum_,t2=t2_,spex=spex_,spey=spey_,tx=t_x,ty=t_y))  
            else:
                source = ColumnDataSource(data=dict(x=Obs_num, ObsID=ObsID_,ObsNum=ObsNum_,t2=t2_,spex=spex_,tx=t_x,ty=t_y))
    TOOLS = "save,wheel_zoom,box_zoom,reset,lasso_select"
    TOOLTIPS = [
            ("Obs #", "@x"),
            ("(x,y)", "($x, $y)"),
            ("Obs: ","@ObsID")
            ]
    
    #rnd_num=str(int(np.round(1000*np.random.random_sample())))      
    rnd_num=timestr()         
    output_file("Diagnostics"+rnd_num+".html",title='Diagnostics',mode='inline') 
    p = figure(tools=TOOLS, tooltips=TOOLTIPS, width=plotwidth, title="Hotelling's T2")
    #p.circle('x','t2',source=source)
    p.scatter('x','t2',source=source)
    if ht2_logscale:
        p.line([0,Obs_num[-1]],[np.log10(mvmobj['T2_lim95']),np.log10(mvmobj['T2_lim95'])],line_color='gold')
        p.line([0,Obs_num[-1]],[np.log10(mvmobj['T2_lim99']),np.log10(mvmobj['T2_lim99'])],line_color='red')
    else:        
        p.line([0,Obs_num[-1]],[mvmobj['T2_lim95'],mvmobj['T2_lim95']],line_color='gold')
        p.line([0,Obs_num[-1]],[mvmobj['T2_lim99'],mvmobj['T2_lim99']],line_color='red')
    
    
    p.xaxis.axis_label = 'Observation sequence'
    p.yaxis.axis_label = "HT2"
    p_list=[p]
    
    p = figure(tools=TOOLS, tooltips=TOOLTIPS, width=plotwidth, title='SPE X')
    #p.circle('x','spex',source=source)
    p.scatter('x','spex',source=source)
    
    
    if spe_logscale:
        p.line([0,Obs_num[-1]],[np.log10(mvmobj['speX_lim95']),np.log10(mvmobj['speX_lim95'])],line_color='gold')
        p.line([0,Obs_num[-1]],[np.log10(mvmobj['speX_lim99']),np.log10(mvmobj['speX_lim99'])],line_color='red')
    else:  
        p.line([0,Obs_num[-1]],[mvmobj['speX_lim95'],mvmobj['speX_lim95']],line_color='gold')
        p.line([0,Obs_num[-1]],[mvmobj['speX_lim99'],mvmobj['speX_lim99']],line_color='red')
    p.xaxis.axis_label = 'Observation sequence'
    p.yaxis.axis_label = 'SPE X-Space'
    p_list.append(p)
    
    p = figure(tools=TOOLS, tooltips=TOOLTIPS, width=plotwidth, title='Outlier Map')
    #p.circle('t2','spex',source=source)
    p.scatter('t2','spex',source=source)
    
    if ht2_logscale:
        vline = Span(location=np.log10(mvmobj['T2_lim99']), dimension='height', line_color='red', line_width=1)
    else:
        vline = Span(location=mvmobj['T2_lim99'], dimension='height', line_color='red', line_width=1)
    if spe_logscale:    
        hline = Span(location=np.log10(mvmobj['speX_lim99']), dimension='width', line_color='red', line_width=1)
    else:
        hline = Span(location=mvmobj['speX_lim99'], dimension='width', line_color='red', line_width=1)
    p.renderers.extend([vline, hline])
    
    p.xaxis.axis_label = "Hotelling's T2"
    p.yaxis.axis_label = 'SPE X-Space'
    p_list.append(p)
    
    
    if 'Q' in mvmobj and not(isinstance(spey_,bool)):
        p = figure(tools=TOOLS, tooltips=TOOLTIPS, height=400, title='SPE Y')
        #p.circle('x','spey',source=source)
        p.scatter('x','spey',source=source,size=10)
        p.line([0,Obs_num[-1]],[mvmobj['speY_lim95'],mvmobj['speY_lim95']],line_color='gold')
        p.line([0,Obs_num[-1]],[mvmobj['speY_lim99'],mvmobj['speY_lim99']],line_color='red')
        p.xaxis.axis_label = 'Observation sequence'
        p.yaxis.axis_label = 'SPE Y-Space'
        p_list.append(p)
    if add_score_plot:
        p = figure(tools=TOOLS, tooltips=TOOLTIPS, width=plotwidth, title='Score Scatter')
        #p.circle('tx', 'ty', source=source,size=7)
        p.scatter('tx', 'ty', source=source,size=10)
        
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
    """    Plot observed vs predicted values
    
    predvsobs(mvmobj,X,Y,*,CLASSID=False,colorby=False,x_space=False)
    
    Args:
        mvmobj: A model created with phi.pca or phi.pls 
        X/Y:     Data [numpy arrays or pandas dataframes] 
        
    Optional Parameters:
        CLASSID: Pandas Data Frame with classifiers per observation, each column is a class
        
        colorby: one of the classes in CLASSID to color by
        
        x_space: = 'False' will skip plotting the obs. vs pred for X *default*
                   'True' will also plot obs vs pred for X
                   
   by Salvador Garcia-Munoz 
   (sgarciam@ic.ac.uk ,salvadorgarciamunoz@gmail.com)

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
    elif isinstance(X,dict):
        X_=X.copy()
        k=list(X.keys())
        ObsID_=X[k[0]].values[:,0].astype(str)
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
    
    if isinstance(CLASSID,bool): # No CLASSIDS
        #rnd_num=str(int(np.round(1000*np.random.random_sample())))
        rnd_num=timestr()
        output_file("ObsvsPred_"+rnd_num+".html",title='ObsvsPred',mode='inline')
        plot_counter=0
        
        if not(isinstance(yhat,bool)): #skip if PCA model sent
            for i in list(range(Y_.shape[1])):
                x_ = Y_[:,i]
                y_ = yhat[:,i]          
                min_value = np.nanmin([np.nanmin(x_),np.nanmin(y_)])
                max_value = np.nanmax([np.nanmax(x_),np.nanmax(y_)])
                
                source = ColumnDataSource(data=dict(x=x_, y=y_,ObsID=ObsID_))
                #p = figure(tools=TOOLS, tooltips=TOOLTIPS,width=600, height=600, title=YVar[i])
                p = figure(tools=TOOLS, tooltips=TOOLTIPS,width=600, height=600, title=YVar[i],x_range=(min_value, max_value),y_range=(min_value, max_value))
                #p.circle('x', 'y', source=source,size=7,color='darkblue')
                p.scatter('x', 'y', source=source,size=7,color='darkblue')
                p.line([min_value,max_value],[min_value,max_value],line_color='cyan',line_dash='dashed')
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
                min_value = np.nanmin([np.nanmin(x_),np.nanmin(y_)])
                max_value = np.nanmax([np.nanmax(x_),np.nanmax(y_)])
 
                source = ColumnDataSource(data=dict(x=x_, y=y_,ObsID=ObsID_))
                p = figure(tools=TOOLS, tooltips=TOOLTIPS,width=600, height=600, title=XVar[i],x_range=(min_value, max_value),y_range=(min_value, max_value))
                #p.circle('x', 'y', source=source,size=7,color='darkblue')
                p.scatter('x', 'y', source=source,size=10,color='darkblue')
                p.line([min_value,max_value],[min_value,max_value],line_color='cyan',line_dash='dashed')
                p.xaxis.axis_label ='Observed'
                p.yaxis.axis_label ='Predicted'
                if plot_counter==0:
                    p_list=[p]
                else:
                    p_list.append(p)
                plot_counter = plot_counter+1
        show(column(p_list))
        
    else: # YES CLASSIDS
        #Classes_=np.unique(CLASSID[colorby]).tolist()
        Classes_=phi.unique(CLASSID,colorby)
        different_colors=len(Classes_)
        #colormap =cm.get_cmap("rainbow")
        colormap = matplotlib.colormaps['rainbow']
        color_mapping=colormap(np.linspace(0,1,different_colors),1,True)
        bokeh_palette=["#%02x%02x%02x" % (r, g, b) for r, g, b in color_mapping[:,0:3]]  
        #rnd_num=str(int(np.round(1000*np.random.random_sample())))
        rnd_num=timestr()
        output_file("ObsvsPred_"+rnd_num+".html",title='ObsvsPred',mode='inline')      
        classid_=list(CLASSID[colorby])
        
        plot_counter=0
        
        if not(isinstance(yhat,bool)): #skip if PCA model sent
            for i in list(range(Y_.shape[1])):
                x_ = Y_[:,i]
                y_ = yhat[:,i]
                min_value = np.nanmin([np.nanmin(x_),np.nanmin(y_)])
                max_value = np.nanmax([np.nanmax(x_),np.nanmax(y_)])
                p = figure(tools=TOOLS, tooltips=TOOLTIPS,width=600, height=600, title=YVar[i],x_range=(min_value, max_value),y_range=(min_value, max_value))
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
                    #p.circle('x','y',source=source,color=color_,legend_label=classid_in_turn)
                    p.scatter('x','y',source=source,color=color_,legend_label=classid_in_turn)
                    p.line([min_value,max_value],[min_value,max_value],line_color='cyan',line_dash='dashed')
                p.xaxis.axis_label ='Observed'
                p.yaxis.axis_label ='Predicted'
                p.legend.click_policy="hide"
                p.legend.location = "top_left"
                if plot_counter==0:
                    p_list=[p]
                    plot_counter = plot_counter+1
                else:
                    p_list.append(p)
              
        if x_space: #
            for i in list(range(X_.shape[1])):
                x_ = X_[:,i]
                y_ = xhat[:,i]
                min_value = np.nanmin([np.nanmin(x_),np.nanmin(y_)])
                max_value = np.nanmax([np.nanmax(x_),np.nanmax(y_)])
                p = figure(tools=TOOLS, tooltips=TOOLTIPS,width=600, height=600, title=XVar[i],x_range=(min_value, max_value),y_range=(min_value, max_value))
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
                    #p.circle('x','y',source=source,color=color_,legend_label=classid_in_turn)
                    p.scatter('x','y',source=source,color=color_,legend_label=classid_in_turn)
                    p.line([min_value,max_value],[min_value,max_value],line_color='cyan',line_dash='dashed')
                p.xaxis.axis_label ='Observed'
                p.yaxis.axis_label ='Predicted'
                p.legend.click_policy="hide"
                p.legend.location = "top_left"
                if plot_counter==0:
                    p_list=[p]
                    plot_counter = plot_counter+1
                else:
                    p_list.append(p)
        show(column(p_list))
    return    

def contributions_plot(mvmobj,X,cont_type,*,Y=False,from_obs=False,to_obs=False,lv_space=False,plotwidth=800,plotheight=600,xgrid=False):
    """Plot contributions to diagnostics
    
contributions_plot(mvmobj,X,cont_type,*,Y=False,from_obs=False,to_obs=False,lv_space=False,plotwidth=800,plotheight=600,xgrid=False):

    Args:
        mvmobj : A dictionary created by phi.pls or phi.pca    
        X/Y:     Data [numpy arrays or pandas dataframes] - Y space is optional
        cont_type: 'ht2'
                   'spe'
                   'scores'
        to_obs: Scalar or list of scalars with observation(s) number(s)| first element is #0
                  - OR -
                Strings or list of strings with observation(s) name(s) [if X/Y are pandas data frames]
     
    Optional Parameters:
        
    from_obs: Scalar or list of scalars with observation(s) number(s) | first element is #0
              - OR -
              Strings or list of strings with observation(s) name(s) [if X/Y are pandas data frames]
              Used to off set calculations for scores or ht2
              "False' will calculate with respect to origin *default if not sent*
              *Note: from_obs is ignored when cont_type='spe'*
            
    lv_space: Latent spaces over which to do the calculations [applicable to 'ht2' and 'scores']
    
    by Salvador Garcia-Munoz 
    (sgarciam@ic.ac.uk ,salvadorgarciamunoz@gmail.com)
    """
    good_to_go=True
    
    #Convert dictionaries into a data frame for MB models
    if isinstance(X,dict):
        firstone=True
        for k in X.keys():
            if firstone:
                X_=[X[k].copy()]
                firstone=False
            else:
                X_.append(X[k].iloc[:,1:])
        X=pd.concat(X_,axis=1)
        
    if isinstance(Y,dict):
        firstone=True
        for k in Y.keys():
            if firstone:
                Y_=[Y[k].copy()]
                firstone=False
            else:
                Y_.append(Y[k].iloc[:,1:])
        Y=pd.concat(Y_,axis=1)
        
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
        elif isinstance(to_obs,bool):
            good_to_go=False
        if not(isinstance(from_obs,bool)):
            if isinstance(from_obs,str):
                from_obs_=ObsID.index(from_obs)
            elif isinstance(from_obs,int):
                from_obs_=from_obs
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
    if cont_type=='scores' and not(isinstance(Y,bool)):
        Y=False
        
    if isinstance(Y,bool) and good_to_go:
        Xconts=phi.contributions(mvmobj,X,cont_type,Y=False,from_obs=from_obs_,to_obs=to_obs_,lv_space=lv_space)
        Yconts=False
    elif not(isinstance(Y,bool)) and good_to_go and ('Q' in mvmobj) and cont_type=='spe':
        
        Xconts,Yconts=phi.contributions(mvmobj,X,cont_type,Y=Y,from_obs=from_obs_,to_obs=to_obs_,lv_space=lv_space)
    
    if 'varidX' in mvmobj:
        XVar=mvmobj['varidX']
    else:
        XVar = []
        for n in list(np.arange(mvmobj['P'].shape[0])+1):
            XVar.append('XVar #'+str(n))               

    #rnd_num=str(int(np.round(1000*np.random.random_sample())))
    rnd_num=timestr()
    output_file("Contributions"+rnd_num+".html",title='Contributions',mode='inline')
    if isinstance(from_obs,list):
        from_txt=", ".join(map(str, from_obs))
        from_txt=" from obs: "+from_txt
    elif isinstance(from_obs,int):
        from_txt=" from obs: "+str(from_obs)
    elif isinstance(from_obs,str):    
        from_txt=" from obs: " + from_obs
    else:
        from_txt=""
    if isinstance(to_obs,list):
        to_txt=", ".join(map(str, to_obs))
        to_txt=", to obs: "+to_txt
    elif isinstance(to_obs,str):    
        to_txt=", to obs: " + to_obs
    elif isinstance(to_obs,int):
        to_txt =", to obs: "+ str(to_obs)
    else:
        to_txt=""
        
    TOOLTIPS = [
                ("Variable","@names")
                ]
    p = figure(x_range=XVar, height=plotheight,width=plotwidth, title="Contributions Plot"+from_txt+to_txt,
                    tools="save,box_zoom,pan,reset",tooltips=TOOLTIPS)
    
    source1 = ColumnDataSource(data=dict(x_=XVar, y_=Xconts[0].tolist(),names=XVar)) 
    #p.vbar(x=XVar, top=Xconts[0].tolist(), width=0.5)
    p.vbar(x='x_', top='y_', source=source1,width=0.5)
    
    
    p.ygrid.grid_line_color = None    
    if xgrid:
        p.xgrid.grid_line_color = 'lightgray'
    else:
        p.xgrid.grid_line_color = None   
    p.yaxis.axis_label = 'Contributions to '+cont_type
    hline = Span(location=0, dimension='width', line_color='black', line_width=2)
    p.renderers.extend([hline])
    p.xaxis.major_label_orientation = math.pi/2
    p_list=[p]
    
    if not(isinstance(Yconts,bool)):
        if 'varidY' in mvmobj:
            YVar=mvmobj['varidY']
        else:
            YVar = []
            for n in list(np.arange(mvmobj['Q'].shape[0])+1):
                YVar.append('YVar #'+str(n))               
        
        p = figure(x_range=YVar, height=plotheight,width=plotwidth, title="Contributions Plot",
                    tools="save,box_zoom,pan,reset")
        #p.vbar(x=YVar, top=Yconts[0].tolist(), width=0.5)
        source1 = ColumnDataSource(data=dict(x_=YVar, y_=Yconts[0].tolist(),names=YVar)) 
        p.vbar(x='x_', top='y_', source=source1,width=0.5)
        
        p.ygrid.grid_line_color = None    
        if xgrid:
            p.xgrid.grid_line_color = 'lightgray'
        else:
            p.xgrid.grid_line_color = None   
        p.yaxis.axis_label = 'Contributions to '+cont_type
        hline = Span(location=0, dimension='width', line_color='black', line_width=2)
        p.renderers.extend([hline])
        p.xaxis.major_label_orientation = math.pi/2
        p_list.append(p)
        
    show(column(p_list))  
    return


def mb_weights(mvmobj,*,plotwidth=600,plotheight=400):
    """Super weights for Multi-block models
    
    mb_weights(mvmobj,*,plotwidth=600,plotheight=400)
    
    Args:
        mvmobj: A multi-block PLS model created with phi.mbpls
        
    by Salvador Garcia-Munoz 
    (sgarciam@ic.ac.uk ,salvadorgarciamunoz@gmail.com)
    """
    A= mvmobj['T'].shape[1]
    lv_prefix='LV #'        
    lv_labels = []   
    for a in list(np.arange(A)+1):
        lv_labels.append(lv_prefix+str(a))    
    XVar=mvmobj['Xblocknames']        
    for i in list(np.arange(A)):
        #rnd_num=str(int(np.round(1000*np.random.random_sample())))
        rnd_num=timestr()
        output_file("blockweights_"+rnd_num+".html",title="Block Weights",mode='inline')         
        px = figure(x_range=XVar, title="Block weights for MBPLS"+lv_labels[i],
             tools="save,box_zoom,hover,reset", tooltips=[("Var:","@x_")],width=plotwidth,height=plotheight)   
        source1 = ColumnDataSource(data=dict(x_=XVar, y_=mvmobj['Wt'][:,i].tolist(),names=XVar)) 
        px.vbar(x='x_', top='y_', source=source1,width=0.5)
        px.y_range.range_padding = 0.1
        px.ygrid.grid_line_color = None
        px.axis.minor_tick_line_color = None
        px.outline_line_color = None
        px.yaxis.axis_label = 'Wt'+str(i+1)+']'
        px.xaxis.major_label_orientation = math.pi/2 
        hline = Span(location=0, dimension='width', line_color='black', line_width=2)
        px.renderers.extend([hline])
        if i==0:
            p_list=[px]
        else:
            p_list.append(px)
    show(column(p_list))  

    return 

def mb_r2pb(mvmobj,*,plotwidth=600,plotheight=400):
    """R2 for each block for Multi-block models
    
    mb_r2pb(mvmobj,*,plotwidth=600,plotheight=400)
    
    Args:
        mvmobj: A multi-block PLS model created with phi.mbpls
        
    by Salvador Garcia-Munoz 
    (sgarciam@ic.ac.uk ,salvadorgarciamunoz@gmail.com)
    """
    A= mvmobj['T'].shape[1]
    lv_prefix='LV #'        
    lv_labels = []   
    for a in list(np.arange(A)+1):
        lv_labels.append(lv_prefix+str(a))    
    r2pbX_dict = {'XVar': mvmobj['Xblocknames']}
    XVar=mvmobj['Xblocknames']        
    for i in list(np.arange(A)):
        r2pbX_dict.update({lv_labels[i] : mvmobj['r2pbX'][:,i].tolist()})
        #rnd_num=str(int(np.round(1000*np.random.random_sample())))
        rnd_num=timestr()
        output_file("r2perblock"+rnd_num+".html",title="R2 per Block",mode='inline') 
        #colormap =cm.get_cmap("rainbow")
        colormap = matplotlib.colormaps['rainbow']
        different_colors=A
        color_mapping=colormap(np.linspace(0,1,different_colors),1,True)
        bokeh_palette=["#%02x%02x%02x" % (r, g, b) for r, g, b in color_mapping[:,0:3]]                 
        px = figure(x_range=XVar, title="r2 per Block for MBPLS",
             tools="save,box_zoom,hover,reset", tooltips="$name @XVar: @$name",width=plotwidth,height=plotheight)        
        px.vbar_stack(lv_labels, x='XVar', width=0.9,color=bokeh_palette,source=r2pbX_dict)
        px.y_range.range_padding = 0.1
        px.ygrid.grid_line_color = None
        px.axis.minor_tick_line_color = None
        px.outline_line_color = None
        px.yaxis.axis_label = 'R2 per Block per LV'
        px.xaxis.major_label_orientation = math.pi/2  
    show(px)
    return


def mb_vip(mvmobj,*,plotwidth=600,plotheight=400):
    """VIP per block for Multi-block models
    
    mb_vip(mvmobj,*,plotwidth=600,plotheight=400)
    
    Args:
        mvmobj: A multi-block PLS model created with phi.mbpls
        
    by Salvador Garcia-Munoz 
    (sgarciam@ic.ac.uk ,salvadorgarciamunoz@gmail.com)
    """
    A= mvmobj['T'].shape[1]
   
    XVar=mvmobj['Xblocknames']        
    Wt=mvmobj['Wt']
    r2y=mvmobj['r2y']
    vip=np.zeros((Wt.shape[0],1))
    if A>1:
        for a in list(range(A)):
            vip=vip+Wt[:,[a]]*r2y[a]
    else:
        vip=Wt[:,[0]]*r2y
        
    vip=np.reshape(vip,-1)
    index=np.argsort(vip)
    index=index[::-1]
    XVar_=[XVar[i] for i in index]
    XVar = XVar_
    vip=vip[index]
    #rnd_num=str(int(np.round(1000*np.random.random_sample())))
    rnd_num=timestr()
    output_file("blockvip"+rnd_num+".html",title="Block VIP",mode='inline') 
    source1 = ColumnDataSource(data=dict(x_=XVar, y_=vip.tolist(),names=XVar))         
    px = figure(x_range=XVar, title="Block VIP for MBPLS",
         tools="save,box_zoom,hover,reset",tooltips=[("Block:","@x_")],width=plotwidth,height=plotheight)   
    
    px.vbar(x='x_', top='y_', source=source1,width=0.5)
    px.y_range.range_padding = 0.1
    px.ygrid.grid_line_color = None
    px.axis.minor_tick_line_color = None
    px.outline_line_color = None
    px.yaxis.axis_label = 'Block VIP'
    px.xaxis.major_label_orientation = math.pi/2 
    hline = Span(location=0, dimension='width', line_color='black', line_width=2)
    px.renderers.extend([hline])
    show(px)  
    return

def barplot(yheights,xtick_labels,*,plotwidth=600,plotheight=600,
            addtitle='',xlabel='',ylabel='',tabtitle='Bar Plot'):
    """ Generic Bar plot with Bokeh

    barplot(yheights,xtick_labels,*,plotwidth=600,plotheight=600,
                addtitle='',xlabel='',ylabel='',tabtitle='Bar Plot'):

    Args:
        yheights: Values of bars
        xtick_labels: [list] Variable identifiers for x axis 
        
    by Salvador Garcia-Munoz 
    (sgarciam@ic.ac.uk ,salvadorgarciamunoz@gmail.com)
    
    """
    rnd_num=timestr()
    output_file("BarPlot_"+rnd_num+".html",title=tabtitle,mode='inline') 
              
    TOOLTIPS = [
                ("Variable","@names")
                ]
    
    p = figure(x_range=xtick_labels, title=addtitle,
        tools="save,box_zoom,pan,reset",tooltips=TOOLTIPS,width=plotwidth)
    
    source1 = ColumnDataSource(data=dict(x_=xtick_labels, y_=yheights,names=xtick_labels)) 
    
    p.vbar(x='x_', top='y_', source=source1,width=0.5)
    p.xgrid.grid_line_color = None
    p.yaxis.axis_label = ylabel
    p.xaxis.axis_label = xlabel
    p.xaxis.major_label_orientation = math.pi/2
    show(p)
    return 

def lineplot(X,*,ids_2_include=False,x_axis=False,plot_title='Main Title',tab_title='Tab Title',
             xaxis_label='X- axis',yaxis_label='',plotheight=400,plotwidth=600,legend_cols=1,
             linecolor='blue',linewidth=2,add_marker=False,individual_plots=False,add_legend=True,
             markercolor='darkblue',markersize=10,
             fill_alpha =0.2,line_alpha=0.4,
             ncx_x_col=False,ncx_y_col=False,ncx_id_col=False,
             CLASSID=False,colorby=False,yaxis_log=False): 
    """  Simple way to plot a column(s) of a Pandas DataFrame with Bokeh.
    
    lineplot(X,*,ids_2_include=False,x_axis=False,plot_title='Main Title',tab_title='Tab Title',
                 xaxis_label='X- axis',yaxis_label='',plotheight=400,plotwidth=600,legend_cols=1,
                 linecolor='blue',linewidth=2,add_marker=False,individual_plots=False,add_legend=True,
                 markercolor='darkblue',markersize=10,
                 fill_alpha =0.2,line_alpha=0.4,
                 ncx_x_col=False,ncx_y_col=False,ncx_id_col=False
                 CLASSID=False,colorby=False,yaxis_log=False):  
    
    Args:
        
        X:          if DataFrame:  Data to be plotted against a common x-axis
                    if list of DataFrames: Data to be ploted non-common x-axis (see 
                     ncx_x_col,ncx_y_col and ncx_id_col arguments)
         
   Optional Parameters:    
             
        ids_2_include:   [list] of ID's to be included in the plot.If X is a DataFrame 
                         these are the column names of data to be plotted. If X is
                         a list of df's then this is the list of IDs (column of df with ids is
                         identified via the ncx_id_col argument)

   
        x_axis:     List or numpy vector with the numeric common x-axis values to be used
                    (Only used when X is a dataframe since x-axis is common)
                    
        plot_title  Title of the plot
        tab_title   Title of the html tab
        xaxis_label self explanatory 
        yaxis_label self explanatory 
        plotheight  self explanatory 
        plotwidth   self explanatory 
        linecolor   self explanatory [ignored when plotting multiple lines]
        linewidth   self explanatory 
        add_marker  self explanatory [ True or False (default)]
        individual_plots self explanatory [True or False(default)]
        add_legend  self explanatory True (default) or False]
        markercolor self explanatory 
        markersize  self explanatory 
        fill_alpha  transparency of the marker
        line_alpha  transparency of the line
        
        Use when plotting each line against an individual x-axis and X is 
        a list of DataFrames with the data to be plotted
    
        ncx_x_col   [string] indicates the column in the df that has the x values
        ncx_y_col   [string] indicates the column in the df that has the y values
        ncx_id_col  [string] indicates the column in the df that has the identifier
        
        CLASSID     [Dataframe]  categorical dataframe to colr code
        colorby     column of the categorical dataframe to color code by
        
        yaxis_log   [boolean] default is False if True, Y axis is scale with log
    
    Programmed by Salvador Garcia-Munoz
    (sgarciam@ic.ac.uk ,salvadorgarciamunoz@gmail.com)
    
    
    """
    if isinstance(X,pd.DataFrame): #Common x-axis
    
        if isinstance(ids_2_include,str):
            ids_2_include=[ids_2_include]
        elif isinstance(ids_2_include,bool):
            ids_2_include=X.columns.tolist()
            
        first_plot=True
        TOOLS = "save,wheel_zoom,box_zoom,pan,reset,box_select,lasso_select"
        TOOLTIPS = [
                    ("Obs #", "@ObsNum"),
                    ("(x,y)", "($x, $y)"),
                    ("ID: ","@ColID")
                    ] 
        rnd_num=timestr()          
        output_file("LinePlot"+rnd_num+".html",title=tab_title,mode='inline')
        if individual_plots:
            for i,this_col_name in enumerate(ids_2_include):
                aux=X.loc[:,this_col_name]
                y_=aux.values  
                if isinstance(x_axis,bool):
                    x_=list(range(1,X.shape[0]+1))
                else:
                    if isinstance(x_axis,list):
                        x_=x_axis
                    elif isinstance(x_axis,np.ndarray):
                        x_=x_axis.tolist()
                    else:
                        print('x_axis must be a list or a numpy vector')
                        x_=list(range(1,X.shape[0]+1))
                ObsNum_=x_    
                if not(first_plot):                   
                    plot_title=''    
                if yaxis_log:
                    p = figure(tools=TOOLS, tooltips=TOOLTIPS,width=plotwidth,height=plotheight,title=plot_title,y_axis_type="log")  
                else:    
                    p = figure(tools=TOOLS, tooltips=TOOLTIPS,width=plotwidth,height=plotheight,title=plot_title)  
                
                source = ColumnDataSource(data=dict(x=x_, y=y_,ColID=[this_col_name]*X.shape[0],ObsNum=ObsNum_))
                p.xaxis.axis_label = xaxis_label
                p.yaxis.axis_label = yaxis_label
                if add_legend:
                    p.line('x', 'y', source=source,line_color=linecolor,color=markercolor,line_width=linewidth,line_alpha=line_alpha,legend_label= this_col_name)
                else:
                    p.line('x', 'y', source=source,line_color=linecolor,color=markercolor,line_width=linewidth,line_alpha=line_alpha)
                #p.circle('x', 'y', source=source)
                hline = Span(location=0, dimension='width', line_color='black', line_width=2)
                p.renderers.extend([hline])
                if add_marker:
                    p.scatter('x', 'y', source=source, color=markercolor,size=markersize,fill_alpha =fill_alpha)
                if first_plot:
                    p_list=[p]
                    first_plot=False
                else:
                    p_list.append(p)
            show(column(p_list))  
        else: # All in one plot
            if isinstance(CLASSID,bool): #No CLASSIDs
              
                if yaxis_log:
                    p = figure(tools=TOOLS, tooltips=TOOLTIPS,width=plotwidth,height=plotheight,title=plot_title,y_axis_type="log")  
                else:    
                    p = figure(tools=TOOLS, tooltips=TOOLTIPS,width=plotwidth,height=plotheight,title=plot_title)  
                   
                
                colormap = matplotlib.colormaps['rainbow']
                different_colors=len(ids_2_include)
                color_mapping=colormap(np.linspace(0,1,different_colors),1,True)
                bokeh_palette=["#%02x%02x%02x" % (r, g, b) for r, g, b in color_mapping[:,0:3]]      
                legend_it=[]
                for this_col_name in ids_2_include:
                    color_=bokeh_palette[ids_2_include.index(this_col_name)]
                    y_=X[this_col_name].values  
                    if isinstance(x_axis,bool):
                        x_=list(range(1,X.shape[0]+1))
                    else:
                        if isinstance(x_axis,list):
                            x_=x_axis
                        elif isinstance(x_axis,np.ndarray):
                            x_=x_axis.tolist()
                        else:
                            print('x_axis must be a list or a numpy vector')
                            x_=list(range(1,X.shape[0]+1))
                    ObsNum_=x_
                    source = ColumnDataSource(data=dict(x=x_, y=y_,ColID=[this_col_name]*X.shape[0],ObsNum=ObsNum_))
                    glyph=p.line('x', 'y', source=source,line_color=color_,line_width=linewidth,#legend_label= this_col_name,
                           color=color_,line_alpha=line_alpha)
                    
                    hline = Span(location=0, dimension='width', line_color='black', line_width=2)
                    p.renderers.extend([hline])
                    if add_marker:
                        glyphm=p.scatter('x', 'y', source=source,color=color_,size=markersize,fill_alpha =fill_alpha)
                        legend_it.append((this_col_name, [glyph,glyphm]))
                    else:
                        legend_it.append((this_col_name, [glyph]))
                p.xaxis.axis_label = xaxis_label
                p.yaxis.axis_label = yaxis_label
                    
                if add_legend:
                    #legend_cols=3
                    ipc=[np.round(len(legend_it)/legend_cols)]*legend_cols              
                    ipc[-1]=len(legend_it)-sum(ipc[:-1])
                    pastit=0
                    for it in ipc:
                        leg_ = Legend(
                            items=legend_it[int(0+pastit):int(pastit+it)])
                            #location=(0,15+pastit*5))
                        pastit+=it
                        p.add_layout(leg_, 'right')
                        leg_.click_policy="hide"  
                show(p)  
            elif (isinstance(CLASSID,pd.DataFrame) and isinstance(colorby,str)):
                
                if yaxis_log:
                    p = figure(tools=TOOLS, tooltips=TOOLTIPS,width=plotwidth,height=plotheight,title=plot_title,y_axis_type="log")  
                else:    
                    p = figure(tools=TOOLS, tooltips=TOOLTIPS,width=plotwidth,height=plotheight,title=plot_title)  
                   
                colormap = matplotlib.colormaps['rainbow']   
                individual_classes=phi.unique(CLASSID,colorby)
                different_colors=len(individual_classes)
                color_mapping=colormap(np.linspace(0,1,different_colors),1,True)
                bokeh_palette=["#%02x%02x%02x" % (r, g, b) for r, g, b in color_mapping[:,0:3]]    
                legend_it=[]
                for c in individual_classes:
                    ids_2_include=CLASSID[CLASSID.columns[0]][CLASSID[colorby]==c].values.tolist()
                    color_=bokeh_palette[individual_classes.index(c)]
                    leg_it=[]
                    for this_col_name in ids_2_include:
                        
                        y_=X[this_col_name].values  
                        if isinstance(x_axis,bool):
                            x_=list(range(1,X.shape[0]+1))
                        else:
                            if isinstance(x_axis,list):
                                x_=x_axis
                            elif isinstance(x_axis,np.ndarray):
                                x_=x_axis.tolist()
                            else:
                                print('x_axis must be a list or a numpy vector')
                                x_=list(range(1,X.shape[0]+1))
                        ObsNum_=x_
                        source = ColumnDataSource(data=dict(x=x_, y=y_,ColID=[this_col_name]*X.shape[0],ObsNum=ObsNum_))
                        glyph=p.line('x', 'y', source=source,line_color=color_,line_width=linewidth,#legend_label= this_col_name,
                               color=color_,line_alpha=line_alpha)
                        if add_marker:
                            glyphm=p.scatter('x', 'y', source=source,color=color_,size=markersize,fill_alpha =fill_alpha)
                            leg_it.extend([glyph,glyphm])
                        else:
                            leg_it.append(glyph)
                            
                    legend_it.append((c,leg_it ))        
                hline = Span(location=0, dimension='width', line_color='black', line_width=2)
                p.renderers.extend([hline])
                p.xaxis.axis_label = xaxis_label
                p.yaxis.axis_label = yaxis_label
                        
                if add_legend:
                    #legend_cols=3
                    ipc=[np.round(len(legend_it)/legend_cols)]*legend_cols              
                    ipc[-1]=len(legend_it)-sum(ipc[:-1])
                    pastit=0
                    for it in ipc:
                        leg_ = Legend(
                            items=legend_it[int(0+pastit):int(pastit+it)])
                            #location=(0,15+pastit*5))
                        pastit+=it
                        p.add_layout(leg_, 'right')
                        leg_.click_policy="hide"  
                show(p) 
                
    elif (isinstance(X,list) and ~isinstance(ncx_y_col,bool) and ~isinstance(ncx_x_col,bool) 
          and ~isinstance(ncx_id_col,bool) ): #each plot as a fifferent x-axis X is a list of dataframes
        # X is a list of dataframes
        # ncx_x_col
        # ncx_y_col -> Could this be a list ?? (plot more than one var in plot)
        # ncx_id_col
        
        if isinstance(ids_2_include,str):
            ids_2_include=[ids_2_include]
        elif isinstance(ids_2_include,bool):
            ids_2_include=[]
            #look into df list to collect list of identifiers
            for x_df in X:
                ids_2_include.append(x_df[ncx_id_col].values[0] ) 
        
        first_plot=True
        TOOLS = "save,wheel_zoom,box_zoom,pan,reset,box_select,lasso_select"
        TOOLTIPS = [
                    ("Obs #", "@ObsNum"),
                    ("(x,y)", "($x, $y)"),
                    ("ID: ","@ColID")
                    ] 
        rnd_num=timestr()          
        output_file("LinePlot"+rnd_num+".html",title=tab_title,mode='inline')
        if individual_plots:
            for i,this_col_name in enumerate(ids_2_include):

                x_df =X[ids_2_include.index(this_col_name)]
                
                y_=X[ids_2_include.index(this_col_name) ][ncx_y_col ].values
                x_=X[ids_2_include.index(this_col_name) ][ncx_x_col ].values

                ObsNum_=x_    
                if not(first_plot):                   
                    plot_title=''       
                if yaxis_log:
                    p = figure(tools=TOOLS, tooltips=TOOLTIPS,width=plotwidth,height=plotheight,title=plot_title,y_axis_type="log")  
                else:    
                    p = figure(tools=TOOLS, tooltips=TOOLTIPS,width=plotwidth,height=plotheight,title=plot_title)  
                   
                    
                source = ColumnDataSource(data=dict(x=x_, y=y_,ColID=[this_col_name]*x_df.shape[0],ObsNum=ObsNum_))
                p.xaxis.axis_label = ncx_x_col
                p.yaxis.axis_label = ncx_y_col
                if add_legend:
                    p.line('x', 'y', source=source,line_color=linecolor,color=markercolor,line_width=linewidth,line_alpha=line_alpha,legend_label= this_col_name)
                else:
                    p.line('x', 'y', source=source,line_color=linecolor,color=markercolor,line_width=linewidth,line_alpha=line_alpha)
                #p.circle('x', 'y', source=source)
                hline = Span(location=0, dimension='width', line_color='black', line_width=2)
                p.renderers.extend([hline])
                if add_marker:
                    p.scatter('x', 'y', source=source, color=markercolor,size=markersize,fill_alpha =fill_alpha)
                if first_plot:
                    p_list=[p]
                    first_plot=False
                else:
                    p_list.append(p)
            show(column(p_list))  
        else:
             
            if yaxis_log:
                p = figure(tools=TOOLS, tooltips=TOOLTIPS,width=plotwidth,height=plotheight,title=plot_title,y_axis_type="log")  
            else:    
                p = figure(tools=TOOLS, tooltips=TOOLTIPS,width=plotwidth,height=plotheight,title=plot_title)  
               
            colormap = matplotlib.colormaps['rainbow']
            different_colors=len(ids_2_include)
            color_mapping=colormap(np.linspace(0,1,different_colors),1,True)
            bokeh_palette=["#%02x%02x%02x" % (r, g, b) for r, g, b in color_mapping[:,0:3]]      
            legend_it=[]
            for this_col_name in ids_2_include:
                x_df =X[ids_2_include.index(this_col_name)]
                color_=bokeh_palette[ids_2_include.index(this_col_name)]
                
                y_=X[ids_2_include.index(this_col_name) ][ncx_y_col ].values
                x_=X[ids_2_include.index(this_col_name) ][ncx_x_col ].values

                ObsNum_=x_
                source = ColumnDataSource(data=dict(x=x_, y=y_,ColID=[this_col_name]*x_df.shape[0],ObsNum=ObsNum_))
                glyph=p.line('x', 'y', source=source,line_color=color_,line_width=linewidth,#legend_label= this_col_name,
                       color=color_,line_alpha=line_alpha)
                
                hline = Span(location=0, dimension='width', line_color='black', line_width=2)
                p.renderers.extend([hline])
                if add_marker:
                    glyphm=p.scatter('x', 'y', source=source,color=color_,size=markersize,fill_alpha =fill_alpha)
                    legend_it.append((this_col_name, [glyph,glyphm]))
                else:
                    legend_it.append((this_col_name, [glyph]))
            p.xaxis.axis_label = xaxis_label
            p.yaxis.axis_label = yaxis_label
            # if add_legend:
            #     legend = Legend(items=legend_it, location='top_right')
            #     legend.click_policy="hide"
            #     p.add_layout(legend, 'right')
                
            if add_legend:
                #legend_cols=3
                ipc=[np.round(len(legend_it)/legend_cols)]*legend_cols              
                ipc[-1]=len(legend_it)-sum(ipc[:-1])
                pastit=0
                for it in ipc:
                    leg_ = Legend(
                        items=legend_it[int(0+pastit):int(pastit+it)])
                        #location=(0,15+pastit*5))
                    pastit+=it
                    p.add_layout(leg_, 'right')
                    leg_.click_policy="hide"    
            show(p)  
    return



def plot_spectra(X,*,xaxis=False,plot_title='Main Title',tab_title='Tab Title',
                 xaxis_label='X- axis',yaxis_label='Y- axis',
                 linecolor='blue',linewidth=2): 
    """Simple way to plot Spectra with Bokeh. 

    plot_spectra(X,*,xaxis=False,plot_title='Main Title',tab_title='Tab Title',
                     xaxis_label='X- axis',yaxis_label='Y- axis',
                     linecolor='blue',linewidth=2)
    Args:
        
        X:      A numpy array or a pandas object with Spectra to be plotted
        xaxis:  wavenumbers or wavelengths to index the x axis of the plot 
                * ignored if X is a pandas dataframe *
            
    Optional Parameters: 
        plot_title
        tab_title
        xaxis_label
        yaxis_label
    
    Programmed by Salvador Garcia-Munoz
    (sgarciam@ic.ac.uk ,salvadorgarciamunoz@gmail.com)
    
    """
    
    if isinstance(X,pd.DataFrame):
        x=X.columns[1:].tolist()
        x=np.array(x)
        x=np.reshape(x,(1,-1))
        y=X.values[:,1:].astype(float)
    elif isinstance(X,np.ndarray):
        y=X.copy()
        if isinstance(xaxis,np.ndarray):
            x=xaxis
            x=np.reshape(x,(1,-1))
        elif isinstance(xaxis,list):
            x=np.array(xaxis)
            x=np.reshape(x,(1,-1))
        elif isinstance(xaxis,bool):
            x=np.array(list(range(X.shape[1])))
            x=np.reshape(x,(1,-1))
    #rnd_num=str(int(np.round(1000*np.random.random_sample())))
    rnd_num=timestr()                
    output_file("Spectra"+rnd_num+".html",title=tab_title,mode='inline')

    p = figure(title=plot_title)
    p.xaxis.axis_label = xaxis_label
    p.yaxis.axis_label = yaxis_label
    p.multi_line(x.tolist()*y.shape[0],y.tolist(),line_color=linecolor,line_width=linewidth)
    show(p)
    return

def scatter_with_labels(x,y,*,xlabel='X var',ylabel='YVar',labels=False,
                        tabtitle='Scatter Plot',plottitle='Scatter',legend_cols=1,
                        CLASSID=False,colorby=False,
                        plotwidth=600,plotheight=600,
                        markercolor='darkblue',markersize=10,
                        fill_alpha =0.2,line_alpha=0.4):
    rnd_num=timestr()
    output_file("Scatter Plot"+rnd_num+".html",title=tabtitle,mode='inline')    
    x_p = x
    y_p = y                        
    TOOLS = "save,wheel_zoom,box_zoom,pan,reset,box_select,lasso_select"
    TOOLTIPS = [
            ("index", "$index"),
            ("(x,y)", "($x, $y)"),
            ("Property","@names")
            ]
    if isinstance(labels,bool):
        labels=[]
        for i in np.arange(len(x)):
            labels.append('Obs '+str(i))
    
    p = figure(tools=TOOLS, tooltips=TOOLTIPS,width=plotwidth,height=plotheight, title=plottitle)
    
    if (isinstance(CLASSID,bool)) and (isinstance(colorby,bool)): # NO CLASSES
    
        source1 = ColumnDataSource(data=dict(x=x_p, y=y_p,names=labels))  
        p.scatter(x='x',y='y', source=source1,size=markersize,color=markercolor,
                  fill_alpha =fill_alpha,line_alpha=line_alpha)
        
    else: #isinstance(x, pd.DataFrame) and isinstance(y, pd.DataFrame): 
        
        #IF CLASSES THEN x and y and labels must be also dataframes
        legend_it = []
        if colorby in CLASSID.columns:   
            x_df=CLASSID.copy()
            x_df.insert(x_df.shape[1],'values',x)
            y_df=CLASSID.copy()
            y_df.insert(y_df.shape[1],'values',y)
            CLASSID_=CLASSID.copy()
            CLASSID_.insert(CLASSID_.shape[1],'labels',labels)
            Classes_=phi.unique(CLASSID,colorby)
            A=len(Classes_)
            colormap = matplotlib.colormaps['rainbow']
            different_colors=A
            color_mapping=colormap(np.linspace(0,1,different_colors),1,True)
            bokeh_palette=["#%02x%02x%02x" % (r, g, b) for r, g, b in color_mapping[:,0:3]]  
            
            for c in Classes_:
                x_aux = x_df['values'][CLASSID[colorby]==c]
                y_aux = y_df['values'][CLASSID[colorby]==c]
                labels_aux=CLASSID_['labels'][CLASSID[colorby]==c]
                source = ColumnDataSource(data=dict(x=x_aux, y=y_aux,names=labels_aux))        
                color_=bokeh_palette[Classes_.index(c)]
                glyph = p.scatter('x','y',source=source,color=color_,size=markersize,fill_alpha =fill_alpha,line_alpha=line_alpha)
                aux_=c
                if isinstance(aux_,(float,int)):
                    aux_=str(aux_)
                legend_it.append((aux_, [glyph]))
            #legend = Legend(items=legend_it, location='top_right')
            #legend.click_policy="hide"
            #p.add_layout(legend, 'right')
            

            ipc=[np.round(len(legend_it)/legend_cols)]*legend_cols              
            ipc[-1]=len(legend_it)-sum(ipc[:-1])
            pastit=0
            for it in ipc:
                leg_ = Legend(
                    items=legend_it[int(0+pastit):int(pastit+it)])
                    #location=(0,15+pastit*5))
                pastit+=it
                p.add_layout(leg_, 'right')
                leg_.click_policy="hide" 
                    

    
    p.xaxis.axis_label = xlabel
    p.yaxis.axis_label = ylabel
    
    vline = Span(location=0, dimension='height', line_color='black', line_width=2)
    # Horizontal line
    hline = Span(location=0, dimension='width', line_color='black', line_width=2)
    p.renderers.extend([vline, hline])
    show(p)   
    