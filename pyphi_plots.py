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
import matplotlib.cm as cm

def r2pv_plot(mvmobj):
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
    XVar = []   
    for n in list(np.arange(num_varX)+1):
        XVar.append('XVar #'+str(n))  
                    
    r2pvX_dict = {'XVar': XVar}
    
    for i in list(np.arange(A)):
        r2pvX_dict.update({lv_labels[i] : mvmobj['r2xpv'][:,i].tolist()})
    
    output_file("r2xpv.html")
    
    colormap =cm.get_cmap("rainbow")
    different_colors=A
    color_mapping=colormap(np.linspace(0,1,different_colors),1,True)
    bokeh_palette=["#%02x%02x%02x" % (r, g, b) for r, g, b in color_mapping[:,0:3]]  
                   
    p = figure(x_range=XVar, plot_height=400, title="R2X Per Variable",
           toolbar_location=None, tools="hover", tooltips="$name @XVar: @$name")

    p.vbar_stack(lv_labels, x='XVar', width=0.9,color=bokeh_palette,source=r2pvX_dict)
    p.y_range.range_padding = 0.1
    p.ygrid.grid_line_color = None
    p.axis.minor_tick_line_color = None
    p.outline_line_color = None

    show(p)
    return
    
    

