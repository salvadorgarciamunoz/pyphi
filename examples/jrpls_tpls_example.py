# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 19:54:20 2023

@author: salva
"""
import pandas as pd
import pyphi as phi
import pyphi_plots as pp

#Load, clean, make sure data matches
jr,materials=phi.parse_materials('jrpls_tpls_dataset.xlsx','Materials')
x=[]
for m in materials:
    x_=pd.read_excel( 'jrpls_tpls_dataset.xlsx',sheet_name=m)
    x.append(x_)
    
xc,jrc=phi.reconcile_rows_to_columns(x, jr)

quality=pd.read_excel('jrpls_tpls_dataset.xlsx',sheet_name='QUALITY')
process=pd.read_excel('jrpls_tpls_dataset.xlsx',sheet_name='PROCESS')

jrc.append(process)
jrc.append(quality)
AUX= phi.reconcile_rows(jrc)

JR_     = AUX[:-2]
process = AUX[-2]
quality = AUX[-1]


Ri={}
for j,m in zip(JR_,materials):
    Ri[m]=j
Xi={}
for x_,m in zip(xc,materials):
    Xi[m]=x_

del j,m,x_,JR_,AUX,jrc,jr,xc,x

jrplsobj=phi.jrpls(Xi,Ri,quality,4)

pp.r2pv(jrplsobj,material='MAT4')

pp.loadings( jrplsobj)
pp.loadings( jrplsobj,material='MAT4')

pp.loadings_map( jrplsobj, [1,2],textalpha=0.2)
pp.loadings_map( jrplsobj, [1,2],material='MAT2')

pp.loadings( jrplsobj)
pp.loadings( jrplsobj,material='MAT2')

pp.weighted_loadings( jrplsobj)
pp.weighted_loadings( jrplsobj,material='MAT2')

pp.score_scatter(jrplsobj,[1,2])
pp.score_scatter(jrplsobj,[1,2],rscores=True)
pp.score_scatter(jrplsobj,[1,2],rscores=True,material='MAT2',addtitle='Mat 2')


pp.vip(jrplsobj,plotwidth=1000 )
pp.vip(jrplsobj,plotwidth=1000 ,material='MAT4',addtitle='Mat4')

#Predict a new blend
# L001	A0129	0.557949425	API
# L001	A0130	0.442050575	API
# L001	Lac0003	1	Lactose
# L001	TLC018	1	Talc
# L001	M0012	1	MgSt
# L001	CS0017	1	Conf. Sugar

# L002	A0130	0.309885057	API
# L002	A0131	0.690114943	API
# L002	Lac0004	1	Lactose
# L002	TLC018	1	Talc
# L002	M0012	1	MgSt
# L002	CS0017	1	Conf. Sugar

rnew={
      'MAT1':        [('A0129',0.557949425 ),('A0130',0.442050575 )],
      'MAT2':    [('Lac0003',1)],
      'MAT3':       [('TLC018', 1) ],
      'MAT4':       [('M0012',  1)  ],
      'MAT5':[('CS0017', 1) ]
      }
jrpreds=phi.jrpls_pred(rnew,jrplsobj)
#%%

# Try a TPLS model
tplsobj=phi.tpls(Xi,Ri,process,quality,4)
#All plots for materials also apply only showing plots specific to the process block
pp.r2pv(tplsobj)
pp.r2pv(tplsobj,zspace=True)
pp.loadings( tplsobj,zspace=True)
pp.weighted_loadings( tplsobj,zspace=True)
pp.loadings_map(tplsobj,[1,2],plotwidth=800,zspace=True,addtitle=' Process Loadings')
pp.vip(tplsobj,plotwidth=1000,zspace=True,addtitle='Z Space')

#%%
# Predict a new blend
rnew={
      'MAT1':        [('A0129',0.557949425 ),('A0130',0.442050575 )],
      'MAT2':        [('Lac0003',1)],
      'MAT3':        [('TLC018', 1) ],
      'MAT4':        [('M0012',  1)  ],
      'MAT5':[('CS0017', 1) ]
      }
znew=process[process['LotID']=='L001']
znew=znew.values.reshape(-1)[1:].astype(float)
preds=phi.tpls_pred(rnew,znew,tplsobj)
pp.r2pv(tplsobj)
pp.r2pv(tplsobj,zspace=True)




