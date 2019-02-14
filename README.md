# pyphi
phi toolbox for multivariate analysis by Sal Garcia (salvadorgarciamunoz@gmail.com , sgarciam@ic.ac.uk )
version 1.0 includes: 
pca   : Principal Components Analysis
pls   : Projection to Latent Structures
savgol: Savitzy-Golay derivative transform
snv   : Standard Normal Variate transform

sample script usage:
===========================================
import pandas as pd
import numpy as np
import pyphi as phi
Cars_Features=pd.read_excel('Automobiles PLS.xls', 'Features', index_col=None, na_values=np.nan)
Cars_Features_data=np.array(Cars_Features.values[:,1:]).astype(float)

Cars_Performance=pd.read_excel('Automobiles PLS.xls', 'Performance', index_col=None, na_values=np.nan)
Cars_Performance_data=np.array(Cars_Performance.values[:,1:]).astype(float)


X=Cars_Features_data[19:59,:]
Y=Cars_Performance_data[19:59,:]
#full data set, will use SVD
pcaobj1=phi.pca(X,3)
#Data set with missing data will use NIPALS
pcaobj2=phi.pca(Cars_Features_data,3)
#Full data set, will use SVD
pls_obj1=phi.pls(X,Y,3)
#Data set with missing data, will use NIPALS
pls_obj2=phi.pls(Cars_Features_data,Cars_Performance_data,3)
#Full data set, forcing NIPALS
pls_obj3=phi.pls(X,Y,3,force_nipals=True)
