#%% example from Yu and MacGregor paper
import pyphi as phi
import pyphi_plots as pp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm




X_df = pd.read_excel('OPLS Test Data.xlsx','X')
Y_df = pd.read_excel('OPLS Test Data.xlsx','Y')
Tcv_opls=pd.read_excel('OPLS scores and loadings.xlsx','Tcv')
Pcv_opls=pd.read_excel('OPLS scores and loadings.xlsx','Pcv')

X=X_df.values
Y=Y_df.values


# Build a 5 LV's model adding CCA flag to calculate Covariant Scores and Loadings
pls_obj=phi.pls(X,Y,5,cca=True)


plt.figure()
plt.plot (pls_obj['Pcv'],'b-',label='by PyPhi using PLS-CCA')
plt.plot (Pcv_opls,'or',alpha=0.3,label='by SIMCA-P OPLS 4+1')
plt.ylabel('Covariant(Predictive) Loading')
plt.xlabel('Variable')
plt.title('PLS-CCA vs OPLS Loadings')
plt.legend()

plt.figure()
plt.plot (pls_obj['Tcv'],'b-',label='by PyPhi using PLS-CCA')
plt.plot (Tcv_opls,'or',alpha=0.3,label='by SIMCA-P OPLS 4+1')
plt.ylabel('Covariant(Predictive) Scores')
plt.xlabel('Observation')
plt.title('PLS-CCA vs OPLS Scores')
plt.legend()

plt.figure()
plt.plot (pls_obj['Tcv'],Tcv_opls,'o',)
plt.xlabel('by PyPhi using PLS-CCA')    
plt.ylabel('by SIMCA-P OPLS 4+1')
plt.title('PLS-CCA vs OPLS SCores')
