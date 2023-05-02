"""
Phi for Python (pyPhi)

by Salvador Garcia (sgarciam@ic.ac.uk salvadorgarciamunoz@gmail.com)
Added May 1st
        * YMB is now added in the same structure as the XMB
        * Corrected the dimensionality of the lwpls prediction, it was a double-nested array.
        
Added Apr 30 {feliz día de los niños}
        * Modified Multi-block PLS to include the block name in the variable name
Added Apr 29
        * Included the unique routine and adjusted the parse_materials routine so materials 
          and lots are in the same order as in the raw data
          
Added Apr 27
        * Enhanced adapt_pls_4_pyomo to use variable names as indices if flag is sent
Added Apr 25
        * Enhanced the varimax_rotation to adjust the r2 and r2pv to the rotated loadings
Added Apr 21
        * Re added varimax_rotation with complete model rotation for PCA and PLS
        
Added Apr 17 
        * Added tpls and tpls_pred
Added Apr 15 
        * Added jrpls model and jrpls_pred
        * Added routines to reconcile columns to rows identifier so that X and R materices
          correspond correctly
        * Added routines to reconcile rows across a list of dataframes and produces a list
          of dataframes containing only those observations present in all dataframes
          
Added on Apr 9 2023
        * Added lpls and lpls_pred routines
        * Added parse_materials to read linear table and produce R or Ri
        
Release as of Nov 23 2022
        * Added a function to export PLS model to gPROMS code

Release as of Aug 22 2022
What was done:
        *Fixed access to NEOS server and use of GAMS instead of IPOPT
        
Release as of Aug 12 2022
What was done:        
        * Fixed the SPE calculations in pls_pred and pca_pred
        * Changed to a more efficient inversion in pca_pred (=pls_pred)
        * Added a pseudo-inverse option in pmp for pca_pred
        
Relase as of now Aug 2 2022
What was done:
        *Added replicate_data

Release Unknown
What was done:
        * Fixed a bug in kernel PCA calculations
        * Changed the syntax of MBPLS arguments
        * Corrected a pretty severe error in pls_pred
        * Fixed a really bizzare one in mbpls

Release Dec 5, 2021
What was done:
        *Added some small documentation to utilities routines

Release Jan 15, 2021
What was done:
        * Added routine cat_2_matrix to conver categorical classifiers to matrices
        * Added Multi-block PLS model

Release Date: NOv 16, 2020
What was done:
        * Fixed small bug un clean_low_variances routine

Release Date: Sep 26 2020
What was done:
        * Added rotation of loadings so that var(t) for ti>=0 is always larger
          than var(t) for ti<0

Release Date: May 27 2020
What was done:
    * Added the estimation of PLS models with missind data using
    non-linear programming per  Journal of Chemometrics, 28(7), pp.575-584.

Release Date: March 30 2020
What was done:
    * Added the estimation of PCA models with missing data using
      non-linear programming per Lopez-Negrete et al. J. Chemometrics 2010; 24: 301–311

Release Date: Aug 22 2019

What was done:
    
    * This header is now included to track high level changes 
    * fixed LWPLS it works now for scalar and multivariable  Y's
    * fixed minor bug in phi.pca and phi.pls when mcsX/Y = False
    

"""

import numpy as np
import pandas as pd
import datetime
from scipy.special import factorial
from scipy import interpolate
from statsmodels.distributions.empirical_distribution import ECDF
from shutil import which
import os
from numpy import eye, asarray, dot, sum, diag
from numpy.linalg import svd

os.environ['NEOS_EMAIL'] = 'pyphisoftware@gmail.com' 

try:
    from pyomo.environ import *
    pyomo_ok = True
except ImportError:
    pyomo_ok = False

if bool(which('gams')):
    gams_ok = True    
else:
    gams_ok = False             # GAMS is run via pyomo

# Check if an IPOPT binary available is availbale
# shutil was introduced in Python 3.2
ipopt_ok = bool(which('ipopt'))

# Check for Pyomo/GAMS interface is available
if pyomo_ok and gams_ok:
    from pyomo.solvers.plugins.solvers.GAMS import GAMSDirect, GAMSShell
    # exeption_flag = True (default) will throw an exception if GAMS
    # is not available
    gams_ok = (GAMSDirect().available(exception_flag=False)
                or GAMSShell().available(exception_flag=False))

def ma57_dummy_check():
    """
    Instantiates a trivial NLP to solve with IPOPT and MA57.
    Returns:
          ma57_ok: boolean, True if IPOPT solved with SolverStaus.ok
    """
    m = ConcreteModel()
    m.x = Var()
    m.Obj = Objective(expr = m.x**2 -1)

    s = SolverFactory('ipopt')
    s.options['linear_solver'] = 'ma57'

    import logging
    pyomo_logger = logging.getLogger('pyomo.core')
    LOG_DEFAULT = pyomo_logger.level
    pyomo_logger.setLevel(logging.ERROR)
    r = s.solve(m)
    pyomo_logger.setLevel(LOG_DEFAULT)
    
    ma57_ok = r.solver.status == SolverStatus.ok
    if ma57_ok:
        print("MA57 available to IPOPT")
    return ma57_ok

if pyomo_ok and ipopt_ok:
    ma57_ok = ma57_dummy_check()
else:
    ma57_ok = False

if not(pyomo_ok) or (not(ipopt_ok) and not(gams_ok)):
    print('Will be using the NEOS server in the absence of IPOPT and GAMS')
    


def pca (X,A,*,mcs=True,md_algorithm='nipals',force_nipals=False,shush=False,cross_val=0):
    """ Principal Components Analysis routine
    
    by Salvador Garcia-Munoz 
    (sgarciam@ic.ac.uk ,salvadorgarciamunoz@gmail.com)
    
    Inputs:
        X : Either a pandas dataframe, or a Numpy Matrix
        
        A : Number of Principal Components to calculate
        
        mcs: 'True'      : Meancenter + autoscale *default if not sent* 
             'False'     : No pre-processing
             'center'    : Only center
             'autoscale' : Only autoscale
             
        md_algorithm: Missing Data algorithm to use
                      'nipals' *default if not sent*
                      'nlp'    Uses non-linear programming approach by Lopez-Negrete et al. J. Chemometrics 2010; 24: 301–311
                      
        force_nipals: If = True  will use NIPALS.
                         = False if X is complete will use SVD. *default if not sent*
                      
        shush: If = True supressess all printed output
                  =  False *default if not sent*
        
        cross_val: If sent a scalar between 0 and 100, will cross validate
                   element wise removing cross_val% of the data every round
                   
                   if ==   0:  Bypass cross-validation  *default if not sent*
    Output:
        A dictionary with all PCA loadings, scores and other diagnostics.
    
    """      
        
    if cross_val==0:
        pcaobj= pca_(X,A,mcs=mcs,md_algorithm=md_algorithm,force_nipals=force_nipals,shush=shush)
        pcaobj['type']='pca'
    elif (cross_val > 0) and (cross_val<100):
        if isinstance(X,np.ndarray):
            X_=X.copy()
        elif isinstance(X,pd.DataFrame):
            X_=np.array(X.values[:,1:]).astype(float)
        #Mean center and scale according to flags
        if isinstance(mcs,bool):
            if mcs:
                #Mean center and autoscale  
                X_,x_mean,x_std = meancenterscale(X_)
            else:    
                x_mean = np.zeros((1,X_.shape[1]))
                x_std  = np.ones((1,X_.shape[1]))
        elif mcs=='center':
            #only center
            X_,x_mean,x_std = meancenterscale(X_,mcs='center')
        elif mcs=='autoscale':
            #only autoscale
            X_,x_mean,x_std = meancenterscale(X_,mcs='autoscale')
            
        #Generate Missing Data Map    
        X_nan_map = np.isnan(X_)
        not_Xmiss = (np.logical_not(X_nan_map))*1
        

        #Initialize TSS per var vector
        X_,Xnanmap=n2z(X_)
        TSS   = np.sum(X_**2)
        TSSpv = np.sum(X_**2,axis=0)
        cols = X_.shape[1]
        rows = X_.shape[0]
        X_ = z2n(X_,Xnanmap)
        
        for a in list(range(A)):
            if not(shush):
                print('Cross validating PC #'+str(a+1))
            #Generate cross-val map starting from missing data
            not_removed_map = not_Xmiss.copy()
            not_removed_map = np.reshape(not_removed_map,(rows*cols,-1))
            #Generate matrix of random numbers and zero out nans
            Xrnd = np.random.random(X_.shape)*not_Xmiss
            indx = np.argsort(np.reshape(Xrnd,(Xrnd.shape[0]*Xrnd.shape[1])))
            elements_to_remove_per_round = np.int(np.ceil((X_.shape[0]*X_.shape[1]) * (cross_val/100)))
            error = np.zeros((rows*cols,1))
            rounds=1
            while np.sum(not_removed_map) > 0 :#While there are still elements to be removed
                #if not(shush):
                #   print('Removing samples round #'+str(rounds)+' for component :'+str(a+1))
                rounds=rounds+1          
                X_copy=X_.copy()
                if indx.size > elements_to_remove_per_round:
                    indx_this_round = indx[0:elements_to_remove_per_round]
                    indx = indx[elements_to_remove_per_round:]
                else: 
                    indx_this_round = indx
                #Place NaN's     
                X_copy                           = np.reshape(X_copy,(rows*cols,1))
                elements_out                     = X_copy[indx_this_round]
                X_copy[indx_this_round]          = np.nan
                X_copy                           = np.reshape(X_copy,(rows,cols))
                #update map
                not_removed_map[indx_this_round] = 0
                #look rows of missing data
                auxmap = np.isnan(X_copy)
                auxmap= (auxmap)*1
                auxmap=np.sum(auxmap,axis=1)
                indx2 = np.where(auxmap==X_copy.shape[1])
                indx2=indx2[0].tolist()
                if len(indx2) > 0:
                    X_copy=np.delete(X_copy,indx2,0)
                pcaobj_ = pca_(X_copy,1,mcs=False,shush=True)
                xhat    = pcaobj_['T'] @ pcaobj_['P'].T
                xhat    = np.insert(xhat, indx2,np.nan,axis=0)
                xhat    = np.reshape(xhat,(rows*cols,1))
                error[indx_this_round] = elements_out - xhat[indx_this_round]
            error = np.reshape(error,(rows,cols))
            error,dummy = n2z(error)
            PRESSpv = np.sum(error**2,axis=0)
            PRESS    = np.sum(error**2)
            
            if a==0:
                q2   = 1 - PRESS/TSS
                q2pv = 1 - PRESSpv/TSSpv
                q2pv = q2pv.reshape(-1,1)
            else:
                q2   = np.hstack((q2,1 - PRESS/TSS))
                aux_ = 1-PRESSpv/TSSpv
                aux_ = aux_.reshape(-1,1)
                q2pv = np.hstack((q2pv,aux_))
            
            #Deflate and go to next PC
            X_copy=X_.copy()
            pcaobj_ = pca_(X_copy,1,mcs=False,shush=True)
            xhat    = pcaobj_['T'] @ pcaobj_['P'].T
            X_,Xnanmap=n2z(X_)
            X_ = (X_ - xhat) * not_Xmiss
            if a==0:
                r2   = 1-np.sum(X_**2)/TSS
                r2pv = 1-np.sum(X_**2,axis=0)/TSSpv
                r2pv = r2pv.reshape(-1,1)
            else:
                r2   = np.hstack((r2,1-np.sum(X_**2)/TSS))
                aux_ = 1-np.sum(X_**2,axis=0)/TSSpv
                aux_ = aux_.reshape(-1,1)
                r2pv = np.hstack((r2pv,aux_))
            X_ = z2n(X_,Xnanmap)
            
        # Fit full model
        pcaobj = pca_(X,A,mcs=mcs,force_nipals=True,shush=True)
        for a in list(range(A-1,0,-1)):
             r2[a]     = r2[a]-r2[a-1]
             r2pv[:,a] = r2pv[:,a]-r2pv[:,a-1]
             q2[a]     = q2[a]-q2[a-1]
             q2pv[:,a] = q2pv[:,a]-q2pv[:,a-1]
        r2xc = np.cumsum(r2)
        q2xc = np.cumsum(q2)
        eigs = np.var(pcaobj['T'],axis=0)
        pcaobj['q2']   = q2
        pcaobj ['q2pv'] = q2pv
        
        if not(shush):   
            print('phi.pca using NIPALS and cross validation ('+str(cross_val)+'%) executed on: '+ str(datetime.datetime.now()) )            
            print('--------------------------------------------------------------')
            print('PC #          Eig      R2X     sum(R2X)      Q2X     sum(Q2X)')
            if A>1:
                for a in list(range(A)):
                    print("PC #"+str(a+1)+":   {:8.3f}    {:.3f}     {:.3f}       {:.3f}     {:.3f}".format(eigs[a], r2[a], r2xc[a],q2[a],q2xc[a]))
            else:
                d1=eigs[0]
                d2=r2xc[0]
                d3=q2xc[0]
                print("PC #"+str(a+1)+":   {:8.3f}    {:.3f}     {:.3f}       {:.3f}     {:.3f}".format(d1, r2, d2,q2,d3))
            print('--------------------------------------------------------------')     
        pcaobj['type']='pca'    
    else:
        pcaobj='Cannot cross validate  with those options'
    return pcaobj

def pca_(X,A,*,mcs=True,md_algorithm='nipals',force_nipals=False,shush=False):
    if isinstance(X,np.ndarray):
        X_=X.copy()
        obsidX = False
        varidX = False
    elif isinstance(X,pd.DataFrame):
        X_=np.array(X.values[:,1:]).astype(float)
        obsidX = X.values[:,0].astype(str)
        obsidX = obsidX.tolist()
        varidX = X.columns.values
        varidX = varidX[1:]
        varidX = varidX.tolist()
            
    if isinstance(mcs,bool):
        if mcs:
            #Mean center and autoscale  
            X_,x_mean,x_std = meancenterscale(X_)  
        else:    
            x_mean = np.zeros((1,X_.shape[1]))
            x_std  = np.ones((1,X_.shape[1]))
    elif mcs=='center':
        X_,x_mean,x_std = meancenterscale(X_,mcs='center')
        #only center
    elif mcs=='autoscale':
        #only autoscale
        X_,x_mean,x_std = meancenterscale(X_,mcs='autoscale')
        
    #Generate Missing Data Map    
    X_nan_map = np.isnan(X_)
    not_Xmiss = (np.logical_not(X_nan_map))*1
    
    if not(X_nan_map.any()) and not(force_nipals) and ((X_.shape[1]/X_.shape[0]>=10) or (X_.shape[0]/X_.shape[1]>=10)):
        #no missing elements
        if not(shush):
            print('phi.pca using SVD executed on: '+ str(datetime.datetime.now()) )
        TSS   = np.sum(X_**2)
        TSSpv = np.sum(X_**2,axis=0)
        if X_.shape[1]/X_.shape[0]>=10:
             [U,S,Th]   = np.linalg.svd(X_ @ X_.T)
             T          = Th.T 
             T          = T[:,0:A]
             P          = X_.T @ T
             for a in list(range(A)):
                 P[:,a] = P[:,a]/np.linalg.norm(P[:,a])
             T          = X_ @ P
        elif X_.shape[0]/X_.shape[1]>=10:
             [U,S,Ph]   = np.linalg.svd(X_.T @ X_)
             P          = Ph.T
             P          = P[:,0:A]
             T          = X_ @ P
        for a in list(range(A)):
            X_ = X_- T[:,[a]]@P[:,[a]].T
            if a==0:
                r2   = 1-np.sum(X_**2)/TSS
                r2pv = 1-np.sum(X_**2,axis=0)/TSSpv
                r2pv = r2pv.reshape(-1,1)
            else:
                r2   = np.hstack((r2,  1-np.sum(X_**2)/TSS))
                aux_ = 1-(np.sum(X_**2,axis=0)/TSSpv)
                r2pv = np.hstack((r2pv,aux_.reshape(-1,1)))
        for a in list(range(A-1,0,-1)):
             r2[a]     = r2[a]-r2[a-1]
             r2pv[:,a] = r2pv[:,a]-r2pv[:,a-1]
        pca_obj={'T':T,'P':P,'r2x':r2,'r2xpv':r2pv,'mx':x_mean,'sx':x_std}
        if not isinstance(obsidX,bool):
            pca_obj['obsidX']=obsidX
            pca_obj['varidX']=varidX
        eigs = np.var(T,axis=0);
        r2xc = np.cumsum(r2)
        if not(shush):
            print('--------------------------------------------------------------')
            print('PC #      Eig        R2X       sum(R2X) ')
            if A>1:
                for a in list(range(A)):
                    print("PC #"+str(a+1)+":   {:8.3f}    {:.3f}     {:.3f}".format(eigs[a], r2[a], r2xc[a]))
            else:
                d1=eigs[0]
                d2=r2xc[0]
                print("PC #"+str(a+1)+":   {:8.3f}    {:.3f}     {:.3f}".format(d1, r2, d2))
            print('--------------------------------------------------------------')      
        T2 = hott2(pca_obj,Tnew=T)
        n = T.shape[0]
        T2_lim99 = (((n-1)*(n+1)*A)/(n*(n-A)))*f99(A,(n-A))
        T2_lim95 = (((n-1)*(n+1)*A)/(n*(n-A)))*f95(A,(n-A))     
        speX = np.sum(X_**2,axis=1,keepdims=1)
        speX_lim95,speX_lim99 = spe_ci(speX)
        pca_obj['T2'] = T2
        pca_obj['T2_lim99']= T2_lim99
        pca_obj['T2_lim95']= T2_lim95
        pca_obj['speX']= speX
        pca_obj['speX_lim99']= speX_lim99
        pca_obj['speX_lim95']= speX_lim95
        return pca_obj
    else:
        if md_algorithm=='nipals':
             #use nipals
             if not(shush):
                 print('phi.pca using NIPALS executed on: '+ str(datetime.datetime.now()) )
             X_,dummy=n2z(X_)
             epsilon=1E-10
             maxit=5000
             TSS   = np.sum(X_**2)
             TSSpv = np.sum(X_**2,axis=0)
             #T=[];
             #P=[];
             #r2=[];
             #r2pv=[];
             #numIT=[];
             for a in list(range(A)):
                 # Select column with largest variance as initial guess
                 ti = X_[:,[np.argmax(std(X_))]]
                 Converged=False
                 num_it=0
                 while Converged==False:
                      #Step 1. p(i)=t' x(i)/t't
                      timat=np.tile(ti,(1,X_.shape[1]))
                      pi=(np.sum(X_*timat,axis=0))/(np.sum((timat*not_Xmiss)**2,axis=0))
                      #Step 2. Normalize p to unit length.
                      pi=pi/np.linalg.norm(pi)
                      #Step 3. tnew= (x*p) / (p'p);
                      pimat=np.tile(pi,(X_.shape[0],1))
                      tn= X_ @ pi.T
                      ptp=np.sum((pimat*not_Xmiss)**2,axis=1)
                      tn=tn/ptp
                      pi=pi.reshape(-1,1)
                      if abs((np.linalg.norm(ti)-np.linalg.norm(tn)))/(np.linalg.norm(ti)) < epsilon:
                          Converged=True
                      if num_it > maxit:
                          Converged=True
                      if Converged:
                          if (len(ti[ti<0])>0) and (len(ti[ti>0])>0): #if scores are above and below zero
                              if np.var(ti[ti<0]) > np.var(ti[ti>=0]):
                                 tn=-tn
                                 ti=-ti
                                 pi=-pi 
                          if not(shush):
                              print('# Iterations for PC #'+str(a+1)+': ',str(num_it))
                          if a==0:
                              T=tn.reshape(-1,1)
                              P=pi
                          else:
                              T=np.hstack((T,tn.reshape(-1,1)))
                              P=np.hstack((P,pi))                           
                          # Deflate X leaving missing as zeros (important!)
                          X_=(X_- ti @ pi.T)*not_Xmiss
                          if a==0:
                              r2   = 1-np.sum(X_**2)/TSS
                              r2pv = 1-np.sum(X_**2,axis=0)/TSSpv
                              r2pv = r2pv.reshape(-1,1)
                          else:
                              r2   = np.hstack((r2,1-np.sum(X_**2)/TSS))
                              aux_ = 1-np.sum(X_**2,axis=0)/TSSpv
                              aux_ = aux_.reshape(-1,1)
                              r2pv = np.hstack((r2pv,aux_))
                      else:
                          num_it = num_it + 1
                          ti = tn.reshape(-1,1)
                 if a==0:
                     numIT=num_it
                 else:
                     numIT=np.hstack((numIT,num_it))
                     
             for a in list(range(A-1,0,-1)):
                 r2[a]     = r2[a]-r2[a-1]
                 r2pv[:,a] = r2pv[:,a]-r2pv[:,a-1]
                 
             eigs = np.var(T,axis=0);
             r2xc = np.cumsum(r2)
             if not(shush):               
                 print('--------------------------------------------------------------')
                 print('PC #      Eig        R2X       sum(R2X) ')
 
                 if A>1:
                     for a in list(range(A)):
                         print("PC #"+str(a+1)+":   {:8.3f}    {:.3f}     {:.3f}".format(eigs[a], r2[a], r2xc[a]))
                 else:
                     d1=eigs[0]
                     d2=r2xc[0]
                     print("PC #"+str(a+1)+":   {:8.3f}    {:.3f}     {:.3f}".format(d1, r2, d2))
                 print('--------------------------------------------------------------')      
        
             pca_obj={'T':T,'P':P,'r2x':r2,'r2xpv':r2pv,'mx':x_mean,'sx':x_std}    
             if not isinstance(obsidX,bool):
                 pca_obj['obsidX']=obsidX
                 pca_obj['varidX']=varidX
                 
             T2 = hott2(pca_obj,Tnew=T)
             n = T.shape[0]
             T2_lim99 = (((n-1)*(n+1)*A)/(n*(n-A)))*f99(A,(n-A))
             T2_lim95 = (((n-1)*(n+1)*A)/(n*(n-A)))*f95(A,(n-A))     
             speX = np.sum(X_**2,axis=1,keepdims=1)
             speX_lim95,speX_lim99 = spe_ci(speX)
             pca_obj['T2'] = T2
             pca_obj['T2_lim99']= T2_lim99
             pca_obj['T2_lim95']= T2_lim95
             pca_obj['speX']= speX
             pca_obj['speX_lim99']= speX_lim99
             pca_obj['speX_lim95']= speX_lim95
             return pca_obj                            
        elif md_algorithm=='nlp' and pyomo_ok:
            #use NLP per Lopez-Negrete et al. J. Chemometrics 2010; 24: 301–311
            if not(shush):
                 print('phi.pca using NLP with Ipopt executed on: '+ str(datetime.datetime.now()) )
            pcaobj_= pca_(X,A,mcs=mcs,md_algorithm='nipals',shush=True)
            pcaobj_= prep_pca_4_MDbyNLP(pcaobj_,X_)
          
            TSS   = np.sum(X_**2)
            TSSpv = np.sum(X_**2,axis=0)
            
            #Set up the model in Pyomo
            model             = ConcreteModel()
            model.A           = Set(initialize = pcaobj_['pyo_A'] )
            model.N           = Set(initialize = pcaobj_['pyo_N'] )
            model.O           = Set(initialize = pcaobj_['pyo_O'] )
            model.P           = Var(model.N,model.A, within = Reals,initialize = pcaobj_['pyo_P_init'])
            model.T           = Var(model.O,model.A, within = Reals,initialize = pcaobj_['pyo_T_init'])
            model.psi         = Param(model.O,model.N,initialize = pcaobj_['pyo_psi'])
            model.X           = Param(model.O,model.N,initialize = pcaobj_['pyo_X'])
            model.delta       = Param(model.A, model.A, initialize=lambda model, a1, a2: 1.0 if a1==a2 else 0)
            
            # Constraints 20b
            def _c20b_con(model, a1, a2):
                return sum(model.P[j, a1] * model.P[j, a2] for j in model.N) == model.delta[a1, a2]
            model.c20b = Constraint(model.A, model.A, rule=_c20b_con)
    
            # Constraints 20c
            def _20c_con(model, a1, a2):
                if a2 < a1:
                    return sum(model.T[o, a1] * model.T[o, a2] for o in model.O) == 0
                else:
                    return Constraint.Skip
            model.c20c = Constraint(model.A, model.A, rule=_20c_con)

            # Constraints 20d
            def mean_zero(model,i):
                return sum (model.T[o,i]  for o in model.O )==0
            model.eq3 = Constraint(model.A,rule=mean_zero)

            def _eq_20a_obj(model):
                return sum(sum((model.X[o,n]- model.psi[o,n] * sum(model.T[o,a] * model.P[n,a] for a in model.A))**2 for n in model.N) for o in model.O)
            model.obj = Objective(rule=_eq_20a_obj)            
            # Setup our solver as either local ipopt, gams:ipopt, or neos ipopt:
            if (ipopt_ok):
                print("Solving NLP using local IPOPT executable")
                solver = SolverFactory('ipopt')

                if (ma57_ok):
                    solver.options['linear_solver'] = 'ma57'

                results = solver.solve(model,tee=True)
            elif (gams_ok):
                print("Solving NLP using GAMS/IPOPT interface")
                # 'just 'ipopt' could work, if no binary in path
                solver = SolverFactory('gams:ipopt')

                # It doesn't seem to notice the opt file when I write it
                results = solver.solve(model, tee=True)
            else:                
                print("Solving NLP using IPOPT on remote NEOS server")
                solver_manager = SolverManagerFactory('neos')
                results = solver_manager.solve(model, opt='ipopt', tee=True)

            T=[]
            for o in model.O:
                 t=[]
                 for a in model.A:
                    t.append(value(model.T[o,a]))
                 T.append(t)   
            T=np.array(T)     
            P=[]
            for n in model.N:
                 p=[]
                 for a in model.A:
                    p.append(value(model.P[n,a]))
                 P.append(p)   
            P=np.array(P)   
            
            # Calculate R2
           
            for a in list(range(0, A)):
                 ti=T[:,[a]]
                 pi=P[:,[a]]
                 if np.var(ti[ti<0]) > np.var(ti[ti>=0]):
                    ti=-ti
                    pi=-pi 
                    T[:,[a]]=-T[:,[a]]
                    P[:,[a]]=-P[:,[a]]                           
                 X_=(X_- ti @ pi.T)*not_Xmiss
                 if a==0:
                    r2   = 1-np.sum(X_**2)/TSS
                    r2pv = 1-np.sum(X_**2,axis=0)/TSSpv
                    r2pv = r2pv.reshape(-1,1)
                 else:
                    r2   = np.hstack((r2,1-np.sum(X_**2)/TSS))
                    aux_ = 1-np.sum(X_**2,axis=0)/TSSpv
                    aux_ = aux_.reshape(-1,1)
                    r2pv = np.hstack((r2pv,aux_))
                    
            for a in list(range(A-1,0,-1)):
                 r2[a]     = r2[a]-r2[a-1]
                 r2pv[:,a] = r2pv[:,a]-r2pv[:,a-1]
                 
            eigs = np.var(T,axis=0);
            r2xc = np.cumsum(r2)
            if not(shush):               
                 print('--------------------------------------------------------------')
                 print('PC #      Eig        R2X       sum(R2X) ')
 
                 if A>1:
                     for a in list(range(A)):
                         print("PC #"+str(a+1)+":   {:8.3f}    {:.3f}     {:.3f}".format(eigs[a], r2[a], r2xc[a]))
                 else:
                     d1=eigs[0]
                     d2=r2xc[0]
                     print("PC #"+str(a+1)+":   {:8.3f}    {:.3f}     {:.3f}".format(d1, r2, d2))
                 print('--------------------------------------------------------------')      
        
            pca_obj={'T':T,'P':P,'r2x':r2,'r2xpv':r2pv,'mx':x_mean,'sx':x_std}    
            if not isinstance(obsidX,bool):
                 pca_obj['obsidX']=obsidX
                 pca_obj['varidX']=varidX
                 
            T2 = hott2(pca_obj,Tnew=T)
            n = T.shape[0]
            T2_lim99 = (((n-1)*(n+1)*A)/(n*(n-A)))*f99(A,(n-A))
            T2_lim95 = (((n-1)*(n+1)*A)/(n*(n-A)))*f95(A,(n-A))     
            speX = np.sum(X_**2,axis=1,keepdims=1)
            speX_lim95,speX_lim99 = spe_ci(speX)
            pca_obj['T2'] = T2
            pca_obj['T2_lim99']= T2_lim99
            pca_obj['T2_lim95']= T2_lim95
            pca_obj['speX']= speX
            pca_obj['speX_lim99']= speX_lim99
            pca_obj['speX_lim95']= speX_lim95
            return pca_obj                  
            
        elif md_algorithm=='nlp' and not( pyomo_ok):
            print('Pyomo was not found in your system sorry')
            print('visit  http://www.pyomo.org/ ')
            pca_obj=1
            return pca_obj
  
def pls(X,Y,A,*,mcsX=True,mcsY=True,md_algorithm='nipals',force_nipals=True,shush=False,cross_val=0,cross_val_X=False):
    """ Projection to  Latent Structures routine
    
    by Salvador Garcia-Munoz 
    (sgarciam@ic.ac.uk ,salvadorgarciamunoz@gmail.com)
    
    Inputs:
        X,Y : Either a pandas dataframe, or a Numpy Matrix
        
        A : Number of Principal Components to calculate
        
        mcsX/mcsY:  'True'      : Will meancenter and autoscale the data *default if not sent*  
                    'False'     : No pre-processing
                    'center'    : Will only center
                    'autoscale' : Will only autoscale
             
        md_algorithm: 'nipals' *default*
                      'nlp'    Uses  algorithm described in Journal of Chemometrics, 28(7), pp.575-584.
                      
        force_nipals: If set to True and if X is complete, will use NIPALS.
                      Otherwise, if X is complete will use SVD.
                      
        shush: If set to True supressess all printed output.
        
        cross_val: If sent a scalar between 0 and 100, will cross validate
                   element wise removing cross_val% of the data every round
                   
                   if ==   0:  Bypass cross-validation  *default if not sent*
                   
        cross_val_X: 'True' : Calculates Q2 values for the X and Y matrices
                     'False': Cross-validation strictly on Y matrix *default if not sent*
    
    Output:
        A dictionary with all PLS loadings, scores and other diagnostics.
    
    """
    if cross_val==0:
        plsobj = pls_(X,Y,A,mcsX=mcsX,mcsY=mcsY,md_algorithm=md_algorithm,force_nipals=force_nipals,shush=shush)  
        plsobj['type']='pls' 
    elif (cross_val > 0) and (cross_val<100):
        if isinstance(X,np.ndarray):
            X_=X.copy()
        elif isinstance(X,pd.DataFrame):
            X_=np.array(X.values[:,1:]).astype(float)
        #Mean center and scale according to flags
        if isinstance(mcsX,bool):
            if mcsX:
                #Mean center and autoscale  
                X_,x_mean,x_std = meancenterscale(X_)
            else:    
                x_mean = np.zeros((1,X_.shape[1]))
                x_std  = np.ones((1,X_.shape[1]))
        elif mcsX=='center':
            #only center
            X_,x_mean,x_std = meancenterscale(X_,mcs='center')
        elif mcsX=='autoscale':
            #only autoscale
            X_,x_mean,x_std = meancenterscale(X_,mcs='autoscale')
        #Generate Missing Data Map    
        X_nan_map = np.isnan(X_)
        not_Xmiss = (np.logical_not(X_nan_map))*1
        
        if isinstance(Y,np.ndarray):
            Y_=Y.copy()
        elif isinstance(Y,pd.DataFrame):
            Y_=np.array(Y.values[:,1:]).astype(float)
        #Mean center and scale according to flags
        if isinstance(mcsY,bool):
            if mcsY:
                #Mean center and autoscale  
                Y_,y_mean,y_std = meancenterscale(Y_)
            else:    
                y_mean = np.zeros((1,Y_.shape[1]))
                y_std  = np.ones((1,Y_.shape[1]))
        elif mcsY=='center':
            #only center
            Y_,y_mean,y_std = meancenterscale(Y_,mcs='center')
        elif mcsY=='autoscale':
            #only autoscale
            Y_,y_mean,y_std = meancenterscale(Y_,mcs='autoscale')
        #Generate Missing Data Map    
        Y_nan_map = np.isnan(Y_)
        not_Ymiss = (np.logical_not(Y_nan_map))*1
        
        #Initialize TSS per var vector
        X_,Xnanmap=n2z(X_)
        TSSX   = np.sum(X_**2)
        TSSXpv = np.sum(X_**2,axis=0)
        colsX = X_.shape[1]
        rowsX = X_.shape[0]
        X_ = z2n(X_,Xnanmap)
        
        Y_,Ynanmap=n2z(Y_)
        TSSY   = np.sum(Y_**2)
        TSSYpv = np.sum(Y_**2,axis=0)
        colsY = Y_.shape[1]
        rowsY = Y_.shape[0]
        Y_ = z2n(Y_,Ynanmap)
        
        
        for a in list(range(A)):
            if not(shush):
                print('Cross validating LV #'+str(a+1))
            #Generate cross-val map starting from missing data
            not_removed_mapY = not_Ymiss.copy()
            not_removed_mapY = np.reshape(not_removed_mapY,(rowsY*colsY,-1))
            #Generate matrix of random numbers and zero out nans
            Yrnd = np.random.random(Y_.shape)*not_Ymiss
            indxY = np.argsort(np.reshape(Yrnd,(Yrnd.shape[0]*Yrnd.shape[1])))
            elements_to_remove_per_roundY = np.int(np.ceil((Y_.shape[0]*Y_.shape[1]) * (cross_val/100)))
            errorY = np.zeros((rowsY*colsY,1))
                
            if cross_val_X:
                #Generate cross-val map starting from missing data
                not_removed_mapX = not_Xmiss.copy()
                not_removed_mapX = np.reshape(not_removed_mapX,(rowsX*colsX,-1))
                #Generate matrix of random numbers and zero out nans
                Xrnd = np.random.random(X_.shape)*not_Xmiss
                indxX = np.argsort(np.reshape(Xrnd,(Xrnd.shape[0]*Xrnd.shape[1])))
                elements_to_remove_per_roundX = np.int(np.ceil((X_.shape[0]*X_.shape[1]) * (cross_val/100)))
                errorX = np.zeros((rowsX*colsX,1))
            else:
                not_removed_mapX=0
                
            number_of_rounds=1    
            while np.sum(not_removed_mapX) > 0 or np.sum(not_removed_mapY) > 0 :#While there are still elements to be removed
                #if not(shush):
                #    print('Random removal round #'+ str(number_of_rounds))
                number_of_rounds=number_of_rounds+1
                X_copy=X_.copy()
                if cross_val_X:
                    if indxX.size > elements_to_remove_per_roundX:
                        indx_this_roundX = indxX[0:elements_to_remove_per_roundX]
                        indxX = indxX[elements_to_remove_per_roundX:]
                    else: 
                        indx_this_roundX = indxX
                    #Place NaN's     
                    X_copy                            = np.reshape(X_copy,(rowsX*colsX,1))
                    elements_outX                     = X_copy[indx_this_roundX]
                    X_copy[indx_this_roundX]          = np.nan
                    X_copy                            = np.reshape(X_copy,(rowsX,colsX))
                    #update map
                    not_removed_mapX[indx_this_roundX] = 0
                    #look rows of missing data
                    auxmap = np.isnan(X_copy)
                    auxmap= (auxmap)*1
                    auxmap=np.sum(auxmap,axis=1)
                    indx2 = np.where(auxmap==X_copy.shape[1])
                    indx2=indx2[0].tolist()
                else:
                    indx2=[];

                        
                Y_copy=Y_.copy()        
                if indxY.size > elements_to_remove_per_roundY:
                    indx_this_roundY = indxY[0:elements_to_remove_per_roundY]
                    indxY = indxY[elements_to_remove_per_roundY:]
                else:                      
                    indx_this_roundY = indxY
                #Place NaN's     
                Y_copy                            = np.reshape(Y_copy,(rowsY*colsY,1))
                elements_outY                     = Y_copy[indx_this_roundY]
                Y_copy[indx_this_roundY]          = np.nan
                Y_copy                            = np.reshape(Y_copy,(rowsY,colsY))
                #update map
                not_removed_mapY[indx_this_roundY] = 0
                #look rows of missing data
                auxmap = np.isnan(Y_copy)
                auxmap = (auxmap)*1
                auxmap = np.sum(auxmap,axis=1)
                indx3  = np.where(auxmap==Y_copy.shape[1])
                indx3  = indx3[0].tolist()
                indx4  = np.unique(indx3+indx2)
                indx4  = indx4.tolist()
                if len(indx4) > 0:
                    X_copy=np.delete(X_copy,indx4,0)
                    Y_copy=np.delete(Y_copy,indx4,0)
                #print('Running PLS')        
                plsobj_ = pls_(X_copy,Y_copy,1,mcsX=False,mcsY=False,shush=True)
                #print('Done with PLS')
                plspred = pls_pred(X_,plsobj_)
                
                if cross_val_X:
                    xhat    = plspred['Tnew'] @ plsobj_['P'].T
                    xhat    = np.reshape(xhat,(rowsX*colsX,1))
                    errorX[indx_this_roundX] = elements_outX - xhat[indx_this_roundX]
                
                yhat    = plspred['Tnew'] @ plsobj_['Q'].T
                yhat    = np.reshape(yhat,(rowsY*colsY,1))
                errorY[indx_this_roundY] = elements_outY - yhat[indx_this_roundY]
                
            if cross_val_X:
                errorX = np.reshape(errorX,(rowsX,colsX))
                errorX,dummy = n2z(errorX)
                PRESSXpv  = np.sum(errorX**2,axis=0)
                PRESSX    = np.sum(errorX**2)
            
            errorY = np.reshape(errorY,(rowsY,colsY))
            errorY,dummy = n2z(errorY)
            PRESSYpv  = np.sum(errorY**2,axis=0)
            PRESSY    = np.sum(errorY**2)
            
            if a==0:
                q2Y   = 1 - PRESSY/TSSY
                q2Ypv = 1 - PRESSYpv/TSSYpv
                q2Ypv = q2Ypv.reshape(-1,1)
                if cross_val_X:
                    q2X   = 1 - PRESSX/TSSX
                    q2Xpv = 1 - PRESSXpv/TSSXpv
                    q2Xpv = q2Xpv.reshape(-1,1)
            else:
                q2Y   = np.hstack((q2Y,1 - PRESSY/TSSY))
                aux_  = 1-PRESSYpv/TSSYpv
                aux_  = aux_.reshape(-1,1)
                q2Ypv = np.hstack((q2Ypv,aux_))
                if cross_val_X:
                    q2X   = np.hstack((q2X,1 - PRESSX/TSSX))
                    aux_  = 1-PRESSXpv/TSSXpv
                    aux_  = aux_.reshape(-1,1)
                    q2Xpv = np.hstack((q2Xpv,aux_)) 
            
            #Deflate and go to next PC
            X_copy=X_.copy()
            Y_copy=Y_.copy()
            plsobj_ = pls_(X_copy,Y_copy,1,mcsX=False,mcsY=False,shush=True)
            xhat    = plsobj_['T'] @ plsobj_['P'].T
            yhat    = plsobj_['T'] @ plsobj_['Q'].T
            X_,Xnanmap=n2z(X_)
            Y_,Ynanmap=n2z(Y_)
            X_ = (X_ - xhat) * not_Xmiss
            Y_ = (Y_ - yhat) * not_Ymiss
            if a==0:
                r2X   = 1-np.sum(X_**2)/TSSX
                r2Xpv = 1-np.sum(X_**2,axis=0)/TSSXpv
                r2Xpv = r2Xpv.reshape(-1,1)
                r2Y   = 1-np.sum(Y_**2)/TSSY
                r2Ypv = 1-np.sum(Y_**2,axis=0)/TSSYpv
                r2Ypv = r2Ypv.reshape(-1,1)
                
            else:
                r2X   = np.hstack((r2X,1-np.sum(X_**2)/TSSX))
                aux_  = 1-np.sum(X_**2,axis=0)/TSSXpv
                aux_  = aux_.reshape(-1,1)
                r2Xpv = np.hstack((r2Xpv,aux_))
                
                r2Y   = np.hstack((r2Y,1-np.sum(Y_**2)/TSSY))
                aux_  = 1-np.sum(Y_**2,axis=0)/TSSYpv
                aux_  = aux_.reshape(-1,1)
                r2Ypv = np.hstack((r2Ypv,aux_))               
            X_ = z2n(X_,Xnanmap)
            Y_ = z2n(Y_,Ynanmap)
            
        # Fit full model
        plsobj = pls_(X,Y,A,mcsX=mcsX,mcsY=mcsY,shush=True)
        for a in list(range(A-1,0,-1)):
             r2X[a]     = r2X[a]-r2X[a-1]
             r2Xpv[:,a] = r2Xpv[:,a]-r2Xpv[:,a-1]
             if cross_val_X:
                 q2X[a]     = q2X[a]-q2X[a-1]
                 q2Xpv[:,a] = q2Xpv[:,a]-q2Xpv[:,a-1]
             else:
                 q2X   = False
                 q2Xpv = False
             
             r2Y[a]     = r2Y[a]-r2Y[a-1]
             r2Ypv[:,a] = r2Ypv[:,a]-r2Ypv[:,a-1]
             q2Y[a]     = q2Y[a]-q2Y[a-1]
             q2Ypv[:,a] = q2Ypv[:,a]-q2Ypv[:,a-1]
             
        r2xc = np.cumsum(r2X)
        r2yc = np.cumsum(r2Y)
        if cross_val_X:
            q2xc = np.cumsum(q2X)
        else:
            q2xc = False
        q2yc = np.cumsum(q2Y)    
        eigs = np.var(plsobj['T'],axis=0)
        
        plsobj['q2Y']   = q2Y
        plsobj['q2Ypv'] = q2Ypv
        if cross_val_X:
            plsobj['q2X']   = q2X
            plsobj['q2Xpv'] = q2Xpv
        
        if not(shush):
            print('phi.pls using NIPALS and cross-validation ('+str(cross_val)+'%) executed on: '+ str(datetime.datetime.now()) )
            if not(cross_val_X):
                print('---------------------------------------------------------------------------------')
                print('PC #       Eig      R2X     sum(R2X)      R2Y     sum(R2Y)      Q2Y     sum(Q2Y)')
                if A>1:
                    for a in list(range(A)):
                        print("PC #"+str(a+1)+":{:8.3f}    {:.3f}     {:.3f}       {:.3f}     {:.3f}       {:.3f}     {:.3f}".format(eigs[a], r2X[a], r2xc[a], r2Y[a], r2yc[a],q2Y[a],q2yc[a]))
                else:
                    d1=eigs[0]
                    d2=r2xc[0]
                    d3=r2yc[0]
                    d4=q2yc[0]
                    print("PC #"+str(a+1)+":{:8.3f}    {:.3f}     {:.3f}       {:.3f}        {:.3f}    {:.3f}     {:.3f}".format(d1, r2X, d2,r2Y,d3,q2Y,d4))
                print('---------------------------------------------------------------------------------')     
            else:
                print('-------------------------------------------------------------------------------------------------------')
                print('PC #       Eig      R2X     sum(R2X)      Q2X     sum(Q2X)      R2Y     sum(R2Y)      Q2Y     sum(Q2Y)')
                if A>1:
                    for a in list(range(A)):
                        print("PC #"+str(a+1)+":{:8.3f}    {:.3f}     {:.3f}       {:.3f}     {:.3f}       {:.3f}     {:.3f}       {:.3f}     {:.3f}".format(eigs[a], r2X[a], r2xc[a],q2X[a],q2xc[a], r2Y[a], r2yc[a],q2Y[a],q2yc[a]))
                else:
                    d1=eigs[0]
                    d2=r2xc[0]
                    d3=q2xc[0]
                    d4=r2yc[0]
                    d5=q2yc[0]
                    print("PC #"+str(a+1)+":{:8.3f}    {:.3f}     {:.3f}       {:.3f}     {:.3f}       {:.3f}     {:.3f}       {:.3f}     {:.3f}".format(d1, r2X, d2,q2X,d3,r2Y,d4,q2Y,d5))
                print('-------------------------------------------------------------------------------------------------------')   
        plsobj['type']='pls'
    elif cross_val==100:
    
          
        if isinstance(X,np.ndarray):
            X_=X.copy()
        elif isinstance(X,pd.DataFrame):
            X_=np.array(X.values[:,1:]).astype(float)
        #Mean center and scale according to flags
        if isinstance(mcsX,bool):
            if mcsX:
                #Mean center and autoscale  
                X_,x_mean,x_std = meancenterscale(X_)
            else:    
                x_mean = np.zeros((1,X_.shape[1]))
                x_std  = np.ones((1,X_.shape[1]))
        elif mcsX=='center':
            #only center
            X_,x_mean,x_std = meancenterscale(X_,mcs='center')
        elif mcsX=='autoscale':
            #only autoscale
            X_,x_mean,x_std = meancenterscale(X_,mcs='autoscale')
        #Generate Missing Data Map    
        X_nan_map = np.isnan(X_)
        not_Xmiss = (np.logical_not(X_nan_map))*1
        
        if isinstance(Y,np.ndarray):
            Y_=Y.copy()
        elif isinstance(Y,pd.DataFrame):
            Y_=np.array(Y.values[:,1:]).astype(float)
        #Mean center and scale according to flags
        if isinstance(mcsY,bool):
            if mcsY:
                #Mean center and autoscale  
                Y_,y_mean,y_std = meancenterscale(Y_)
            else:    
                y_mean = np.zeros((1,Y_.shape[1]))
                y_std  = np.ones((1,Y_.shape[1]))
        elif mcsY=='center':
            #only center
            Y_,y_mean,y_std = meancenterscale(Y_,mcs='center')
        elif mcsY=='autoscale':
            #only autoscale
            Y_,y_mean,y_std = meancenterscale(Y_,mcs='autoscale')
        #Generate Missing Data Map    
        Y_nan_map = np.isnan(Y_)
        not_Ymiss = (np.logical_not(Y_nan_map))*1
        
        #Initialize TSS per var vector
        X_,Xnanmap=n2z(X_)
        TSSX   = np.sum(X_**2)
        TSSXpv = np.sum(X_**2,axis=0)
        colsX = X_.shape[1]
        rowsX = X_.shape[0]
        X_ = z2n(X_,Xnanmap)
        
        Y_,Ynanmap=n2z(Y_)
        TSSY   = np.sum(Y_**2)
        TSSYpv = np.sum(Y_**2,axis=0)
        colsY = Y_.shape[1]
        rowsY = Y_.shape[0]
        Y_ = z2n(Y_,Ynanmap)
        
        
        for a in list(range(A)):
            errorY = np.zeros((rowsY*colsY,1))               
            if cross_val_X:
                errorX = np.zeros((rowsX*colsX,1))
                
            for o in list(range(X.shape[0])): # Removing one at a time
                X_copy=X_.copy()
                Y_copy=Y_.copy()   
                
                elements_outX =X_copy[o,:].copy()
                elements_outY =Y_copy[o,:].copy()
                X_copy = np.delete(X_copy,o,0)
                Y_copy = np.delete(Y_copy,o,0)
        
                plsobj_ = pls_(X_copy,Y_copy,1,mcsX=False,mcsY=False,shush=True)
                plspred = pls_pred(elements_outX,plsobj_)
                
                if o==0:
                    if cross_val_X:
                        errorX= elements_outX - plspred['Xhat']
                    errorY= elements_outY - plspred['Yhat']
                else:
                    if cross_val_X:
                        errorX= np.vstack((errorX,elements_outX - plspred['Xhat']))
                    errorY= np.vstack((errorY,elements_outY - plspred['Yhat']))
                  
            if cross_val_X:
                errorX,dummy = n2z(errorX)
                PRESSXpv  = np.sum(errorX**2,axis=0)
                PRESSX    = np.sum(errorX**2)
            
            errorY,dummy = n2z(errorY)
            PRESSYpv  = np.sum(errorY**2,axis=0)
            PRESSY    = np.sum(errorY**2)
            
            if a==0:
                q2Y   = 1 - PRESSY/TSSY
                q2Ypv = 1 - PRESSYpv/TSSYpv
                q2Ypv = q2Ypv.reshape(-1,1)
                if cross_val_X:
                    q2X   = 1 - PRESSX/TSSX
                    q2Xpv = 1 - PRESSXpv/TSSXpv
                    q2Xpv = q2Xpv.reshape(-1,1)
            else:
                q2Y   = np.hstack((q2Y,1 - PRESSY/TSSY))
                aux_  = 1-PRESSYpv/TSSYpv
                aux_  = aux_.reshape(-1,1)
                q2Ypv = np.hstack((q2Ypv,aux_))
                if cross_val_X:
                    q2X   = np.hstack((q2X,1 - PRESSX/TSSX))
                    aux_  = 1-PRESSXpv/TSSXpv
                    aux_  = aux_.reshape(-1,1)
                    q2Xpv = np.hstack((q2Xpv,aux_)) 
            
            #Deflate and go to next PC
            X_copy=X_.copy()
            Y_copy=Y_.copy()
            plsobj_ = pls_(X_copy,Y_copy,1,mcsX=False,mcsY=False,shush=True)
            xhat    = plsobj_['T'] @ plsobj_['P'].T
            yhat    = plsobj_['T'] @ plsobj_['Q'].T
            X_,Xnanmap=n2z(X_)
            Y_,Ynanmap=n2z(Y_)
            X_ = (X_ - xhat) * not_Xmiss
            Y_ = (Y_ - yhat) * not_Ymiss
            if a==0:
                r2X   = 1-np.sum(X_**2)/TSSX
                r2Xpv = 1-np.sum(X_**2,axis=0)/TSSXpv
                r2Xpv = r2Xpv.reshape(-1,1)
                r2Y   = 1-np.sum(Y_**2)/TSSY
                r2Ypv = 1-np.sum(Y_**2,axis=0)/TSSYpv
                r2Ypv = r2Ypv.reshape(-1,1)
                
            else:
                r2X   = np.hstack((r2X,1-np.sum(X_**2)/TSSX))
                aux_  = 1-np.sum(X_**2,axis=0)/TSSXpv
                aux_  = aux_.reshape(-1,1)
                r2Xpv = np.hstack((r2Xpv,aux_))
                
                r2Y   = np.hstack((r2Y,1-np.sum(Y_**2)/TSSY))
                aux_  = 1-np.sum(Y_**2,axis=0)/TSSYpv
                aux_  = aux_.reshape(-1,1)
                r2Ypv = np.hstack((r2Ypv,aux_))               
            X_ = z2n(X_,Xnanmap)
            Y_ = z2n(Y_,Ynanmap)
            
        # Fit full model
        plsobj = pls_(X,Y,A,mcsX=mcsX,mcsY=mcsY,shush=True)
        for a in list(range(A-1,0,-1)):
             r2X[a]     = r2X[a]-r2X[a-1]
             r2Xpv[:,a] = r2Xpv[:,a]-r2Xpv[:,a-1]
             if cross_val_X:
                 q2X[a]     = q2X[a]-q2X[a-1]
                 q2Xpv[:,a] = q2Xpv[:,a]-q2Xpv[:,a-1]
             else:
                 q2X   = False
                 q2Xpv = False
             
             r2Y[a]     = r2Y[a]-r2Y[a-1]
             r2Ypv[:,a] = r2Ypv[:,a]-r2Ypv[:,a-1]
             q2Y[a]     = q2Y[a]-q2Y[a-1]
             q2Ypv[:,a] = q2Ypv[:,a]-q2Ypv[:,a-1]
             
        r2xc = np.cumsum(r2X)
        r2yc = np.cumsum(r2Y)
        if cross_val_X:
            q2xc = np.cumsum(q2X)
        else:
            q2xc = False
        q2yc = np.cumsum(q2Y)    
        eigs = np.var(plsobj['T'],axis=0)
        
        plsobj['q2Y']   = q2Y
        plsobj['q2Ypv'] = q2Ypv
        plsobj['type']  = 'pls'
        if cross_val_X:
            plsobj['q2X']   = q2X
            plsobj['q2Xpv'] = q2Xpv
        
        if not(shush):
            if not(cross_val_X):
                print('---------------------------------------------------------------------------------')
                print('PC #       Eig      R2X     sum(R2X)      R2Y     sum(R2Y)      Q2Y     sum(Q2Y)')
                if A>1:
                    for a in list(range(A)):
                        print("PC #"+str(a+1)+":{:8.3f}    {:.3f}     {:.3f}       {:.3f}     {:.3f}       {:.3f}     {:.3f}".format(eigs[a], r2X[a], r2xc[a], r2Y[a], r2yc[a],q2Y[a],q2yc[a]))
                else:
                    d1=eigs[0]
                    d2=r2xc[0]
                    d3=r2yc[0]
                    d4=q2yc[0]
                    print("PC #"+str(a+1)+":{:8.3f}    {:.3f}     {:.3f}       {:.3f}     {:.3f}       {:.3f}     {:.3f}".format(d1, r2X, d2,r2Y,d3,q2Y,d4))
                print('---------------------------------------------------------------------------------')     
            else:
                print('-------------------------------------------------------------------------------------------------------')
                print('PC #       Eig      R2X     sum(R2X)      Q2X     sum(Q2X)      R2Y     sum(R2Y)      Q2Y     sum(Q2Y)')
                if A>1:
                    for a in list(range(A)):
                        print("PC #"+str(a+1)+":{:8.3f}    {:.3f}     {:.3f}       {:.3f}     {:.3f}       {:.3f}     {:.3f}       {:.3f}     {:.3f}".format(eigs[a], r2X[a], r2xc[a],q2X[a],q2xc[a], r2Y[a], r2yc[a],q2Y[a],q2yc[a]))
                else:
                    d1=eigs[0]
                    d2=r2xc[0]
                    d3=q2xc[0]
                    d4=r2yc[0]
                    d5=q2yc[0]
                    print("PC #"+str(a+1)+":{:8.3f}    {:.3f}     {:.3f}       {:.3f}     {:.3f}       {:.3f}     {:.3f}       {:.3f}     {:.3f}".format(d1, r2X, d2,q2X,d3,r2Y,d4,q2Y,d5))
                print('-------------------------------------------------------------------------------------------------------')   
    
    else:
        plsobj='Cannot cross validate  with those options'
    return plsobj    
        
def pls_(X,Y,A,*,mcsX=True,mcsY=True,md_algorithm='nipals',force_nipals=False,shush=False):
    if isinstance(X,np.ndarray):
        X_ = X.copy()
        obsidX = False
        varidX = False
    elif isinstance(X,pd.DataFrame):
        X_=np.array(X.values[:,1:]).astype(float)
        obsidX = X.values[:,0].astype(str)
        obsidX = obsidX.tolist()
        varidX = X.columns.values
        varidX = varidX[1:]
        varidX = varidX.tolist()
        
    if isinstance(Y,np.ndarray):
        Y_=Y.copy()
        obsidY = False
        varidY = False
    elif isinstance(Y,pd.DataFrame):
        Y_=np.array(Y.values[:,1:]).astype(float)
        obsidY = Y.values[:,0].astype(str)
        obsidY = obsidY.tolist()
        varidY = Y.columns.values
        varidY = varidY[1:]
        varidY = varidY.tolist()        
        
   
    if isinstance(mcsX,bool):
        if mcsX:
            #Mean center and autoscale  
            X_,x_mean,x_std = meancenterscale(X_)
        else:    
            x_mean = np.zeros((1,X_.shape[1]))
            x_std  = np.ones((1,X_.shape[1]))
    elif mcsX=='center':
        X_,x_mean,x_std = meancenterscale(X_,mcs='center')
        #only center      
    elif mcsX=='autoscale':
        #only autoscale
        X_,x_mean,x_std = meancenterscale(X_,mcs='autoscale')
        
    if isinstance(mcsY,bool):
        if mcsY:
            #Mean center and autoscale  
            Y_,y_mean,y_std = meancenterscale(Y_)
        else:    
            y_mean = np.zeros((1,Y_.shape[1]))
            y_std  = np.ones((1,Y_.shape[1]))
    elif mcsY=='center':
        Y_,y_mean,y_std = meancenterscale(Y_,mcs='center')
        #only center      
    elif mcsY=='autoscale':
        #only autoscale
        Y_,y_mean,y_std = meancenterscale(Y_,mcs='autoscale')    
        
    #Generate Missing Data Map    
    X_nan_map = np.isnan(X_)
    not_Xmiss = (np.logical_not(X_nan_map))*1
    Y_nan_map = np.isnan(Y_)
    not_Ymiss = (np.logical_not(Y_nan_map))*1
    
    if (not(X_nan_map.any()) and not(Y_nan_map.any())) and not(force_nipals):
        #no missing elements
        if not(shush):
            print('phi.pls using SVD executed on: '+ str(datetime.datetime.now()) )
        TSSX   = np.sum(X_**2)
        TSSXpv = np.sum(X_**2,axis=0)
        TSSY   = np.sum(Y_**2)
        TSSYpv = np.sum(Y_**2,axis=0)
        
        for a in list(range(A)):
            [U_,S,Wh]   = np.linalg.svd((X_.T @ Y_) @ (Y_.T @ X_))
            w          = Wh.T
            w          = w[:,[0]]
            t          = X_ @ w
            q          = Y_.T @ t / (t.T @ t)
            u          = Y_ @ q /(q.T @ q)
            p          = X_.T  @ t / (t.T @ t)
            
            X_ = X_- t @ p.T
            Y_ = Y_- t @ q.T
            
            if a==0:
                W     = w.reshape(-1,1)
                T     = t.reshape(-1,1)
                Q     = q.reshape(-1,1)
                U     = u.reshape(-1,1)
                P     = p.reshape(-1,1)
                
                r2X   = 1-np.sum(X_**2)/TSSX
                r2Xpv = 1-np.sum(X_**2,axis=0)/TSSXpv
                r2Xpv = r2Xpv.reshape(-1,1)
                
                r2Y   = 1-np.sum(Y_**2)/TSSY
                r2Ypv = 1-np.sum(Y_**2,axis=0)/TSSYpv
                r2Ypv = r2Ypv.reshape(-1,1)
            else:
                W     = np.hstack((W,w.reshape(-1,1)))
                T     = np.hstack((T,t.reshape(-1,1)))
                Q     = np.hstack((Q,q.reshape(-1,1)))
                U     = np.hstack((U,u.reshape(-1,1)))
                P     = np.hstack((P,p.reshape(-1,1)))
                
                r2X_   = 1-np.sum(X_**2)/TSSX
                r2Xpv_ = 1-np.sum(X_**2,axis=0)/TSSXpv
                r2Xpv_ = r2Xpv_.reshape(-1,1)
                r2X    = np.hstack((r2X,r2X_))
                r2Xpv  = np.hstack((r2Xpv,r2Xpv_))
                
                r2Y_   = 1-np.sum(Y_**2)/TSSY
                r2Ypv_ = 1-np.sum(Y_**2,axis=0)/TSSYpv
                r2Ypv_ = r2Ypv_.reshape(-1,1)
                r2Y    = np.hstack((r2Y,r2Y_))
                r2Ypv  = np.hstack((r2Ypv,r2Ypv_))
        for a in list(range(A-1,0,-1)):
            r2X[a]     = r2X[a]-r2X[a-1]
            r2Xpv[:,a] = r2Xpv[:,a]-r2Xpv[:,a-1]
            r2Y[a]     = r2Y[a]-r2Y[a-1]
            r2Ypv[:,a] = r2Ypv[:,a]-r2Ypv[:,a-1]
        Ws=W @ np.linalg.pinv(P.T @ W)
        #Adjustment
        Ws[:,0]=W[:,0]
        eigs = np.var(T,axis=0);
        r2xc = np.cumsum(r2X)
        r2yc = np.cumsum(r2Y)
        if not(shush):
            print('--------------------------------------------------------------')
            print('LV #     Eig       R2X       sum(R2X)   R2Y       sum(R2Y)')
            if A>1:    
                for a in list(range(A)):
                    print("LV #"+str(a+1)+":   {:6.3f}    {:.3f}     {:.3f}      {:.3f}     {:.3f}".format(eigs[a], r2X[a], r2xc[a],r2Y[a],r2yc[a]))
            else:
                d1=eigs[0]
                d2=r2xc[0]
                d3=r2yc[0]
                print("LV #"+str(a+1)+":   {:6.3f}    {:.3f}     {:.3f}      {:.3f}     {:.3f}".format(d1, r2X, d2,r2Y,d3))
            print('--------------------------------------------------------------')   
        
        pls_obj={'T':T,'P':P,'Q':Q,'W':W,'Ws':Ws,'U':U,'r2x':r2X,'r2xpv':r2Xpv,'mx':x_mean,'sx':x_std,'r2y':r2Y,'r2ypv':r2Ypv,'my':y_mean,'sy':y_std}  
        if not isinstance(obsidX,bool):
            pls_obj['obsidX']=obsidX
            pls_obj['varidX']=varidX
        if not isinstance(obsidY,bool):
            pls_obj['obsidY']=obsidY
            pls_obj['varidY']=varidY
            
        T2 = hott2(pls_obj,Tnew=T)
        n  = T.shape[0]
        T2_lim99 = (((n-1)*(n+1)*A)/(n*(n-A)))*f99(A,(n-A))
        T2_lim95 = (((n-1)*(n+1)*A)/(n*(n-A)))*f95(A,(n-A))     
        speX = np.sum(X_**2,axis=1,keepdims=1)
        speX_lim95,speX_lim99 = spe_ci(speX)
        speY = np.sum(Y_**2,axis=1,keepdims=1)
        speY_lim95,speY_lim99 = spe_ci(speY)
        pls_obj['T2']          = T2
        pls_obj['T2_lim99']    = T2_lim99
        pls_obj['T2_lim95']    = T2_lim95
        pls_obj['speX']        = speX
        pls_obj['speX_lim99']  = speX_lim99
        pls_obj['speX_lim95']  = speX_lim95
        pls_obj['speY']        = speY
        pls_obj['speY_lim99']  = speY_lim99
        pls_obj['speY_lim95']  = speY_lim95
        return pls_obj
    else:
        if md_algorithm=='nipals':
             #use nipals
             if not(shush):
                 print('phi.pls using NIPALS executed on: '+ str(datetime.datetime.now()) )
             X_,dummy=n2z(X_)
             Y_,dummy=n2z(Y_)
             epsilon=1E-9
             maxit=2000

             TSSX   = np.sum(X_**2)
             TSSXpv = np.sum(X_**2,axis=0)
             TSSY   = np.sum(Y_**2)
             TSSYpv = np.sum(Y_**2,axis=0)
             
             #T=[];
             #P=[];
             #r2=[];
             #r2pv=[];
             #numIT=[];
             for a in list(range(A)):
                 # Select column with largest variance in Y as initial guess
                 ui = Y_[:,[np.argmax(std(Y_))]]
                 Converged=False
                 num_it=0
                 while Converged==False:
                      # %Step 1. w=X'u/u'u
                      uimat=np.tile(ui,(1,X_.shape[1]))
                      wi=(np.sum(X_*uimat,axis=0))/(np.sum((uimat*not_Xmiss)**2,axis=0))
                      #Step 2. Normalize w to unit length.
                      wi=wi/np.linalg.norm(wi)
                      #Step 3. ti= (Xw)/(w'w);
                      wimat=np.tile(wi,(X_.shape[0],1))
                      ti= X_ @ wi.T
                      wtw=np.sum((wimat*not_Xmiss)**2,axis=1)
                      ti=ti/wtw
                      ti=ti.reshape(-1,1)
                      wi=wi.reshape(-1,1)
                      #Step 4 q=Y't/t't
                      timat=np.tile(ti,(1,Y_.shape[1]))
                      qi=(np.sum(Y_*timat,axis=0))/(np.sum((timat*not_Ymiss)**2,axis=0))
                      #Step 5 un=(Yq)/(q'q)
                      qimat=np.tile(qi,(Y_.shape[0],1))
                      qi=qi.reshape(-1,1)
                      un= Y_ @ qi
                      qtq=np.sum((qimat*not_Ymiss)**2,axis=1)
                      qtq=qtq.reshape(-1,1)
                      un=un/qtq
                      un=un.reshape(-1,1)
                      
                      if abs((np.linalg.norm(ui)-np.linalg.norm(un)))/(np.linalg.norm(ui)) < epsilon:
                          Converged=True
                      if num_it > maxit:
                          Converged=True
                      if Converged:
                          if np.var(ti[ti<0]) > np.var(ti[ti>=0]):
                             ti=-ti
                             wi=-wi
                             un=-un
                             qi=-qi
                          if not(shush):
                              print('# Iterations for LV #'+str(a+1)+': ',str(num_it))
                          # Calculate P's for deflation p=Xt/(t't)      
                          timat=np.tile(ti,(1,X_.shape[1]))
                          pi=(np.sum(X_*timat,axis=0))/(np.sum((timat*not_Xmiss)**2,axis=0))
                          pi=pi.reshape(-1,1)
                          # Deflate X leaving missing as zeros (important!)
                          X_=(X_- ti @ pi.T)*not_Xmiss
                          Y_=(Y_- ti @ qi.T)*not_Ymiss
                          
                          if a==0:
                              T=ti
                              P=pi
                              W=wi
                              U=un
                              Q=qi
                              r2X   = 1-np.sum(X_**2)/TSSX
                              r2Xpv = 1-np.sum(X_**2,axis=0)/TSSXpv
                              r2Xpv = r2Xpv.reshape(-1,1)
                              r2Y   = 1-np.sum(Y_**2)/TSSY
                              r2Ypv = 1-np.sum(Y_**2,axis=0)/TSSYpv
                              r2Ypv = r2Ypv.reshape(-1,1)
                          else:
                              T=np.hstack((T,ti.reshape(-1,1)))
                              U=np.hstack((U,un.reshape(-1,1)))
                              P=np.hstack((P,pi))   
                              W=np.hstack((W,wi))
                              Q=np.hstack((Q,qi))
                                             
                              r2X_   = 1-np.sum(X_**2)/TSSX
                              r2Xpv_ = 1-np.sum(X_**2,axis=0)/TSSXpv
                              r2Xpv_ = r2Xpv_.reshape(-1,1)
                              r2X    = np.hstack((r2X,r2X_))
                              r2Xpv  = np.hstack((r2Xpv,r2Xpv_))
                
                              r2Y_   = 1-np.sum(Y_**2)/TSSY
                              r2Ypv_ = 1-np.sum(Y_**2,axis=0)/TSSYpv
                              r2Ypv_ = r2Ypv_.reshape(-1,1)
                              r2Y    = np.hstack((r2Y,r2Y_))
                              r2Ypv  = np.hstack((r2Ypv,r2Ypv_))
                      else:
                          num_it = num_it + 1
                          ui = un
                 if a==0:
                     numIT=num_it
                 else:
                     numIT=np.hstack((numIT,num_it))
                     
             for a in list(range(A-1,0,-1)):
                 r2X[a]     = r2X[a]-r2X[a-1]
                 r2Xpv[:,a] = r2Xpv[:,a]-r2Xpv[:,a-1]
                 r2Y[a]     = r2Y[a]-r2Y[a-1]
                 r2Ypv[:,a] = r2Ypv[:,a]-r2Ypv[:,a-1]
             
             Ws=W @ np.linalg.pinv(P.T @ W)
             #Adjustment
             Ws[:,0]=W[:,0]
             eigs = np.var(T,axis=0);
             r2xc = np.cumsum(r2X)
             r2yc = np.cumsum(r2Y)
             if not(shush):
                 print('--------------------------------------------------------------')
                 print('LV #     Eig       R2X       sum(R2X)   R2Y       sum(R2Y)')
                 if A>1:    
                     for a in list(range(A)):
                         print("LV #"+str(a+1)+":   {:6.3f}    {:.3f}     {:.3f}      {:.3f}     {:.3f}".format(eigs[a], r2X[a], r2xc[a],r2Y[a],r2yc[a]))
                 else:
                    d1=eigs[0]
                    d2=r2xc[0]
                    d3=r2yc[0]
                    print("LV #"+str(a+1)+":   {:6.3f}    {:.3f}     {:.3f}      {:.3f}     {:.3f}".format(d1, r2X, d2,r2Y,d3))
                 print('--------------------------------------------------------------')   
                       
             pls_obj={'T':T,'P':P,'Q':Q,'W':W,'Ws':Ws,'U':U,'r2x':r2X,'r2xpv':r2Xpv,'mx':x_mean,'sx':x_std,'r2y':r2Y,'r2ypv':r2Ypv,'my':y_mean,'sy':y_std}  
             if not isinstance(obsidX,bool):
                 pls_obj['obsidX']=obsidX
                 pls_obj['varidX']=varidX
             if not isinstance(obsidY,bool):
                pls_obj['obsidY']=obsidY
                pls_obj['varidY']=varidY    
                
             T2 = hott2(pls_obj,Tnew=T)
             n  = T.shape[0]
             T2_lim99 = (((n-1)*(n+1)*A)/(n*(n-A)))*f99(A,(n-A))
             T2_lim95 = (((n-1)*(n+1)*A)/(n*(n-A)))*f95(A,(n-A))     
             speX = np.sum(X_**2,axis=1,keepdims=1)
             speX_lim95,speX_lim99 = spe_ci(speX)
             speY = np.sum(Y_**2,axis=1,keepdims=1)
             speY_lim95,speY_lim99 = spe_ci(speY)
             pls_obj['T2']          = T2
             pls_obj['T2_lim99']    = T2_lim99
             pls_obj['T2_lim95']    = T2_lim95
             pls_obj['speX']        = speX
             pls_obj['speX_lim99']  = speX_lim99
             pls_obj['speX_lim95']  = speX_lim95
             pls_obj['speY']        = speY
             pls_obj['speY_lim99']  = speY_lim99
             pls_obj['speY_lim95']  = speY_lim95
             return pls_obj   
                         
        elif md_algorithm=='nlp':
            #use NLP per Journal of Chemometrics, 28(7), pp.575-584. and a modification from Sal.
            shush=False         
            if not(shush):
                 print('phi.pls using NLP with Ipopt executed on: '+ str(datetime.datetime.now()) )
            X_,dummy=n2z(X_)
            Y_,dummy=n2z(Y_)
            plsobj_= pls_(X,Y,A,mcsX=mcsX,mcsY=mcsY,md_algorithm='nipals',shush=True)
            plsobj_= prep_pls_4_MDbyNLP(plsobj_,X_,Y_)
              
            TSSX   = np.sum(X_**2)
            TSSXpv = np.sum(X_**2,axis=0)
            TSSY   = np.sum(Y_**2)
            TSSYpv = np.sum(Y_**2,axis=0)
            
            #Set up the model in Pyomo
            model             = ConcreteModel()
            model.A           = Set(initialize = plsobj_['pyo_A'] )
            model.N           = Set(initialize = plsobj_['pyo_N'] )
            model.M           = Set(initialize = plsobj_['pyo_M'])
            model.O           = Set(initialize = plsobj_['pyo_O'] )
            model.P           = Var(model.N,model.A, within = Reals,initialize = plsobj_['pyo_P_init'])
            model.T           = Var(model.O,model.A, within = Reals,initialize = plsobj_['pyo_T_init'])
            model.psi         = Param(model.O,model.N,initialize = plsobj_['pyo_psi'])
            model.X           = Param(model.O,model.N,initialize = plsobj_['pyo_X'])
            model.theta       = Param(model.O,model.M,initialize = plsobj_['pyo_theta'])
            model.Y           = Param(model.O,model.M,initialize = plsobj_['pyo_Y'])           
            model.delta       = Param(model.A, model.A, initialize=lambda model, a1, a2: 1.0 if a1==a2 else 0)
            
            # Constraints 27b and 27c
            def _c27bc_con(model, a1, a2):
                return sum(model.P[j, a1] * model.P[j, a2] for j in model.N) == model.delta[a1, a2]
            model.c27bc = Constraint(model.A, model.A, rule=_c27bc_con)
            
            # Constraints 27d
            def _27d_con(model, a1, a2):
                if a2 < a1:
                    return sum(model.T[o, a1] * model.T[o, a2] for o in model.O) == 0
                else:
                    return Constraint.Skip
            model.c27d = Constraint(model.A, model.A, rule=_27d_con)
            
            # Constraints 27e
            def _27e_con(model,i):
                return sum (model.T[o,i]  for o in model.O )==0
            model.c27e = Constraint(model.A,rule=_27e_con)
            
            def _eq_27a_obj(model):
                return sum(sum(sum( (model.theta[o,m]*model.Y[o,m]) * (model.X[o,n]- model.psi[o,n] * sum(model.T[o,a] * model.P[n,a] for a in model.A)) for o in model.O)**2 for n in model.N) for m in model.M)
            model.obj = Objective(rule=_eq_27a_obj)
            
            # Setup our solver as either local ipopt, gams:ipopt, or neos ipopt:
            if (ipopt_ok):
                print("Solving NLP using local IPOPT executable")
                solver = SolverFactory('ipopt')
            
                if (ma57_ok):
                    solver.options['linear_solver'] = 'ma57'
            
                results = solver.solve(model,tee=True)
            elif (gams_ok):
                print("Solving NLP using GAMS/IPOPT interface")
                # 'just 'ipopt' could work, if no binary in path
                solver = SolverFactory('gams:ipopt')
            
                # It doesn't seem to notice the opt file when I write it
                results = solver.solve(model, tee=True)
            else:
                print("Solving NLP using IPOPT on remote NEOS server")
                solver_manager = SolverManagerFactory('neos')
                results = solver_manager.solve(model, opt='ipopt', tee=True)
            
            T=[]
            for o in model.O:
                 t=[]
                 for a in model.A:
                    t.append(value(model.T[o,a]))
                 T.append(t)   
            T=np.array(T)     
            P=[]
            for n in model.N:
                 p=[]
                 for a in model.A:
                    p.append(value(model.P[n,a]))
                 P.append(p)   
            P=np.array(P)   
            
           # Obtain a Ws with NLP
           #Set up the model in Pyomo
            Taux               = np2D2pyomo(T)
            modelb             = ConcreteModel()
            modelb.A           = Set(initialize = plsobj_['pyo_A'] )
            modelb.N           = Set(initialize = plsobj_['pyo_N'] )
            modelb.O           = Set(initialize = plsobj_['pyo_O'] )
            modelb.Ws          = Var(model.N,model.A, within = Reals,initialize = plsobj_['pyo_Ws_init'])
            modelb.T           = Param(model.O,model.A, within = Reals,initialize = Taux)
            modelb.psi         = Param(model.O,model.N,initialize = plsobj_['pyo_psi'])
            modelb.X           = Param(model.O,model.N,initialize = plsobj_['pyo_X'])
       
            def _eq_obj(model):
                return sum(sum( (model.T[o,a] - sum(model.psi[o,n] * model.X[o,n] * model.Ws[n,a] for n in model.N))**2  for a in model.A)  for o in model.O)
            modelb.obj = Objective(rule=_eq_obj)
            # Setup our solver as either local ipopt, gams:ipopt, or neos ipopt:
            if (ipopt_ok):
                print("Solving NLP using local IPOPT executable")
                solver = SolverFactory('ipopt')
            
                if (ma57_ok):
                    solver.options['linear_solver'] = 'ma57'
            
                results = solver.solve(modelb,tee=True)
            elif (gams_ok):
                print("Solving NLP using GAMS/IPOPT interface")
                # 'just 'ipopt' could work, if no binary in path
                solver = SolverFactory('gams:ipopt')
            
                # It doesn't seem to notice the opt file when I write it
                results = solver.solve(modelb, tee=True)
            else:
                print("Solving NLP using IPOPT on remote NEOS server")
                solver_manager = SolverManagerFactory('neos')
                results = solver_manager.solve(modelb, opt='ipopt', tee=True)

            Ws=[]
            for n in modelb.N:
                 ws=[]
                 for a in modelb.A:
                    ws.append(value(modelb.Ws[n,a]))
                 Ws.append(ws)   
            Ws=np.array(Ws)               
            
            
            Xhat              = T @ P.T
            Xaux              = X_.copy()
            Xaux[X_nan_map]   = Xhat[X_nan_map]
            Xaux              = np2D2pyomo(Xaux)
            Taux              = np2D2pyomo(T)
            
            #Set up the model in Pyomo
            model2             = ConcreteModel()
            model2.A           = Set(initialize = plsobj_['pyo_A'] )
            model2.N           = Set(initialize = plsobj_['pyo_N'] )
            model2.M           = Set(initialize = plsobj_['pyo_M'])
            model2.O           = Set(initialize = plsobj_['pyo_O'] )
            
            model2.T           = Param(model.O,model.A, within = Reals,initialize = Taux)
            model2.Q           = Var(model.M,model.A, within = Reals,initialize = plsobj_['pyo_Q_init'])
            model2.X           = Param(model.O,model.N,initialize = plsobj_['pyo_X'])
            model2.theta       = Param(model.O,model.M,initialize = plsobj_['pyo_theta'])
            model2.Y           = Param(model.O,model.M,initialize = plsobj_['pyo_Y'])           
            model2.delta       = Param(model.A, model.A, initialize=lambda model, a1, a2: 1.0 if a1==a2 else 0)
            
            
            def _eq_36a_mod_obj(model):
                return sum(sum(sum( (model.X[o,n]) * (model.Y[o,m]- model.theta[o,m] * sum(model.T[o,a] * model.Q[m,a] for a in model.A)) for o in model.O)**2 for n in model.N) for m in model.M)
            model2.obj = Objective(rule=_eq_36a_mod_obj)
            
            # Setup our solver as either local ipopt, gams:ipopt, or neos ipopt:
            if (ipopt_ok):
                print("Solving NLP using local IPOPT executable")
                solver = SolverFactory('ipopt')
            
                if (ma57_ok):
                    solver.options['linear_solver'] = 'ma57'
            
                results = solver.solve(model2,tee=True)
            elif (gams_ok):
                print("Solving NLP using GAMS/IPOPT interface")
                # 'just 'ipopt' could work, if no binary in path
                solver = SolverFactory('gams:ipopt')
            
                # It doesn't seem to notice the opt file when I write it
                results = solver.solve(model2, tee=True)
            else:
                print("Solving NLP using IPOPT on remote NEOS server")
                solver_manager = SolverManagerFactory('neos')
                results = solver_manager.solve(model2, opt='ipopt', tee=True)
            
               
            Q=[]
            for m in model2.M:
                 q=[]
                 for a in model2.A:
                    q.append(value(model2.Q[m,a]))
                 Q.append(q)   
            Q=np.array(Q)  
            
         
            # Calculate R2
               
            for a in list(range(0, A)):
                 ti=T[:,[a]]
                 pi=P[:,[a]]
                 qi=Q[:,[a]]
                 X_=(X_- ti @ pi.T)*not_Xmiss
                 Y_=(Y_- ti @ qi.T)*not_Ymiss
                 if a==0:        
                    r2X   = 1-np.sum(X_**2)/TSSX
                    r2Xpv = 1-np.sum(X_**2,axis=0)/TSSXpv
                    r2Xpv = r2Xpv.reshape(-1,1)        
                    r2Y   = 1-np.sum(Y_**2)/TSSY
                    r2Ypv = 1-np.sum(Y_**2,axis=0)/TSSYpv
                    r2Ypv = r2Ypv.reshape(-1,1)
                 else:        
                    r2X_   = 1-np.sum(X_**2)/TSSX
                    r2Xpv_ = 1-np.sum(X_**2,axis=0)/TSSXpv
                    r2Xpv_ = r2Xpv_.reshape(-1,1)
                    r2X    = np.hstack((r2X,r2X_))
                    r2Xpv  = np.hstack((r2Xpv,r2Xpv_))
                    r2Y_   = 1-np.sum(Y_**2)/TSSY
                    r2Ypv_ = 1-np.sum(Y_**2,axis=0)/TSSYpv
                    r2Ypv_ = r2Ypv_.reshape(-1,1)
                    r2Y    = np.hstack((r2Y,r2Y_))
                    r2Ypv  = np.hstack((r2Ypv,r2Ypv_))
                            
                    
            for a in list(range(A-1,0,-1)):
                r2X[a]     = r2X[a]-r2X[a-1]
                r2Xpv[:,a] = r2Xpv[:,a]-r2Xpv[:,a-1]
                r2Y[a]     = r2Y[a]-r2Y[a-1]
                r2Ypv[:,a] = r2Ypv[:,a]-r2Ypv[:,a-1]
                 
            #Ws=W @ np.linalg.pinv(P.T @ W)
            #Ws=P
            eigs = np.var(T,axis=0);
            r2xc = np.cumsum(r2X)
            r2yc = np.cumsum(r2Y)
            if not(shush):
                print('--------------------------------------------------------------')
                print('LV #     Eig       R2X       sum(R2X)   R2Y       sum(R2Y)')
                if A>1:    
                    for a in list(range(A)):
                        print("LV #"+str(a+1)+":   {:6.3f}    {:.3f}     {:.3f}      {:.3f}     {:.3f}".format(eigs[a], r2X[a], r2xc[a],r2Y[a],r2yc[a]))
                else:
                    d1=eigs[0]
                    d2=r2xc[0]
                    d3=r2yc[0]
                    print("LV #"+str(a+1)+":   {:6.3f}    {:.3f}     {:.3f}      {:.3f}     {:.3f}".format(d1, r2X, d2,r2Y,d3))
                print('--------------------------------------------------------------')   
            W=1
            U=1
            pls_obj={'T':T,'P':P,'Q':Q,'W':W,'Ws':Ws,'U':U,'r2x':r2X,'r2xpv':r2Xpv,'mx':x_mean,'sx':x_std,'r2y':r2Y,'r2ypv':r2Ypv,'my':y_mean,'sy':y_std}  
            if not isinstance(obsidX,bool):
                pls_obj['obsidX']=obsidX
                pls_obj['varidX']=varidX
            if not isinstance(obsidY,bool):
                pls_obj['obsidY']=obsidY
                pls_obj['varidY']=varidY
                
            T2 = hott2(pls_obj,Tnew=T)
            n  = T.shape[0]
            T2_lim99 = (((n-1)*(n+1)*A)/(n*(n-A)))*f99(A,(n-A))
            T2_lim95 = (((n-1)*(n+1)*A)/(n*(n-A)))*f95(A,(n-A))     
            speX = np.sum(X_**2,axis=1,keepdims=1)
            speX_lim95,speX_lim99 = spe_ci(speX)
            speY = np.sum(Y_**2,axis=1,keepdims=1)
            speY_lim95,speY_lim99 = spe_ci(speY)
            pls_obj['T2']          = T2
            pls_obj['T2_lim99']    = T2_lim99
            pls_obj['T2_lim95']    = T2_lim95
            pls_obj['speX']        = speX
            pls_obj['speX_lim99']  = speX_lim99
            pls_obj['speX_lim95']  = speX_lim95
            pls_obj['speY']        = speY
            pls_obj['speY_lim99']  = speY_lim99
            pls_obj['speY_lim95']  = speY_lim95
            return pls_obj 

def lwpls(xnew,loc_par,mvmobj,X,Y,*,shush=False):
    """
    LWPLS algorithm in: International Journal of Pharmaceutics 421 (2011) 269– 274
    
    Implemented by Salvador Garcia-Munoz
    (sgarciam@ic.ac.uk ,salvadorgarciamunoz@gmail.com)
    
    xnew   : Regressor vector to make prediction
    loc_par: Localization parameter
    mvmobj : PLS model between X and Y 
    X,Y    : Training set for mvmobj (PLS model) numpy arrays or pandas dataframes
    shush  : ='True'  will silent outpuit
              'False' will display outpuit  *default if not sent*
    
    """
    if not(shush):
        print('phi.lwpls executed on: '+ str(datetime.datetime.now()) )
    xnew=np.reshape(xnew,(-1,1))     
    
    if isinstance(X,pd.DataFrame):
        X=np.array(X.values[:,1:]).astype(float)

    if isinstance(Y,pd.DataFrame):
        Y=np.array(Y.values[:,1:]).astype(float)
        
    vip=np.sum(np.abs(mvmobj['Ws'] * np.tile(mvmobj['r2y'],(mvmobj['Ws'].shape[0],1)) ),axis=1)
    vip=np.reshape(vip,(len(vip),-1))
    theta=vip; #Using element wise operations for speed, no need for matrix notation

    D     = X - np.tile(xnew.T,(X.shape[0],1))
    d2    = D * np.tile(theta.T,(X.shape[0],1)) * D
    d2    = np.sqrt(np.sum(d2,axis=1))
    omega = np.exp(-d2/(np.var(d2,ddof=1)*loc_par))
    OMEGA = np.diag(omega)
    omega = np.reshape(omega,(len(omega),-1))
    
    X_weighted_mean = np.sum((np.tile(omega,(1,X.shape[1])) * X),axis=0)/np.sum(omega)
    Y_weighted_mean = np.sum((np.tile(omega,(1,Y.shape[1])) * Y),axis=0)/np.sum(omega)
    
    X_weighted_mean=np.reshape(X_weighted_mean,(len(X_weighted_mean),-1))
    Y_weighted_mean=np.reshape(Y_weighted_mean,(len(Y_weighted_mean),-1))
    
    Xi = X - X_weighted_mean.T
    Yi = Y - Y_weighted_mean.T
    
    xnewi = xnew - X_weighted_mean
    yhat=Y_weighted_mean
    
    for a in list(range(0,mvmobj['T'].shape[1])):
        [U_,S,Wh]=np.linalg.svd(Xi.T @ OMEGA @ Yi @ Yi.T @ OMEGA @ Xi)
        w           = Wh.T
        w           = w[:,[0]]
        t = Xi @ w
        p = Xi.T @ OMEGA @ t / (t.T @ OMEGA @ t)        
        q = Yi.T @ OMEGA @ t / (t.T @ OMEGA @ t)
        
        tnew = xnewi.T @ w
        yhat  = yhat + q @ tnew         
        Xi    = Xi - t @ p.T
        Yi    = Yi - t @ q.T
        xnewi = xnewi - p @ tnew
    return yhat[0].T
    
    
def pca_pred(Xnew,pcaobj,*,algorithm='p2mp'):
    if isinstance(Xnew,np.ndarray):
        X_=Xnew.copy()
        if X_.ndim==1:
            X_=np.reshape(X_,(1,-1))
    elif isinstance(Xnew,pd.DataFrame):
        X_=np.array(Xnew.values[:,1:]).astype(float)

    X_nan_map = np.isnan(X_)    
    if not(X_nan_map.any()):
        X_mcs= X_- np.tile(pcaobj['mx'],(X_.shape[0],1))
        X_mcs= X_mcs/(np.tile(pcaobj['sx'],(X_.shape[0],1)))      
        tnew =  X_mcs @ pcaobj['P']
        xhat = (tnew @ pcaobj['P'].T) * np.tile(pcaobj['sx'],(X_.shape[0],1)) + np.tile(pcaobj['mx'],(X_.shape[0],1))
        var_t = (pcaobj['T'].T @ pcaobj['T'])/pcaobj['T'].shape[0]
        htt2 = np.sum((tnew @ np.linalg.inv(var_t)) * tnew,axis=1)
        spe   = X_mcs -(tnew @ pcaobj['P'].T)
        spe  = np.sum(spe**2,axis=1,keepdims=True) 
        xpred={'Xhat':xhat,'Tnew':tnew, 'speX':spe,'T2':htt2}
    elif algorithm=='p2mp':  # Using Projection to the model plane method for missing data    
        X_nan_map = np.isnan(X_)
        not_Xmiss = (np.logical_not(X_nan_map))*1
        
        Xmcs=((X_-np.tile(pcaobj['mx'],(X_.shape[0],1)))/(np.tile(pcaobj['sx'],(X_.shape[0],1))))
        Xmcs,dummy=n2z(Xmcs)
        for i in list(range(Xmcs.shape[0])):
            row_missing_map=not_Xmiss[[i],:]
            tempP = pcaobj['P'] * np.tile(row_missing_map.T,(1,pcaobj['P'].shape[1]))
            PTP = tempP.T @ tempP  
            try:
                #tnew_ =    np.linalg.inv(PTP) @ tempP.T         @ Xmcs[[i],:].T                #inneficient  
                tnew_,resid,rank,s = np.linalg.lstsq(PTP,(tempP.T  @ Xmcs[[i],:].T),rcond=None) #better 
            except:
                tnew_ = np.linalg.pinv(PTP) @ tempP.T  @ Xmcs[[i],:].T                          #if the sh** hits the fan
            if i==0:
                tnew = tnew_.T
            else:
                tnew = np.vstack((tnew,tnew_.T))
        xhat = (tnew @ pcaobj['P'].T) * np.tile(pcaobj['sx'],(X_.shape[0],1)) + np.tile(pcaobj['mx'],(X_.shape[0],1))        
        var_t = (pcaobj['T'].T @ pcaobj['T'])/pcaobj['T'].shape[0]
        htt2 = np.sum((tnew @ np.linalg.inv(var_t)) * tnew,axis=1)
        spe   = Xmcs -(tnew @ pcaobj['P'].T)
        spe  = spe * not_Xmiss
        spe  = np.sum(spe**2,axis=1,keepdims=True) 
        xpred={'Xhat':xhat,'Tnew':tnew, 'speX':spe,'T2':htt2}
    return xpred

def pls_pred(Xnew,plsobj,*,algorithm='p2mp',force_deflation=False):
    if isinstance(Xnew,np.ndarray):
        X_=Xnew.copy()
        if X_.ndim==1:
            X_=np.reshape(X_,(1,-1))
    elif isinstance(Xnew,pd.DataFrame):
        X_=np.array(Xnew.values[:,1:]).astype(float)
    elif isinstance(Xnew,dict):
        data_=[]
        names_=[]
        for k in Xnew.keys():
            data_.append(Xnew[k])
            names_.append(k)
        XMB={'data':data_,'blknames':names_}

        c=0
        for i,x in enumerate(XMB['data']):        
            x_=x.values[:,1:].astype(float)                     
            if c==0:
                X_=x_.copy() 
            else:    
                X_=np.hstack((X_,x_))
            c=c+1        
       
    X_nan_map = np.isnan(X_)    
    if not(X_nan_map.any()) and not(force_deflation):
        tnew = ((X_-np.tile(plsobj['mx'],(X_.shape[0],1)))/(np.tile(plsobj['sx'],(X_.shape[0],1)))) @ plsobj['Ws']
        yhat = (tnew @ plsobj['Q'].T) * np.tile(plsobj['sy'],(X_.shape[0],1)) + np.tile(plsobj['my'],(X_.shape[0],1))
        xhat = (tnew @ plsobj['P'].T) * np.tile(plsobj['sx'],(X_.shape[0],1)) + np.tile(plsobj['mx'],(X_.shape[0],1))
        var_t = (plsobj['T'].T @ plsobj['T'])/plsobj['T'].shape[0]
        htt2 = np.sum((tnew @ np.linalg.inv(var_t)) * tnew,axis=1)
        speX  = ((X_-np.tile(plsobj['mx'],(X_.shape[0],1)))/(np.tile(plsobj['sx'],(X_.shape[0],1))))-(tnew @ plsobj['P'].T)
        speX  = np.sum(speX**2,axis=1,keepdims=True) 
        ypred ={'Yhat':yhat,'Xhat':xhat,'Tnew':tnew,'speX':speX,'T2':htt2}
    elif algorithm=='p2mp':
        X_nan_map = np.isnan(X_)
        not_Xmiss = (np.logical_not(X_nan_map))*1
        Xmcs=((X_-np.tile(plsobj['mx'],(X_.shape[0],1)))/(np.tile(plsobj['sx'],(X_.shape[0],1))))
        Xmcs,dummy=n2z(Xmcs)        
        for i in list(range(Xmcs.shape[0])):
            row_missing_map=not_Xmiss[[i],:]
            tempW = plsobj['W'] * np.tile(row_missing_map.T,(1,plsobj['W'].shape[1]))
            
            for a in list(range(plsobj['W'].shape[1])):
                WTW    = tempW[:,[a]].T @ tempW[:,[a]]
                #tnew_aux = np.linalg.inv(WTW) @ tempW[:,[a]].T  @ Xmcs[[i],:].T
                tnew_aux,resid,rank,s = np.linalg.lstsq(WTW,(tempW[:,[a]].T  @ Xmcs[[i],:].T),rcond=None)
                Xmcs[[i],:] = (Xmcs[[i],:] - tnew_aux @ plsobj['P'][:,[a]].T) * row_missing_map
                if a==0:
                    tnew_=tnew_aux
                else:
                    tnew_=np.vstack((tnew_,tnew_aux))
                    
            if i==0:
                tnew = tnew_.T
            else:
                tnew = np.vstack((tnew,tnew_.T))
        yhat = (tnew @ plsobj['Q'].T) * np.tile(plsobj['sy'],(X_.shape[0],1)) + np.tile(plsobj['my'],(X_.shape[0],1))
        xhat = (tnew @ plsobj['P'].T) * np.tile(plsobj['sx'],(X_.shape[0],1)) + np.tile(plsobj['mx'],(X_.shape[0],1))
        var_t = (plsobj['T'].T @ plsobj['T'])/plsobj['T'].shape[0]
        htt2 = np.sum((tnew @ np.linalg.inv(var_t)) * tnew,axis=1)
        X_,dummy=n2z(X_)
        speX  = ((X_-np.tile(plsobj['mx'],(X_.shape[0],1)))/(np.tile(plsobj['sx'],(X_.shape[0],1))))-(tnew @ plsobj['P'].T)
        speX  = speX*not_Xmiss
        speX  = np.sum(speX**2,axis=1,keepdims=True) 
        ypred ={'Yhat':yhat,'Xhat':xhat,'Tnew':tnew,'speX':speX,'T2':htt2}     
    return ypred

def hott2(mvmobj,*,Xnew=False,Tnew=False):
    if isinstance(Xnew,bool) and not(isinstance(Tnew,bool)):
        var_t = (mvmobj['T'].T @ mvmobj['T'])/mvmobj['T'].shape[0]
        hott2_ = np.sum((Tnew @ np.linalg.inv(var_t)) * Tnew,axis=1)
    elif isinstance(Tnew,bool) and not(isinstance(Xnew,bool)):
        if 'Q' in mvmobj:  
            xpred = pls_pred(Xnew,mvmobj)  
        else:
            xpred = pca_pred(Xnew,mvmobj)  
        Tnew=xpred['Tnew']    
        var_t = (mvmobj['T'].T @ mvmobj['T'])/mvmobj['T'].shape[0]
        hott2_ = np.sum((Tnew @ np.linalg.inv(var_t)) * Tnew,axis=1)
    elif isinstance(Xnew,bool) and isinstance(Tnew,bool):
        var_t = (mvmobj['T'].T @ mvmobj['T'])/mvmobj['T'].shape[0]
        Tnew =mvmobj['T']
        hott2_ = np.sum((Tnew @ np.linalg.inv(var_t)) * Tnew,axis=1)
    return hott2_

def spe(mvmobj,Xnew,*,Ynew=False):
        if 'Q' in mvmobj:  
            xpred = pls_pred(Xnew,mvmobj)  
        else:
            xpred = pca_pred(Xnew,mvmobj)  
        Tnew=xpred['Tnew']
        if isinstance(Xnew,np.ndarray):
            X_=Xnew.copy()
        elif isinstance(Xnew,pd.DataFrame):
            X_=np.array(Xnew.values[:,1:]).astype(float)
        if isinstance(Ynew,np.ndarray):
            Y_=Ynew.copy()
        elif isinstance(Ynew,pd.DataFrame):
            Y_=np.array(Ynew.values[:,1:]).astype(float)
        
        Xnewhat= Tnew @ mvmobj['P'].T
        Xres = X_ - np.tile(mvmobj['mx'],(Xnew.shape[0],1))
        Xres = Xres / np.tile(mvmobj['sx'],(Xnew.shape[0],1))
        Xres = Xres - Xnewhat
        spex_ =  np.sum(Xres**2,axis=1,keepdims=True)
        
        if not(isinstance(Ynew,np.bool)) and ('Q' in mvmobj):
            Ynewhat= Tnew @ mvmobj['Q'].T
            Yres = Y_   - np.tile(mvmobj['my'],(Ynew.shape[0],1))
            Yres = Yres / np.tile(mvmobj['sy'],(Ynew.shape[0],1))
            Yres = Yres - Ynewhat
            spey_ =  np.sum(Yres**2,axis=1,keepdims=True)
            return spex_,spey_
        else:      
            return spex_


def z2n(X,X_nan_map):
    X[X_nan_map==1] = np.nan
    return X

def n2z(X):
    X_nan_map = np.isnan(X)            
    if X_nan_map.any():
        X_nan_map       = X_nan_map*1
        X[X_nan_map==1] = 0
    else:
        X_nan_map       = X_nan_map*1
    return X,X_nan_map

def mean(X):
    X_nan_map = np.isnan(X)
    X_ = X.copy()
    if X_nan_map.any():
        X_nan_map       = X_nan_map*1
        X_[X_nan_map==1] = 0
        aux             = np.sum(X_nan_map,axis=0)
        #Calculate mean without accounting for NaN'
        x_mean = np.sum(X_,axis=0,keepdims=1)/(np.ones((1,X_.shape[1]))*X_.shape[0]-aux)
    else:
        x_mean = np.mean(X_,axis=0,keepdims=1)
    return x_mean

def std(X):
    x_mean=mean(X)
    x_mean=np.tile(x_mean,(X.shape[0],1))
    X_nan_map = np.isnan(X)
    if X_nan_map.any():
        X_nan_map             = X_nan_map*1
        X_                    = X.copy()
        X_[X_nan_map==1]      = 0
        aux_mat               = (X_-x_mean)**2
        aux_mat[X_nan_map==1] = 0
        aux                   = np.sum(X_nan_map,axis=0)
        #Calculate mean without accounting for NaN's
        x_std = np.sqrt((np.sum(aux_mat,axis=0,keepdims=1))/(np.ones((1,X_.shape[1]))*(X_.shape[0]-1-aux)))
    else:
        x_std = np.sqrt(np.sum((X-x_mean)**2,axis=0,keepdims=1)/(np.ones((1,X.shape[1]))*(X.shape[0]-1)))
    return x_std
   
def meancenterscale(X,*,mcs=True):
    '''
    Inputs:
        X: Matrix to be meancenterd ONLY works with Numpy matrices
        mcs = True | center | autoscale        
    '''
        
    if isinstance(mcs,bool):
        if mcs:
            x_mean = mean(X)
            x_std  = std(X)
            X      = X-np.tile(x_mean,(X.shape[0],1))
            X      = X/np.tile(x_std,(X.shape[0],1))
        else:
            x_mean = np.nan
            x_std  = np.nan
    elif mcs=='center':
         x_mean = mean(X)
         X      = X-np.tile(x_mean,(X.shape[0],1))
         x_std  = np.ones((1,X.shape[1]))
    elif mcs=='autoscale':
         x_std  = std(X) 
         X      = X/np.tile(x_std,(X.shape[0],1))
         x_mean = np.zeros((1,X.shape[1]))
    else:
        x_mean=np.nan
        x_std=np.nan
    return X,x_mean,x_std

def snv (x):
    """
    Inputs:
        x: Spectra
    Outputs:
        x: Post-processed Spectra
    """
    
    if isinstance(x,pd.DataFrame):
        x_columns=x.columns
        x_values= x.values
        x_values[:,1:] = snv(x_values[:,1:].astype(float))
        xpd=pd.DataFrame(x_values,columns=x_columns)
        return xpd
    else:
        if x.ndim ==2:
            mean_x = np.mean(x,axis=1,keepdims=1)     
            mean_x = np.tile(mean_x,(1,x.shape[1]))
            x      = x - mean_x
            std_x  = np.sum(x**2,axis=1)/(x.shape[1]-1)
            std_x  = np.sqrt(std_x)
            std_x  = np.reshape(std_x,(len(std_x),1))
            std_x =  np.tile(std_x,(1,x.shape[1]))
            x      = x/std_x
            return x
        else:
            x = x - np.mean(x)
            stdx = np.sqrt(np.sum(x**2)/(len(x)-1))
            x = x/stdx
            return x
    
def savgol(ws,od,op,Dm):
    """
    Savitzky-Golay filter for spectra
    inputs:
    ws : Window Size
    od: Order of the derivative
    op: Order of the polynomial
    Dm: Spectra
    
    Outputs:
        Dm_sg, M
        
        Dm_sg: Processed Spectra
        M:     Transformation Matrix for new samples

    """
    if isinstance(Dm,pd.DataFrame):
        x_columns=Dm.columns.tolist()
        FirstElement=[x_columns[0]]
        x_columns=x_columns[1:]
        FirstElement.extend(x_columns[ws:-ws])
        x_values= Dm.values
        Col1= Dm.values[:,0].tolist()
        Col1=np.reshape(Col1,(-1,1))
        aux, M = savgol(ws,od,op,x_values[:,1:].astype(float))
        data_=np.hstack((Col1,aux))
        xpd=pd.DataFrame(data=data_,columns=FirstElement)
        return xpd,M
    else:
        if Dm.ndim==1: 
            l = Dm.shape[0]
        else:
            l = Dm.shape[1]
        
        x_vec=np.arange(-ws,ws+1)
        x_vec=np.reshape(x_vec,(len(x_vec),1))
        X = np.ones((2*ws+1,1))
        for oo in np.arange(1,op+1):
            X=np.hstack((X,x_vec**oo))
        XtXiXt=np.linalg.inv(X.T @ X) @ X.T
        coeffs=XtXiXt[od,:] * factorial(od)
        coeffs=np.reshape(coeffs,(1,len(coeffs)))
        for i in np.arange(1,l-2*ws+1):
            if i==1:
                M=np.hstack((coeffs,np.zeros((1,l-2*ws-1))))
            elif i < l-2*ws:
                m_= np.hstack((np.zeros((1,i-1)), coeffs))
                m_= np.hstack((m_,np.zeros((1,l-2*ws-1-i+1))))
                M = np.vstack((M,m_))
            else:
                m_=np.hstack((np.zeros((1,l-2*ws-1)),coeffs))
                M = np.vstack((M,m_))
        if Dm.ndim==1: 
            Dm_sg=  M @ Dm
        else:
            Dm_sg= Dm @ M.T
        return Dm_sg,M

def np2D2pyomo(arr,*,varids=False):
    if not(varids):
        output=dict(((i+1,j+1), arr[i][j]) for i in range(arr.shape[0]) for j in range(arr.shape[1]))
    else:
        output=dict(((varids[i],j+1), arr[i][j]) for i in range(arr.shape[0]) for j in range(arr.shape[1]))
    return output

def np1D2pyomo(arr,*,indexes=False):
    if arr.ndim==2:
        arr=arr[0]
    if isinstance(indexes,bool):
        output=dict(((j+1), arr[j]) for j in range(len(arr)))
    elif isinstance(indexes,list):
        output=dict((indexes[j], arr[j]) for j in range(len(arr)))
    return output




def adapt_pls_4_pyomo(plsobj,*,use_var_ids=False):
    plsobj_ = plsobj.copy()
    
    A = plsobj['T'].shape[1]
    N = plsobj['P'].shape[0]
    M = plsobj['Q'].shape[0]
    
    
    pyo_A = np.arange(1,A+1)  #index for LV's
    pyo_N = np.arange(1,N+1)  #index for columns of X
    pyo_M = np.arange(1,M+1)  #index for columns of Y
    pyo_A = pyo_A.tolist()
    if not(use_var_ids):
        pyo_N = pyo_N.tolist()
        pyo_M = pyo_M.tolist()    
        pyo_Ws = np2D2pyomo(plsobj['Ws'])
        pyo_Q  = np2D2pyomo(plsobj['Q'])
        pyo_P  = np2D2pyomo(plsobj['P'])    
        var_t = np.var(plsobj['T'],axis=0)    
        pyo_var_t = np1D2pyomo(var_t)
        pyo_mx    = np1D2pyomo(plsobj['mx'])
        pyo_sx    = np1D2pyomo(plsobj['sx'])
        pyo_my    = np1D2pyomo(plsobj['my'])
        pyo_sy    = np1D2pyomo(plsobj['sy'])
    else:    
        pyo_N = plsobj['varidX']
        pyo_M = plsobj['varidY']
        pyo_Ws = np2D2pyomo(plsobj['Ws'],varids=plsobj['varidX'])
        pyo_Q  = np2D2pyomo(plsobj['Q'] ,varids=plsobj['varidY'])
        pyo_P  = np2D2pyomo(plsobj['P'] ,varids=plsobj['varidX'])
        var_t = np.var(plsobj['T'],axis=0)    
        pyo_var_t = np1D2pyomo(var_t)
        pyo_mx    = np1D2pyomo(plsobj['mx'],indexes=plsobj['varidX'])
        pyo_sx    = np1D2pyomo(plsobj['sx'],indexes=plsobj['varidX'])
        pyo_my    = np1D2pyomo(plsobj['my'],indexes=plsobj['varidY'])
        pyo_sy    = np1D2pyomo(plsobj['sy'],indexes=plsobj['varidY'])
            
    plsobj_['pyo_A']      = pyo_A
    plsobj_['pyo_N']      = pyo_N
    plsobj_['pyo_M']      = pyo_M
    plsobj_['pyo_Ws']     = pyo_Ws
    plsobj_['pyo_Q']      = pyo_Q
    plsobj_['pyo_P']      = pyo_P
    plsobj_['pyo_var_t']  = pyo_var_t
    plsobj_['pyo_mx']     = pyo_mx
    plsobj_['pyo_sx']     = pyo_sx
    plsobj_['pyo_my']     = pyo_my
    plsobj_['pyo_sy']     = pyo_sy
    plsobj_['speX_lim95'] =  plsobj['speX_lim95']
    return plsobj_

def spe_ci(spe):    
    chi =np.array(
            [[   0,    1.6900,    4.0500],
          [1.0000,    3.8400,    6.6300],
          [2.0000,    5.9900,    9.2100],
          [3.0000,    7.8100,   11.3400],
          [4.0000,    9.4900,   13.2800],
          [5.0000,   11.0700,   15.0900],
          [6.0000,   12.5900,   16.8100],
          [7.0000,   14.0700,   18.4800],
          [8.0000,   15.5100,   20.0900],
          [9.0000,   16.9200,   21.6700],
          [10.0000,   18.3100,   23.2100],
          [11.0000,   19.6800,   24.7200],
          [12.0000,   21.0300,   26.2200],
          [13.0000,   22.3600,   27.6900],
          [14.0000,   23.6800,   29.1400],
          [15.0000,   25.0000,   30.5800],
          [16.0000,   26.3000,   32.0000],
          [17.0000,   27.5900,   33.4100],
          [18.0000,  28.8700 ,  34.8100],
          [19.0000,   30.1400,   36.1900],
          [20.0000,   31.4100,   37.5700],
          [21.0000,   32.6700,   38.9300],
          [22.0000,   33.9200,   40.2900],
          [23.0000,   35.1700,   41.6400],
          [24.0000,   36.4200,   42.9800],
          [25.0000,   37.6500,   44.3100],
          [26.0000,   38.8900,   45.6400],
          [27.0000,   40.1100,   46.9600],
          [28.0000,   41.3400,   48.2800],
          [29.0000,   42.5600,   49.5900],
          [30.0000,   43.7700,   50.8900],
          [40.0000,   55.7600,   63.6900],
          [50.0000,   67.5000,   76.1500],
          [60.0000,   79.0800,   88.3800],
          [70.0000,   90.5300,  100.4000],
          [80.0000,  101.9000,  112.3000],
          [90.0000,  113.1000,  124.1000],
          [100.0000,  124.3000,  135.8000 ]])

    spem=np.mean(spe)
    spev=np.var(spe,ddof=1)
    g=(spev/(2*spem))
    h=(2*spem**2)/spev

    lim95=np.interp(h,chi[:,0],chi[:,1])
    lim99=np.interp(h,chi[:,0],chi[:,2]);
    lim95= g*lim95
    lim99= g*lim99
    return lim95,lim99

def single_score_conf_int(t):
    '''
    Confidence intervals for a single t-score (used for bar and line plots)
    Input: a column vector of t-scores values
    Outputs: the two limits based on a t-distribution
    '''
    tstud =np.array(
            [[1,       12.706,       63.657],
             [2,        4.303,        9.925],
             [3,        3.182,        5.841],
             [4,        2.776,        4.604],
             [5,        2.571,        4.032],
             [6,        2.447,        3.707],
             [7,        2.365,        3.499],
             [8,        2.306,        3.355],
             [9,        2.262,        3.250],
             [10,       2.228,        3.169],
             [11,       2.201,        3.106],
             [12,       2.179,        3.055],
             [13,       2.160,        3.012],
             [14,       2.145,        2.977],
             [15,       2.131,        2.947],
             [16,       2.120,        2.921],
             [17,       2.110,        2.898],
             [18,       2.101,        2.878],
             [19,       2.093,        2.861],
             [20,       2.086,        2.845],
             [21,       2.080,        2.831],
             [22,       2.074,        2.819],
             [23,       2.069,        2.807],
             [24,       2.064,        2.797],
             [25,       2.060,        2.787],
             [26,       2.056,        2.779],
             [27,       2.052,        2.771],
             [28,       2.048,        2.763],
             [29,       2.045,        2.756],
             [30,       2.042,        2.750],
             [40,       2.021,        2.704],
             [60,       2.000,        2.660],
             [120,      1.980,        2.617],
             [1000,     1.960,        2.576],
             [1E6,      1.960,        2.576]])

    st=np.var(t,ddof=1)
    lim95=np.interp(t.shape[0],tstud[:,0],tstud[:,1])
    lim99=np.interp(t.shape[0],tstud[:,0],tstud[:,2])
    lim95=lim95*np.sqrt(st)
    lim99=lim99*np.sqrt(st)
    return lim95,lim99

def f99(i,j):
    tab1= np.array(
            [[0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0,20.0,21.0,22.0,23.0,24.0,25.0,26.0,27.0,28.0,29.0,30.0,40.0,60.0,120.0],
             [1.0,4052,98.5,34.12,21.2,16.26,13.75,12.25,11.26,10.56,10.04,9.65,9.33,9.07,8.86,8.68,8.53,8.4,8.29,8.18,8.1,8.02,7.95,7.88,7.82,7.77,7.72,7.68,7.64,7.6,7.56,7.31,7.08,6.85],
             [2.0,4999.5,99,30.82,18,13.27,10.92,9.55,8.65,8.02,7.56,7.21,6.93,6.7,6.51,6.36,6.23,6.11,6.01,5.93,5.85,5.78,5.72,5.66,5.61,5.57,5.53,5.49,5.45,5.42,5.39,5.18,4.98,4.79],
             [3.0,5403,99.17,29.46,16.69,12.06,9.78,8.45,7.59,6.99,6.55,6.22,5.95,5.74,5.56,5.42,5.29,5.18,5.09,5.01,4.94,4.87,4.82,4.76,4.72,4.68,4.64,4.6,4.57,4.54,4.51,4.31,4.13,3.95],
             [4.0,5625,99.25,28.71,15.98,11.39,9.15,7.85,7.01,6.42,5.99,5.67,5.4,5.21,5.04,4.89,4.77,4.67,4.58,4.5,4.43,4.37,4.31,4.26,4.22,4.18,4.14,4.11,4.07,4.04,4.02,3.83,3.65,3.48],
             [5.0,5764,99.3,28.24,15.52,10.97,8.75,7.46,6.63,6.06,5.64,5.32,5.06,4.86,4.69,4.56,4.44,4.34,4.25,4.17,4.1,4.04,3.99,3.94,3.9,3.85,3.82,3.78,3.75,3.73,3.7,3.51,3.34,3.17],
             [6.0,5859,99.33,27.91,15.21,10.67,8.47,7.19,6.37,5.8,5.39,5.07,4.82,4.62,4.46,4.32,4.2,4.1,4.01,3.94,3.87,3.81,3.76,3.71,3.67,3.63,3.59,3.56,3.53,3.5,3.47,3.29,3.12,2.96],
             [7.0,5928,99.36,27.67,14.98,10.46,8.26,6.99,6.18,5.61,5.2,4.89,4.64,4.44,4.28,4.14,4.03,3.93,3.84,3.77,3.7,3.64,3.59,3.54,3.5,3.46,3.42,3.39,3.36,3.33,3.3,3.12,2.95,2.79],
             [8.0,5982,99.37,27.49,14.8,10.29,8.1,6.84,6.03,5.47,5.06,4.74,4.5,4.3,4.14,4,3.89,3.79,3.71,3.63,3.56,3.51,3.45,3.41,3.36,3.32,3.29,3.26,3.23,3.2,3.17,2.99,2.82,2.66],
             [9.0,6022,99.39,27.35,14.66,10.16,7.98,6.72,5.91,5.35,4.94,4.63,4.39,4.19,4.03,3.89,3.78,3.68,3.6,3.52,3.46,3.4,3.35,3.3,3.26,3.11,3.18,3.15,3.12,3.09,3.07,2.89,2.72,2.56],
             [10.0,6056,99.4,27.23,14.55,10.05,7.87,6.62,5.81,5.26,4.85,4.54,4.3,4.1,3.94,3.8,3.69,3.59,3.51,3.43,3.37,3.31,3.26,3.21,3.17,3.13,3.09,3.06,3.03,3,2.98,2.8,2.63,2.47],
             [12.0,6106,99.42,27.05,14.37,9.89,7.72,6.47,5.67,5.11,4.71,4.4,4.16,3.96,3.8,3.67,3.55,3.46,3.37,3.3,3.23,3.17,3.12,3.07,3.03,2.99,2.96,2.93,2.9,2.87,2.84,2.66,2.5,2.34],
             [15.0,6157,99.43,26.87,14.2,9.72,7.56,6.31,5.52,4.96,4.56,4.25,4.01,3.82,3.66,3.52,3.41,3.31,3.23,3.15,3.09,3.03,2.98,2.93,2.89,2.85,2.81,2.78,2.75,2.73,2.7,2.52,2.35,2.19],
             [20.0,6209,99.45,26.69,14.02,9.55,7.4,6.16,5.36,4.81,4.41,4.1,3.86,3.66,3.51,3.37,3.26,3.16,3.08,3,2.94,2.88,2.83,2.78,2.74,2.7,2.66,2.63,2.6,2.57,2.55,2.37,2.2,2.03],
             [24.0,6235,99.46,26.6,13.93,9.47,7.31,6.07,5.28,4.73,4.33,4.02,3.78,3.59,3.43,3.29,3.18,3.08,3,2.92,2.86,2.8,2.75,2.7,2.66,2.62,2.58,2.55,2.52,2.49,2.47,2.29,2.12,1.95],
             [30.0,6261,99.47,26.5,13.84,9.38,7.23,5.99,5.2,4.65,4.25,3.94,3.7,3.51,3.35,3.21,3.1,3,2.92,2.84,2.78,2.72,2.67,2.62,2.58,2.54,2.5,2.47,2.44,2.41,2.39,2.2,2.03,1.86],
             [40.0,6287,99.47,26.41,13.75,9.29,7.14,5.91,5.12,4.57,4.17,3.86,3.62,3.43,3.27,3.13,3.02,2.92,2.84,2.76,2.69,2.64,2.58,2.54,2.49,2.45,2.42,2.38,2.35,2.33,2.3,2.11,1.94,1.76],
             [60.0,6313,99.48,26.32,13.65,9.2,7.06,5.82,5.03,4.48,4.08,3.78,3.54,3.34,3.18,3.05,2.93,2.83,2.75,2.67,2.61,2.55,2.5,2.45,2.4,2.36,2.33,2.29,2.26,2.23,2.21,2.02,1.84,1.66],
             [120.0,6339,99.49,26.22,13.56,9.11,6.97,5.74,4.95,4.4,4,3.69,3.45,3.25,3.09,2.96,2.84,2.75,2.66,2.58,2.52,2.46,2.4,2.35,2.31,2.27,2.23,2.2,2.17,2.14,2.11,1.92,1.73,1.53]])
    # F-statistic where v2=infinity
    tab2=np.array(
             [[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,12.0,15.0,20.0,24.0,30.0,40.0,60.0,120.0],
             [ 6.63 ,4.61 ,3.78 ,3.32 ,3.02 ,2.80 ,2.64 ,2.51 ,2.41 ,2.32 ,2.18 ,2.04 ,1.88 ,1.79, 1.70, 1.59, 1.47, 1.32]])
    tab2=tab2.T
    tab3 =np.array(
            [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 40.0, 60.0, 120.0],
             [6366, 99.50, 26.13, 13.46, 9.02, 6.88, 5.65, 4.86, 4.31, 3.91, 3.60, 3.36, 3.17, 3.00, 2.87, 2.75, 2.65, 2.57, 2.49, 2.42, 2.36, 2.31, 2.26, 2.21, 2.17, 2.13, 2.10, 2.06, 2.03, 2.01, 1.80, 1.60, 1.38]])
    tab3=tab3.T
    
    if i<=120 and j<=120:
        Y=tab1[1:,0]
        X=tab1[0,1:]
        Z=tab1[1:,1:]
        f = interpolate.interp2d(X, Y, Z, kind='cubic')
        f99_=f(j,i)
        f99_=f99_[0]
    elif i>120 and j<=120:
        f99_=np.interp(j,tab3[:,0],tab3[:,1])
    elif i<=120 and j>120:
        f99_=np.interp(i,tab2[:,0],tab2[:,1])
    elif i>120 and j>120:
        f99_=1
    return f99_

def f95(i,j):
    tab1= np.array(
            [[0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0,20.0,21.0,22.0,23.0,24.0,25.0,26.0,27.0,28.0,29.0,30.0,40.0,60.0,120.0],
             [1.0,161.4,18.51,10.13,7.71,6.61,5.99,5.59,5.32,5.12,4.96,4.84,4.75,4.67,4.6,4.54,4.49,4.45,4.41,4.38,4.35,4.32,4.3,4.28,4.26,4.24,4.23,4.21,4.2,4.18,4.17,4.08,4,3.92],
             [2.0,199.5,19,9.55,6.94,5.79,5.14,4.74,4.46,4.26,4.1,3.98,3.89,3.81,3.74,3.68,3.63,3.59,3.55,3.52,3.49,3.47,3.44,3.42,3.4,3.39,3.37,3.35,3.34,3.33,3.32,3.23,3.15,3.07],
             [3.0,215.7,19.16,9.28,6.59,5.41,4.76,4.35,4.07,3.86,3.71,3.59,3.49,3.41,3.34,3.29,3.24,3.2,3.16,3.13,3.1,3.07,3.05,3.03,3.01,2.99,2.98,2.96,2.95,2.93,2.92,2.84,2.76,2.68],
             [4.0,224.6,19.25,9.12,6.39,5.19,4.53,4.12,3.84,3.63,3.48,3.36,3.26,3.18,3.11,3.06,3.01,2.96,2.93,2.9,2.87,2.84,2.82,2.8,2.78,2.76,2.74,2.73,2.71,2.7,2.69,2.61,2.53,2.45],
             [5.0,230.2,19.3,9.01,6.26,5.05,4.39,3.97,3.69,3.48,3.33,3.2,3.11,3.03,2.96,2.9,2.85,2.81,2.77,2.74,2.71,2.68,2.66,2.64,2.62,2.6,2.59,2.57,2.56,2.55,2.53,2.45,2.37,2.29],
             [6.0,234,19.33,8.94,6.16,4.95,4.28,3.87,3.58,3.37,3.22,3.09,3,2.92,2.85,2.79,2.74,2.7,2.66,2.63,2.6,2.57,2.55,2.53,2.51,2.49,2.47,2.46,2.45,2.43,2.42,2.34,2.25,2.17],
             [7.0,236.8,19.35,8.89,6.09,4.88,4.21,3.79,3.5,3.29,3.14,3.01,2.91,2.83,2.76,2.71,2.66,2.61,2.58,2.54,2.51,2.49,2.46,2.44,2.42,2.4,2.39,2.37,2.36,2.35,2.33,2.25,2.17,2.09],
             [8.0,238.9,19.37,8.85,6.04,4.82,4.15,3.73,3.44,3.23,3.07,2.95,2.85,2.77,2.7,2.64,2.59,2.55,2.51,2.48,2.45,2.42,2.4,2.37,2.36,2.34,2.32,2.31,2.29,2.28,2.27,2.18,2.1,2.02],
             [9.0,240.5,19.38,8.81,6,4.77,4.1,3.68,3.39,3.18,3.02,2.9,2.8,2.71,2.65,2.59,2.54,2.49,2.46,2.42,2.39,2.37,2.34,2.32,2.3,2.28,2.27,2.25,2.24,2.22,2.21,2.12,2.04,1.96],
             [10.0,241.9,19.4,8.79,5.96,4.74,4.06,3.64,3.35,3.14,2.98,2.85,2.75,2.67,2.6,2.54,2.49,2.45,2.41,2.38,2.35,2.32,2.3,2.27,2.25,2.24,2.22,2.2,2.19,2.18,2.16,2.08,1.99,1.91],
             [12.0,243.9,19.41,8.74,5.91,4.68,4,3.57,3.28,3.07,2.91,2.79,2.69,2.6,2.53,2.48,2.42,2.38,2.34,2.31,2.28,2.25,2.23,2.2,2.18,2.16,2.15,2.13,2.12,2.1,2.09,2,1.92,1.83],
             [15.0,245.9,19.43,8.7,5.86,4.62,3.94,3.51,3.22,3.01,2.85,2.72,2.62,2.53,2.46,2.4,2.35,2.31,2.27,2.23,2.2,2.18,2.15,2.13,2.11,2.09,2.07,2.06,2.04,2.03,2.01,1.92,1.84,1.75],
             [20.0,248,19.45,8.66,5.8,4.56,3.87,3.44,3.15,2.94,2.77,2.65,2.54,2.46,2.39,2.33,2.28,2.23,2.19,2.16,2.12,2.1,2.07,2.05,2.03,2.01,1.99,1.97,1.96,1.94,1.93,1.84,1.75,1.66],
             [24.0,249.1,19.45,8.64,5.77,4.53,3.84,3.41,3.12,2.9,2.74,2.61,2.51,2.42,2.35,2.29,2.24,2.19,2.15,2.11,2.08,2.05,2.03,2.01,1.98,1.96,1.95,1.93,1.91,1.9,1.89,1.79,1.7,1.61],
             [30.0,250.1,19.46,8.62,5.75,4.5,3.81,3.38,3.08,2.86,2.7,2.57,2.47,2.38,2.31,2.25,2.19,2.15,2.11,2.07,2.04,2.01,1.98,1.96,1.94,1.92,1.9,1.88,1.87,1.85,1.84,1.74,1.65,1.55],
             [40.0,251.1,19.47,8.59,5.72,4.46,3.77,3.34,3.04,2.83,2.66,2.53,2.43,2.34,2.27,2.2,2.15,2.1,2.06,2.03,1.99,1.96,1.94,1.91,1.89,1.87,1.85,1.84,1.82,1.81,1.79,1.69,1.59,1.5],
             [60.0,252.2,19.48,8.57,5.69,4.43,3.74,3.3,3.01,2.79,2.62,2.49,2.38,2.3,2.22,2.16,2.11,2.06,2.02,1.98,1.95,1.92,1.89,1.86,1.84,1.82,1.8,1.79,1.77,1.75,1.74,1.64,1.53,1.43],
             [120.0,253.3,19.49,8.55,5.66,4.4,3.7,3.27,2.97,2.75,2.58,2.45,2.34,2.25,2.18,2.11,2.06,2.01,1.97,1.93,1.9,1.87,1.84,1.81,1.79,1.77,1.75,1.73,1.71,1.7,1.68,1.58,1.47,1.35]])
    tab2=np.array(
            [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 15.0, 20.0, 24.0, 30.0, 40.0, 60.0, 120.0],
            [3.84, 3.00, 2.60, 2.37, 2.21, 2.10, 2.01, 1.94, 1.88, 1.83, 1.75, 1.67, 1.57, 1.52, 1.46, 1.39, 1.32, 1.22]])
    tab2=tab2.T
    
    tab3=np.array(
            [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 40.0, 60.0, 120.0],
            [245.3, 19.50, 8.53 ,5.63, 4.36, 3.67, 3.23, 2.93, 2.71, 2.54, 2.40, 2.30, 2.21, 2.13, 2.07, 2.01, 1.96, 1.92, 1.88, 1.84, 1.81, 1.78, 1.76, 1.73, 1.71, 1.69, 1.67, 1.65, 1.64, 1.62, 1.51, 1.39, 1.25]])
    tab3=tab3.T
    if i<=120 and j<=120:
        Y=tab1[1:,0]
        X=tab1[0,1:]
        Z=tab1[1:,1:]
        f = interpolate.interp2d(X, Y, Z, kind='cubic')
        f95_=f(j,i)
        f95_=f95_[0]
    elif i>120 and j<=120:
        f95_=np.interp(j,tab3[:,0],tab3[:,1])
    elif i<=120 and j>120:
        f95_=np.interp(i,tab2[:,0],tab2[:,1])
    elif i>120 and j>120:
        f95_=1
    return f95_
    

def scores_conf_int_calc(st,N):
    n_points=100
    cte2=((N-1)*(N+1)*(2))/(N*(N-2))
    f95_=cte2*f95(2,N-2)
    f99_=cte2*f99(2,N-2)
    xd95=np.sqrt(f95_*st[0,0])
    xd99=np.sqrt(f99_*st[0,0])
    xd95=np.linspace(-xd95,xd95,num=n_points)
    xd99=np.linspace(-xd99,xd99,num=n_points)
    
    st=np.linalg.inv(st)
    s11=st[0,0]
    s22=st[1,1]
    s12=st[0,1]
    s21=st[1,0]

    a=np.tile(s22,n_points)
    b=xd95*np.tile(s12,n_points)+xd95*np.tile(s21,n_points)
    c=(xd95**2)*np.tile(s11,n_points)-f95_
    safe_chk=b**2-4*a*c
    safe_chk[safe_chk<0]=0
    yd95p=(-b+np.sqrt(safe_chk))/(2*a)
    yd95n=(-b-np.sqrt(safe_chk))/(2*a)
    
    a=np.tile(s22,n_points)
    b=xd99*np.tile(s12,n_points)+xd99*np.tile(s21,n_points)
    c=(xd99**2)*np.tile(s11,n_points)-f99_
    safe_chk=b**2-4*a*c
    safe_chk[safe_chk<0]=0

    yd99p=(-b+np.sqrt(safe_chk))/(2*a)
    yd99n=(-b-np.sqrt(safe_chk))/(2*a)
    
    return xd95,xd99,yd95p,yd95n,yd99p,yd99n

def contributions(mvmobj,X,cont_type,*,Y=False,from_obs=False,to_obs=False,lv_space=False):
    """
    Calculate contributions to diagnostics
    by Salvador Garcia-Munoz 
    (sgarciam@ic.ac.uk ,salvadorgarciamunoz@gmail.com)
    
    mvmobj : A dictionary created by phi.pls or phi.pca
    
    X/Y:     Data [numpy arrays or pandas dataframes] - Y space is optional
    
    cont_type: 'ht2'
               'spe'
               'scores'
               
    from_obs: Scalar or list of scalars with observation(s) number(s) to offset (FROM)
              
    to_obs: Scalar or list of scalars with observation(s) number(s) to calculate contributions (TO)          
            *Note: from_obs is ignored when cont_type='spe'*
            
    lv_space: Latent spaces over which to do the calculations [applicable to 'ht2' and 'scores']
    """
    if isinstance(lv_space,np.bool):
        lv_space=list(range(mvmobj['T'].shape[1]))
    elif isinstance(lv_space,int):
        lv_space=np.array([lv_space])-1
        lv_space=lv_space.tolist()
    elif isinstance(lv_space,list):
        lv_space=np.array(lv_space)-1
        lv_space=lv_space.tolist()
    
    if isinstance(to_obs,int):
        to_obs=np.array([to_obs])
        to_obs=to_obs.tolist()
    elif isinstance(to_obs,list):
        to_obs=np.array(to_obs)
        to_obs=to_obs.tolist()
    
    if not(isinstance(from_obs,np.bool)):    
        if isinstance(from_obs,int):
           from_obs=np.array([from_obs])
           from_obs=from_obs.tolist()
        elif isinstance(from_obs,list):
            from_obs=np.array(from_obs)
            from_obs=from_obs.tolist()
        
    if isinstance(X,np.ndarray):
        X_ = X.copy()
    elif isinstance(X,pd.DataFrame):
        X_=np.array(X.values[:,1:]).astype(float)      
    if not(isinstance(Y,np.bool)):
        if isinstance(Y,np.ndarray):
            Y_=Y.copy()
        elif isinstance(Y,pd.DataFrame):
            Y_=np.array(Y.values[:,1:]).astype(float)
    if cont_type=='ht2' or cont_type=='scores':
        X_,dummy=n2z(X_)
        X_=((X_-np.tile(mvmobj['mx'],(X_.shape[0],1)))/(np.tile(mvmobj['sx'],(X_.shape[0],1))))
        t_stdevs=np.std(mvmobj['T'],axis=0,ddof=1)     
        if 'Q' in mvmobj:
            loadings=mvmobj['Ws']
        else:
            loadings=mvmobj['P']    
        to_obs_mean=np.mean(X_[to_obs,:],axis=0,keepdims=True)   
        to_cont=np.zeros((1,X_.shape[1]))
        for a in lv_space:    
                aux_=(to_obs_mean * np.abs(loadings[:,a].T))/t_stdevs[a]
                if cont_type=='scores':
                    to_cont=to_cont + aux_
                else:
                    to_cont=to_cont + aux_**2
        if not(isinstance(from_obs,np.bool)):
            from_obs_mean=np.mean(X_[from_obs,:],axis=0,keepdims=1)    
            from_cont=np.zeros((1,X_.shape[1]))
            for a in lv_space:    
                    aux_=(from_obs_mean * np.abs(loadings[:,a].T))/t_stdevs[a]
                    if cont_type=='scores':
                        from_cont=from_cont + aux_
                    else:
                        from_cont=from_cont + aux_**2
            calc_contribution = to_cont - from_cont
            return calc_contribution            
        else: 
            return to_cont

    elif cont_type=='spe':
        X_=X_[to_obs,:]
        if 'Q' in mvmobj:
            pred=pls_pred(X_,mvmobj)
        else:
            pred=pca_pred(X_,mvmobj)
        Xhat=pred['Xhat']       
        Xhatmcs=((Xhat-np.tile(mvmobj['mx'],(Xhat.shape[0],1)))/(np.tile(mvmobj['sx'],(Xhat.shape[0],1))))
        X_=((X_-np.tile(mvmobj['mx'],(X_.shape[0],1)))/(np.tile(mvmobj['sx'],(X_.shape[0],1))))
        Xerror=(X_-Xhatmcs)
        Xerror,dummy=n2z(Xerror)
        contsX=((Xerror)**2)*np.sign(Xerror)
        contsX=np.mean(contsX,axis=0,keepdims=True)
        
        if not(isinstance(Y,np.bool)):
            Y_=Y_[to_obs,:]
            Yhat=pred['Yhat']
            Yhatmcs=((Yhat-np.tile(mvmobj['my'],(Yhat.shape[0],1)))/(np.tile(mvmobj['sy'],(Yhat.shape[0],1))))
            Y_=((Y_-np.tile(mvmobj['my'],(Y_.shape[0],1)))/(np.tile(mvmobj['sy'],(Y_.shape[0],1))))
            Yerror=(Y_-Yhatmcs)
            Yerror,dummy=n2z(Yerror)
            contsY=((Yerror)**2)*np.sign(Yerror)
            contsY=np.mean(contsY,axis=0,keepdims=True)
            return contsX,contsY
        else:
            return contsX

def clean_empty_rows(X,*,shush=False):
    '''
    Input: 
        X: Matrix to be cleaned of empty rows (all np.nan)
    Output:
        X: Without observations removed
    '''
    if isinstance(X,np.ndarray):
        X_     = X.copy()
        ObsID_ = []
        for n in list(np.arange(X.shape[0])+1):
            ObsID_.append('Obs #'+str(n))  
    elif isinstance(X,pd.DataFrame):
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
                print('Removing row ', ObsID_[i], ' due to 100% missing data')
        if isinstance(X,pd.DataFrame):
            X_=X.drop(X.index.values[indx].tolist())
        else:
            X_=np.delete(X_,indx,0)
        return X_
    else:
        return X
    
def clean_low_variances(X,*,shush=False,min_var=1E-10):
    '''
 Input:
     X: Matrix to be cleaned for columns of low variance
     shush: 'True' disables output to console
Returns:     
X_clean:  Matrix without low variance columns
cols_removed:  Columns removed
    '''
    cols_removed=[]
    if isinstance(X,pd.DataFrame):
        X_=np.array(X.values[:,1:]).astype(float)
        varidX = X.columns.values
        varidX = varidX[1:]
        varidX = varidX.tolist()
    else:
        X_=X.copy()
        varidX=[]
        for n in list(np.arange(X.shape[1])+1):
            varidX.append('Var #'+str(n))   
            
    #find columns with all data missing, a column must have at least 3 samples
    X_nan_map = np.isnan(X_)
    Xmiss = X_nan_map*1
    Xmiss = np.sum(Xmiss,axis=0)
    
    #indx = find(Xmiss, lambda x: x==X_.shape[0])
    indx = find(Xmiss, lambda x: (x>=(X_.shape[0]-3)))
    
    
    if len(indx)>0:
        for i in indx:
            if not(shush):
                print('Removing variable ', varidX[i], ' due to 100% missing data')
        if isinstance(X,pd.DataFrame):
            for i in indx:
                cols_removed.append(varidX[i])
            indx = np.array(indx)
            indx = indx +1
            X_pd=X.drop(X.columns[indx],axis=1)
            X_=np.array(X_pd.values[:,1:]).astype(float)
        else:
            for i in indx:
                cols_removed.append(varidX[i])
            X_=np.delete(X_,indx,1)
    else:
        X_pd=X.copy()
        
    new_cols=X_pd.columns[1:].tolist() 
    std_x=std(X_)
    std_x=std_x.flatten()
    
    indx = find(std_x, lambda x: x<min_var)
    if len(indx)>0:
        for i in indx:
            if not(shush):
                print('Removing variable ', new_cols[i], ' due to low variance')
        if isinstance(X_pd,pd.DataFrame):
            for i in indx:
                cols_removed.append(new_cols[i])
            indx = np.array(indx)
            indx = indx +1
            X_=X_pd.drop(X_pd.columns[indx],axis=1)
        else:
            X_=np.delete(X_,indx,1)
            #cols_removed.extend(varidX[indx])
            for j in indx:
                cols_removed.append(varidX[j])
    
        return X_,cols_removed    
    else:
        return X_pd,cols_removed
    
def find(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]

def conv_pls_2_eiot(plsobj,*,r_length=False):
    plsobj_ = plsobj.copy()
    
    A = plsobj['T'].shape[1]
    N = plsobj['P'].shape[0]
    M = plsobj['Q'].shape[0]
    
    
    pyo_A = np.arange(1,A+1)  #index for LV's
    pyo_N = np.arange(1,N+1)  #index for columns of X
    pyo_M = np.arange(1,M+1)  #index for columns of Y
    pyo_A = pyo_A.tolist()
    pyo_N = pyo_N.tolist()
    pyo_M = pyo_M.tolist()
    
    pyo_Ws = np2D2pyomo(plsobj['Ws'])
    pyo_Q  = np2D2pyomo(plsobj['Q'])
    pyo_P  = np2D2pyomo(plsobj['P'])
    
    var_t = np.var(plsobj['T'],axis=0)
    
    pyo_var_t = np1D2pyomo(var_t)
    pyo_mx    = np1D2pyomo(plsobj['mx'])
    pyo_sx    = np1D2pyomo(plsobj['sx'])
    pyo_my    = np1D2pyomo(plsobj['my'])
    pyo_sy    = np1D2pyomo(plsobj['sy'])
    
    
    if not isinstance(r_length,bool):
        if r_length < N:   
            indx_r     = np.arange(1,r_length+1)
            indx_rk_eq = np.arange(r_length+1,N+1)
            indx_r     = indx_r.tolist()
            indx_rk_eq = indx_rk_eq.tolist()
        elif r_length == N:
            indx_r  = pyo_N
            indx_rk_eq=0
        else:
            print('r_length >> N !!')
            print('Forcing r_length=N')
            indx_r  = pyo_N
            indx_rk_eq=0
            
    else:
        if not r_length:
           indx_r  = pyo_N 
           indx_rk_eq = 0
            
    plsobj_['pyo_A']      = pyo_A
    plsobj_['pyo_N']      = pyo_N
    plsobj_['pyo_M']      = pyo_M
    plsobj_['pyo_Ws']     = pyo_Ws
    plsobj_['pyo_Q']      = pyo_Q
    plsobj_['pyo_P']      = pyo_P
    plsobj_['pyo_var_t']  = pyo_var_t
    plsobj_['indx_r']     = indx_r
    plsobj_['indx_rk_eq'] = indx_rk_eq
    plsobj_['pyo_mx']     = pyo_mx
    plsobj_['pyo_sx']     = pyo_sx
    plsobj_['pyo_my']     = pyo_my
    plsobj_['pyo_sy']     = pyo_sy
    plsobj_['S_I']        = np.nan
    plsobj_['pyo_S_I']    = np.nan
    plsobj_['var_t']      = var_t
    return plsobj_    
        
def prep_pca_4_MDbyNLP(pcaobj,X):
    pcaobj_ = pcaobj.copy()
    X_nan_map = np.isnan(X)
    psi = (np.logical_not(X_nan_map))*1
    X,dummy=n2z(X)
    
    A = pcaobj['T'].shape[1]
    O = pcaobj['T'].shape[0]
    N = pcaobj['P'].shape[0]
   
    pyo_A = np.arange(1,A+1)  #index for LV's
    pyo_N = np.arange(1,N+1)  #index for columns of X (rows of P)
    pyo_O = np.arange(1,O+1)  #index for rows of X 
    pyo_A = pyo_A.tolist()
    pyo_N = pyo_N.tolist()
    pyo_O = pyo_O.tolist()
    
    pyo_P_init  = np2D2pyomo(pcaobj['P'])
    pyo_T_init  = np2D2pyomo(pcaobj['T'])
    pyo_X       = np2D2pyomo(X)
    pyo_psi     = np2D2pyomo(psi)

    
    pcaobj_['pyo_A']      = pyo_A
    pcaobj_['pyo_N']      = pyo_N
    pcaobj_['pyo_O']      = pyo_O
    pcaobj_['pyo_P_init'] = pyo_P_init
    pcaobj_['pyo_T_init'] = pyo_T_init
    pcaobj_['pyo_X']      = pyo_X
    pcaobj_['pyo_psi']    = pyo_psi

    return pcaobj_    
    

def prep_pls_4_MDbyNLP(plsobj,X,Y):
    plsobj_ = plsobj.copy()
    X_nan_map = np.isnan(X)
    psi = (np.logical_not(X_nan_map))*1
    X,dummy=n2z(X)

    Y_nan_map = np.isnan(Y)
    theta = (np.logical_not(Y_nan_map))*1
    Y,dummy=n2z(Y)

    
    A = plsobj['T'].shape[1]
    O = plsobj['T'].shape[0]
    N = plsobj['P'].shape[0]
    M = plsobj['Q'].shape[0]
    
    pyo_A = np.arange(1,A+1)  #index for LV's
    pyo_N = np.arange(1,N+1)  #index for columns of X (rows of P)
    pyo_O = np.arange(1,O+1)  #index for rows of X 
    pyo_M = np.arange(1,M+1)  #index for columns of Y (rows of Q)
    pyo_A = pyo_A.tolist()
    pyo_N = pyo_N.tolist()
    pyo_O = pyo_O.tolist()
    pyo_M = pyo_M.tolist()
    
    pyo_P_init  = np2D2pyomo(plsobj['P'])
    pyo_Ws_init = np2D2pyomo(plsobj['Ws'])
    pyo_T_init  = np2D2pyomo(plsobj['T'])
    pyo_Q_init  = np2D2pyomo(plsobj['Q'])
    pyo_X       = np2D2pyomo(X)
    pyo_Y       = np2D2pyomo(Y)
    pyo_psi     = np2D2pyomo(psi)
    pyo_theta   = np2D2pyomo(theta)

    
    plsobj_['pyo_A']       = pyo_A
    plsobj_['pyo_N']       = pyo_N
    plsobj_['pyo_O']       = pyo_O
    plsobj_['pyo_M']       = pyo_M
    plsobj_['pyo_P_init']  = pyo_P_init
    plsobj_['pyo_Ws_init'] = pyo_Ws_init
    plsobj_['pyo_T_init']  = pyo_T_init
    plsobj_['pyo_Q_init']  = pyo_Q_init    
    plsobj_['pyo_X']       = pyo_X
    plsobj_['pyo_psi']     = pyo_psi
    plsobj_['pyo_Y']       = pyo_Y
    plsobj_['pyo_theta']   = pyo_theta

    return plsobj_   

def cat_2_matrix(X):
    '''
    cat_2_matrix(X) - Convert categorical data into binary matrices
    
    X is a Pandas Data Frame with categorical descriptors
    
    returns Xmat, XmatMB - 
    Xmat is matrix of binary coded data, 
    XmatMB same data, orgainzed as a list of matrices for Multi-block modeling
    
    '''
    FirstOne=True
    Xmat=[]
    Xcat=[]
    XcatMB=[]
    XmatMB=[]
    blknames=[]
    
    for x in X:
        if not(FirstOne):
            blknames.append(x)
            categories=np.unique(X[x])
            XcatMB.append(categories)
            Xmat_=[]
            for c in categories:
                Xcat.append(c)
                xmat_=(X[x]==c)*1
                Xmat.append(xmat_)
                Xmat_.append(xmat_)
                
            Xmat_=np.array(Xmat_)
            Xmat_=Xmat_.T
            Xmat_ = pd.DataFrame(Xmat_,columns=categories) 
            Xmat_.insert(0,firstcol,X[firstcol]) 
            XmatMB.append(Xmat_)            
        else:
            firstcol=x
            FirstOne=False
    Xmat=np.array(Xmat)
    Xmat=Xmat.T
    Xmat = pd.DataFrame(Xmat,columns=Xcat) 
    Xmat.insert(0,firstcol,X[firstcol])      
    XmatMB={'data':XmatMB,'blknames':blknames}
    return Xmat,XmatMB

def mbpls(XMB,YMB,A,*,mcsX=True,mcsY=True,md_algorithm_='nipals',force_nipals_=False,shush_=False,cross_val_=0,cross_val_X_=False):
    '''
    Multi-block PLS model using the approach by Westerhuis, J. Chemometrics, 12, 301–321 (1998)
    
    Inputs
    ----------
    XMB : Dictionary or PandasDataFrame
        Dictionary structure:
        {'BlockName1':block_1_data_pd,
         'BlockName2':block_2_data_pd}

    YMB : Dictionary or PandasDataFrame
        Dictionary structure:
        {'BlockName1':block_1_data_pd,
         'BlockName2':block_2_data_pd}
          
    '''
    x_means=[]
    x_stds=[]
    y_means=[]
    y_stds=[]
    Xblk_scales=[]
    Yblk_scales=[]
    Xcols_per_block=[]
    Ycols_per_block=[]
    X_var_names=[]
    Y_var_names=[]
    obsids=[]
    
    if isinstance(XMB,dict):
        data_=[]
        names_=[]
        for k in XMB.keys():
            data_.append(XMB[k])
            names_.append(k)
        XMB={'data':data_,'blknames':names_}
        
        
        x=XMB['data'][0]        
        columns=x.columns.tolist()
        obsid_column_name=columns[0]        
        obsids=x[obsid_column_name].tolist()
        
        c=0
        for x in XMB['data']:        
            x_=x.values[:,1:].astype(float)         
            columns=x.columns.tolist()
            for i,h in enumerate(columns):
                if i!=0:
                    X_var_names.append( XMB['blknames'][c]+' '+h)
                    
            if isinstance(mcsX,bool):
                if mcsX:
                    #Mean center and autoscale  
                    x_,x_mean_,x_std_ = meancenterscale(x_)
                else:    
                    x_mean_ = np.zeros((1,x_.shape[1]))
                    x_std_  = np.ones((1,x_.shape[1]))
            elif mcsX[c]=='center':
                #only center
                x_,x_mean_,x_std_ = meancenterscale(x_,mcs='center')
            elif mcsX[c]=='autoscale':
                #only autoscale
                x_,x_mean_,x_std_ = meancenterscale(x_,mcs='autoscale')
            blck_scale=np.sqrt(np.sum(std(x_)**2))
            
            x_means.append(x_mean_)
            x_stds.append(x_std_)                   
            Xblk_scales.append(blck_scale)
            Xcols_per_block.append(x_.shape[1])
            
            x_=x_/blck_scale
            if c==0:
                X_=x_.copy() 
            else:    
                X_=np.hstack((X_,x_))
            c=c+1
    elif isinstance(XMB,pd.DataFrame):
        columns=XMB.columns.tolist()
        obsid_column_name=columns[0]        
        obsids=XMB[obsid_column_name].tolist()            
        for i,h in enumerate(columns):
            if i!=0:
                X_var_names.append(h)
        x_=XMB.values[:,1:].astype(float)
        if isinstance(mcsX,bool):
            if mcsX:
                #Mean center and autoscale  
                x_,x_mean_,x_std_ = meancenterscale(x_)
            else:    
                x_mean_ = np.zeros((1,x_.shape[1]))
                x_std_  = np.ones((1,x_.shape[1]))
        elif mcsX[c]=='center':
            #only center
            x_,x_mean_,x_std_ = meancenterscale(x_,mcs='center')
        elif mcsX[c]=='autoscale':
            #only autoscale
            x_,x_mean_,x_std_ = meancenterscale(x_,mcs='autoscale')
        blck_scale=np.sqrt(np.sum(std(x_)**2))
        
        x_means.append(x_mean_)
        x_stds.append(x_std_)                   
        Xblk_scales.append(blck_scale)
        Xcols_per_block.append(x_.shape[1])
        x_=x_/blck_scale
        X_=x_.copy() 
        
    if isinstance(YMB,dict):
        data_=[]
        names_=[]
        for k in YMB.keys():
            data_.append(YMB[k])
            names_.append(k)
        YMB={'data':data_,'blknames':names_}
        
        c=0
        for y in YMB['data']:        
            y_=y.values[:,1:].astype(float)
            columns=y.columns.tolist()
            for i,h in enumerate(columns):
                if i!=0:
                    Y_var_names.append(h)
            if isinstance(mcsY,bool):
                if mcsY:
                    #Mean center and autoscale  
                    y_,y_mean_,y_std_ = meancenterscale(y_)
                else:    
                    y_mean_ = np.zeros((1,y_.shape[1]))
                    y_std_  = np.ones((1,y_.shape[1]))
            elif mcsY[c]=='center':
                #only center
                y_,y_mean_,y_std_ = meancenterscale(y_,mcs='center')
            elif mcsY[c]=='autoscale':
                #only autoscale
                y_,y_mean_,y_std_ = meancenterscale(y_,mcs='autoscale')
            blck_scale=np.sqrt(np.sum(std(y_)**2))
            
            y_means.append(y_mean_)
            y_stds.append(y_std_)                   
            Yblk_scales.append(blck_scale)
            Ycols_per_block.append(y_.shape[1])
            y_=y_/blck_scale
            if c==0:
                Y_=y_.copy() 
            else:    
                Y_=np.hstack((Y_,y_))
            c=c+1    
    elif isinstance(YMB,pd.DataFrame):
        y_=YMB.values[:,1:].astype(float)
        columns=YMB.columns.tolist()
        for i,h in enumerate(columns):
            if i!=0:
                Y_var_names.append(h)
        
        
        if isinstance(mcsY,bool):
            if mcsY:
                #Mean center and autoscale  
                y_,y_mean_,y_std_ = meancenterscale(y_)
            else:    
                y_mean_ = np.zeros((1,y_.shape[1]))
                y_std_  = np.ones((1,y_.shape[1]))
        elif mcsY[c]=='center':
            #only center
            y_,y_mean_,y_std_ = meancenterscale(y_,mcs='center')
        elif mcsY[c]=='autoscale':
            #only autoscale
            y_,y_mean_,y_std_ = meancenterscale(y_,mcs='autoscale')
        blck_scale=np.sqrt(np.sum(std(y_)**2))
        
        y_means.append(y_mean_)
        y_stds.append(y_std_)                   
        Yblk_scales.append(blck_scale)
        Ycols_per_block.append(y_.shape[1])
        y_=y_/blck_scale
        Y_=y_.copy() 
        
        
    X_pd=pd.DataFrame(X_,columns=X_var_names)
    X_pd.insert(0,obsid_column_name,obsids)

    Y_pd=pd.DataFrame(Y_,columns=Y_var_names)
    Y_pd.insert(0,obsid_column_name,obsids)

    pls_obj_=pls(X_pd,Y_pd,A,mcsX=False,mcsY=False,md_algorithm=md_algorithm_,force_nipals=force_nipals_,shush=shush_,cross_val=cross_val_,cross_val_X=cross_val_X_)          
    pls_obj_['type']='mbpls'
    #Calculate block loadings, scores, weights
    Wsb=[]
    Wb=[]
    Tb=[]
     
    for i,c in enumerate(Xcols_per_block):
        if i==0:
            start_index=0
            end_index=c            
        else:
            start_index=np.sum(Xcols_per_block[0:i])
            end_index = start_index + c

        wsb_=pls_obj_['Ws'][start_index:end_index,:].copy()
        for j in list(range(wsb_.shape[1])):
            wsb_[:,j]=wsb_[:,j]/np.linalg.norm(wsb_[:,j])
        Wsb.append(wsb_)
        
        wb_=pls_obj_['W'][start_index:end_index,:].copy()
        for j in list(range(wb_.shape[1])):
            wb_[:,j]=wb_[:,j]/np.linalg.norm(wb_[:,j])
        Wb.append(wb_)
        
        Xb=X_[:,start_index:end_index]
        tb=[]
        X_nan_map = np.isnan(Xb)
        not_Xmiss = (np.logical_not(X_nan_map))*1
        Xb,dummy=n2z(Xb)
        TSS = np.sum(Xb**2)
        
        for a in list(range(A)):            
            w_=wb_[:,[a]]
            w_t = np.tile(w_.T,(Xb.shape[0],1))
            w_t = w_t * not_Xmiss
            w_t = np.sum(w_t**2,axis=1,keepdims=True)           
            tb_=(Xb @ w_) / w_t
            if a==0:
                tb=tb_
            else:
                tb=np.hstack((tb,tb_))
            
            tb_t = np.tile(tb_.T,(Xb.shape[1],1))
            tb_t = tb_t * not_Xmiss.T
            tb_t = np.sum(tb_t**2,axis=1,keepdims=True)
            pb_  = (Xb.T @ tb_) / tb_t
            
            Xb = (Xb - tb_ @ pb_.T) * not_Xmiss
            r2pb_aux=1-(np.sum(Xb**2)/TSS)
            if a==0:
                r2pb_=r2pb_aux
            else:
                r2pb_=np.hstack((r2pb_,r2pb_aux))
        if i==0:
            r2pbX=r2pb_
        else:            
            r2pbX=np.vstack((r2pbX,r2pb_))
        Tb.append(tb)
    for a in list(range(A)):
        T_a=[]
        u=pls_obj_['U'][:,[a]].copy()
        for i,c in enumerate(Xcols_per_block):
            if i==0:
                T_a = Tb[i][:,[a]]
            else:
                T_a = np.hstack((T_a,Tb[i][:,[a]]))
        wt_=(T_a.T @ u) / (u.T @ u)
        if a==0:
            Wt=wt_
        else:
            Wt=np.hstack((Wt,wt_))
    
    pls_obj_['x_means']=x_means
    pls_obj_['x_stds']=x_stds
    pls_obj_['y_means']=y_means
    pls_obj_['y_stds']=y_stds
    pls_obj_['Xblk_scales']=Xblk_scales
    pls_obj_['Yblk_scales']=Yblk_scales
    pls_obj_['Wsb']=Wsb
    pls_obj_['Wt']=Wt
    mx_=[]
    for i,l in enumerate(x_means):
        for j in l[0]:
            mx_.append(j)
    pls_obj_['mx']=np.array(mx_)    
    sx_=[]
    for i,l in enumerate(x_stds):
        for j in l[0]:
            sx_.append(j*Xblk_scales[i])
    pls_obj_['sx']=np.array(sx_)
    my_=[]
    for i,l in enumerate(y_means):
        for j in l[0]:
            my_.append(j)
    pls_obj_['my']=np.array(my_)    
    sy_=[]
    for i,l in enumerate(y_stds):
        for j in l[0]:
            sy_.append(j*Yblk_scales[i])
    pls_obj_['sy']=np.array(sy_)
    
    
    if isinstance(XMB,dict):
        for a in list(range(A-1,0,-1)):
             r2pbX[:,[a]]     = r2pbX[:,[a]]-r2pbX[:,[a-1]]
        r2pbXc = np.cumsum(r2pbX,axis=1)
    
        pls_obj_['r2pbX']=r2pbX
        pls_obj_['r2pbXc']=r2pbXc
    else:
        for a in list(range(A-1,0,-1)):
             r2pbX[a]     = r2pbX[a]-r2pbX[a-1]
        r2pbXc = np.cumsum(r2pbX)
    
        pls_obj_['r2pbX']=r2pbX
        pls_obj_['r2pbXc']=r2pbXc

    if isinstance(XMB,dict):
        pls_obj_['Xblocknames']=XMB['blknames']
    if isinstance(YMB,dict):
        pls_obj_['Yblocknames']=YMB['blknames']
    return pls_obj_

def replicate_data(mvm_obj,X,num_replicates,*,as_set=False):
    
    if 'Q' in mvm_obj:
        pls_preds=pls_pred(X,mvm_obj)
        xhat=pls_preds['Xhat']
        tnew=pls_preds['Tnew']
    else:
        pca_preds=pca_pred(X,mvm_obj)
        xhat=pca_preds['Xhat']
        tnew=pca_preds['Tnew']
    xhat = (xhat - np.tile(mvm_obj['mx'],(xhat.shape[0],1)))/np.tile(mvm_obj['sx'],(xhat.shape[0],1))
    xhat = tnew @ mvm_obj['P'].T
    xmcs = (X.values[:,1:] - np.tile(mvm_obj['mx'],(xhat.shape[0],1)))/np.tile(mvm_obj['sx'],(xhat.shape[0],1))
    data_residuals=xmcs-xhat

    if not(as_set):
        if np.mod(num_replicates,X.shape[0])==0:
            reps=num_replicates/X.shape[0]            
            new_set=np.tile(xhat,(int(reps),1))
        else:
            reps=np.floor(num_replicates/X.shape[0])
            new_set=np.tile(xhat,(int(reps),1))
            reps=np.mod(num_replicates,X.shape[0])
            new_set=np.vstack((new_set,xhat[:reps,:]))                
        obsids=[]
        for i in np.arange(new_set.shape[0])+1:
            obsids.append('clone'+str(i))
            
    else:            
        new_set=np.tile(xhat,(num_replicates,1))
        obsids=[]
        obsid_o=X[X.columns[0]].values.astype(str).tolist()
        for i in np.arange(num_replicates)+1:
            post_fix=['_clone'+str(i)]*X.shape[0]
            obsids_=[m+n for m,n in zip(obsid_o,post_fix)]
            obsids.extend(obsids_)
            
    for i in list(range(0,data_residuals.shape[1])):
#        plt.figure()
#        plt.hist(data_residuals[:,i])
#        plt.title('Original Residual Distribution')
        ecdf = ECDF(data_residuals[:,i])
        new_residual= np.random.uniform(0,1,new_set.shape[0])
        y=np.array(ecdf.y.tolist())
        x=np.array(ecdf.x.tolist())
        new_residual=np.interp(new_residual,y[1:],x[1:])
        
        if i==0:
            uncertainty_matrix=np.reshape(new_residual,(-1,1))
        else:
            uncertainty_matrix=np.hstack((uncertainty_matrix,np.reshape(new_residual,(-1,1))))
    new_set=new_set+uncertainty_matrix
    new_set=(new_set * np.tile(mvm_obj['sx'],(new_set.shape[0],1)))+np.tile(mvm_obj['mx'],(new_set.shape[0],1))
    new_set_pd=pd.DataFrame(new_set,columns=X.columns[1:].tolist())
    new_set_pd.insert(0,X.columns[0],obsids)
    
    return new_set_pd

def export_2_gproms(mvmobj,*,fname='phi_export.txt'):
    top_lines=[     
    'PARAMETER',
    'X_VARS AS ORDERED_SET',
    'Y_VARS AS ORDERED_SET',
    'A      AS INTEGER',
    'VARIABLE',
    'X_MEANS as ARRAY(X_VARS)    OF no_type',
    'X_STD   AS ARRAY(X_VARS)    OF no_type',
    'Y_MEANS as ARRAY(Y_VARS)    OF no_type',
    'Y_STD   AS ARRAY(Y_VARS)    OF no_type',
    'Ws      AS ARRAY(X_VARS,A)  OF no_type',
    'Q       AS ARRAY(Y_VARS,A)  OF no_type',
    'P       AS ARRAY(X_VARS,A)  OF no_type',
    'T       AS ARRAY(A)         OF no_type',
    'Tvar    AS ARRAY(A)         OF no_type',
    'X_HAT   AS ARRAY(X_VARS)    OF no_type # Mean-centered and scaled',
    'Y_HAT   AS ARRAY(Y_VARS)    OF no_type # Mean-centered and scaled',
    'X_PRED  AS ARRAY(X_VARS)    OF no_type # In original units',
    'Y_PRED  AS ARRAY(Y_VARS)    OF no_type # In original units',
    'X_NEW   AS ARRAY(X_VARS)    OF no_type # In original units',
    'X_MCS   AS ARRAY(X_VARS)    OF no_type # Mean-centered and scaled',
    'HT2                         AS no_type',
    'SPEX                        AS no_type',
    'SET']
    
    x_var_line="X_VARS:=['" + mvmobj['varidX'][0] + "'"
    for v in mvmobj['varidX'][1:]:
        x_var_line=x_var_line+",'"+v+"'"
    x_var_line=x_var_line+'];'

    y_var_line="Y_VARS:=['" + mvmobj['varidY'][0] + "'"
    if len(mvmobj['varidY']) > 1:
        for v in mvmobj['varidY'][1:]:
            y_var_line=y_var_line+",'"+v+"'"
    y_var_line=y_var_line+'];'
    

    top_lines.append(x_var_line)
    top_lines.append(y_var_line)
    top_lines.append('A:='+str(mvmobj['T'].shape[1])+';')
    
    mid_lines=[
    'EQUATION',
    'X_MCS * X_STD = (X_NEW-X_MEANS);',
    'FOR j:=1 TO A DO',
    'T(j) = SIGMA(X_MCS*Ws(,j));',
    'END',
    'FOR i IN Y_VARS DO',
    'Y_HAT(i) = SIGMA(T*Q(i,));',
    'END',
    'FOR i IN X_VARS DO',
    'X_HAT(i) = SIGMA(T*P(i,));',
    'END',
    '(X_HAT * X_STD) + X_MEANS = X_PRED;',
    '(Y_HAT * Y_STD) + Y_MEANS = Y_PRED;',
    'HT2  = SIGMA ((T^2)/Tvar);',
    'SPEX = SIGMA ((X_MCS - X_HAT)^2);']

    assign_lines=['ASSIGN']
    for i,xvar in enumerate(mvmobj['varidX']):     
        assign_lines.append("X_MEANS('"+xvar+"') := "+str(mvmobj['mx'][0,i])+";" )

    for i,xvar in enumerate(mvmobj['varidX']):     
        assign_lines.append("X_STD('"+xvar+"') := "+str(mvmobj['sx'][0,i])+";" )

    for i,yvar in enumerate(mvmobj['varidY']):     
        assign_lines.append("Y_MEANS('"+yvar+"') := "+str(mvmobj['my'][0,i])+";" )

    for i,yvar in enumerate(mvmobj['varidY']):     
        assign_lines.append("Y_STD('"+yvar+"') := "+str(mvmobj['sy'][0,i])+";" )

    for i,xvar in enumerate(mvmobj['varidX']):     
        for j in np.arange(mvmobj['Ws'].shape[1]):
            assign_lines.append("Ws('"+xvar+"',"+str(j+1)+") := "+str(mvmobj['Ws'][i,j])+";")

    for i,xvar in enumerate(mvmobj['varidX']):     
        for j in np.arange(mvmobj['P'].shape[1]):
            assign_lines.append("P('"+xvar+"',"+str(j+1)+") := "+str(mvmobj['P'][i,j])+";")
    
    for i,yvar in enumerate(mvmobj['varidY']):     
        for j in np.arange(mvmobj['Q'].shape[1]):
            assign_lines.append("Q('"+yvar+"',"+str(j+1)+") := "+str(mvmobj['Q'][i,j])+";")
    tvar=np.std(mvmobj['T'],axis=0,ddof=1)
    for j in np.arange(mvmobj['T'].shape[1]):
        assign_lines.append("Tvar("+str(j+1)+") := "+str(tvar[j])+";" ) 
 
    lines=top_lines
    lines.extend(mid_lines)
    lines.extend(assign_lines)

    with open(fname, "w") as outfile:
        outfile.write("\n".join(lines))

        return
def unique(df,colid):    
    '''
    replacement of the np.unique routine, specifically for dataframes
    returns unique values in the order found in the dataframe
    df:     A pandas dataframe
    colid:  Column identifier 
    
    '''
    aux=df.drop_duplicates(subset=colid,keep='first')
    unique_entries=aux[colid].values.tolist()
    return unique_entries
    
def parse_materials(filename,sheetname):
    '''
    Routine to parse out compositions from linear table
    This reads an excel file with four columns:
        'Finished Product Lot'	'Material Lot'	'Ratio or Quantity'	'Material'
        
    where the usage per batch of finished product is recorded. e.g.

'Finished Product Lot'	'Material Lot'	'Ratio or Quantity'	'Material'
        A001                 A                0.75             Drug
        A001                 B                0.25             Drug
        A001                 Z                1.0              Excipient
        .                    .                 .                 .
        .                    .                 .                 .
        .                    .                 .                 .
        
    Returns:
        JR = Joint R matrix of material consumption, list of dataframes
        materials_used = Names of materials 
    '''
    materials=pd.read_excel(filename,sheet_name=sheetname)

    ok=True
    for lot in unique(materials,'Finished Product Lot'):
        this_lot=materials[materials["Finished Product Lot"]==lot]
        for mt,m in zip(this_lot['Material'].values, this_lot['Material Lot'].values):
            try:
               if np.isnan(m):
                   print('Lot '+lot+' has no Material Lot for '+mt)
                   ok=False
                   break
            except:
                d=1
        if not(ok):
            break
        print('Lot :'+lot+' ratio/qty adds to '+str(np.sum(this_lot['Ratio or Quantity'].values) ))
    if ok:    
        JR=[]
        materials_used=unique(materials,'Material')
        fp_lots=unique(materials,'Finished Product Lot')
        for m in materials_used:
            r_mat=[]
            mat_lots=np.unique(materials['Material Lot'][materials['Material']==m]).tolist()
            for lot in fp_lots:
                rvec=np.zeros(len(mat_lots))
                this_lot_this_mat=materials[(materials["Finished Product Lot"]==lot) &
                                            (materials['Material']==m)]
                for l,r in zip(this_lot_this_mat['Material Lot'].values,
                               this_lot_this_mat['Ratio or Quantity'].values):
                    rvec[mat_lots.index(l)]=r
                r_mat.append(rvec)    
            r_mat_pd=pd.DataFrame(np.array(r_mat),columns=mat_lots)
            r_mat_pd.insert(0,'FPLot',fp_lots)    
            JR.append(r_mat_pd)
        return JR, materials_used
    else:
        print('Data needs revision')
        return False,False
    
def isin_ordered_col0(df,alist):
    df_=df.copy()
    df_=df[df[df.columns[0]].isin(alist)]
    df_=df_.set_index(df_.columns[0] )
    df_=df_.reindex(alist)
    df_=df_.reset_index()
    return df_
    
def reconcile_rows(df_list):
    all_rows=[]
    for df in df_list:
        all_rows.extend(df[df.columns[0]].values.tolist())
    all_rows=np.unique(all_rows)    
    for df in df_list:
        rows=df[df.columns[0]].values.tolist()
        rows_=[]
        for r in all_rows:
            if r in rows:
                rows_.append(r)
        all_rows=rows_.copy()
    new_df_list=[]
    for df in df_list:
        df=isin_ordered_col0(df,all_rows)
        new_df_list.append(df)
    return new_df_list
    
def reconcile_rows_to_columns(df_list_r,df_list_c): 
    df_list_r_o=[]
    df_list_c_o=[]
    allids=[]
    for dfr,dfc in zip(df_list_r,df_list_c ):
        all_ids  = dfc.columns[1:].tolist()
        all_ids.extend(dfr[dfr.columns[0]].values.tolist())
        all_ids = np.unique(all_ids)
        
        all_ids_=[]
        rows=dfr[dfr.columns[0]].values.tolist()
        cols=dfc.columns[1:].tolist()
        for i in all_ids:
            if i in rows:
                all_ids_.append(i)
        all_ids=[]         
        for i in all_ids_:
            if i in cols:
                all_ids.append(i)
                
        dfr_ =isin_ordered_col0(dfr,all_ids)
        
        dfc_=dfc[all_ids]
        dfc_.insert(0,dfc.columns[0],dfc[dfc.columns[0]].values.tolist())
        df_list_r_o.append(dfr_)
        df_list_c_o.append(dfc_)
        allids.append(all_ids)
    return df_list_r_o,df_list_c_o
    
def _Ab_btbinv(A,b,A_not_nan_map):
    # project c = Ab/b'b
    # A = [i x j]  b=[j x 1] c = [i x 1]
    b_mat=np.tile(b.T,(A.shape[0],1))
    c =(np.sum(A*b_mat,axis=1))/(np.sum((b_mat*A_not_nan_map)**2,axis=1))
    return c.reshape(-1,1)    

def lpls(X,R,Y,A,*,shush=False):
    '''
    #LPLS Algorithm per Muteki et. al Chemom.Intell.Lab.Syst.85(2007) 186 – 194
    # X = [ m x p ] Phys. Prop. DataFrame of             materials x mat. properties
    # R = [ b x m ] Blending ratios DataFrame of         blends    x materials
    # Y = [ b x n ] Product characteristics DataFrame of blends    x prod. properties
    #first column of all dataframes is the observation identifier
    # A = Number of components
    '''
    if isinstance(X,np.ndarray):
        X_ = X.copy()
        obsidX = False
        varidX = False
    elif isinstance(X,pd.DataFrame):
        X_=np.array(X.values[:,1:]).astype(float)
        obsidX = X.values[:,0].astype(str)
        obsidX = obsidX.tolist()
        varidX = X.columns.values
        varidX = varidX[1:]
        varidX = varidX.tolist()
        
    if isinstance(Y,np.ndarray):
        Y_=Y.copy()
        obsidY = False
        varidY = False
    elif isinstance(Y,pd.DataFrame):
        Y_=np.array(Y.values[:,1:]).astype(float)
        obsidY = Y.values[:,0].astype(str)
        obsidY = obsidY.tolist()
        varidY = Y.columns.values
        varidY = varidY[1:]
        varidY = varidY.tolist()    
        
    if isinstance(R,np.ndarray):
        R_=R.copy()
        obsidR = False
        varidR = False
    elif isinstance(R,pd.DataFrame):
        R_=np.array(R.values[:,1:]).astype(float)
        obsidR = R.values[:,0].astype(str)
        obsidR = obsidR.tolist()
        varidR = R.columns.values
        varidR = varidR[1:]
        varidR = varidR.tolist()  
        
    X_,x_mean,x_std = meancenterscale(X_)
    Y_,y_mean,y_std = meancenterscale(Y_)
    R_,r_mean,r_std = meancenterscale(R_)
    
    #Generate Missing Data Map    
    X_nan_map = np.isnan(X_)
    not_Xmiss = (np.logical_not(X_nan_map))*1
    Y_nan_map = np.isnan(Y_)
    not_Ymiss = (np.logical_not(Y_nan_map))*1
    R_nan_map = np.isnan(R_)
    not_Rmiss = (np.logical_not(R_nan_map))*1

    #use nipals
    if not(shush):
        print('phi.lpls using NIPALS executed on: '+ str(datetime.datetime.now()) )
    X_,dummy=n2z(X_)
    Xhat = np.zeros(X_.shape)
    Y_,dummy=n2z(Y_)
    R_,dummy=n2z(R_)
    epsilon=1E-9
    maxit=2000

    TSSX   = np.sum(X_**2)
    TSSXpv = np.sum(X_**2,axis=0)
    TSSY   = np.sum(Y_**2)
    TSSYpv = np.sum(Y_**2,axis=0)
    TSSR   = np.sum(R_**2)
    TSSRpv = np.sum(R_**2,axis=0)
    
    #T=[];
    #P=[];
    #r2=[];
    #r2pv=[];
    #numIT=[];
    for a in list(range(A)):
        # Select column with largest variance in Y as initial guess
        ui = Y_[:,[np.argmax(std(Y_))]]
        Converged=False
        num_it=0
        while Converged==False:
            
     # _Ab_btbinv(A,b,A_not_nan_map):
     #  project c = Ab/b'b

             #Step 1. h=R'u/u'u
             hi = _Ab_btbinv(R_.T,ui,not_Rmiss.T)
             #print('step 1')
             
             #Step 2. s = X'h/(h'h)
             si = _Ab_btbinv(X_.T, hi,not_Xmiss.T)
             #print('step 2')
                   
             #Normalize s to unit length.
             si=si/np.linalg.norm(si)
             
             #Step 3. ri= (Xs)/(s's)
             ri = _Ab_btbinv(X_,si,not_Xmiss)
             #print('step 3')
             #Step 4. t = Rr/(r'r)
             ti = _Ab_btbinv(R_,ri,not_Rmiss)
             #print('step 4')
             #Step 5 q=Y't/t't
             qi = _Ab_btbinv(Y_.T,ti,not_Ymiss.T)
             
             #Step 5 un=(Yq)/(q'q)
             un = _Ab_btbinv(Y_,qi,not_Ymiss)
             #print('step 5')
             
             if abs((np.linalg.norm(ui)-np.linalg.norm(un)))/(np.linalg.norm(ui)) < epsilon:
                 Converged=True
                 
             if num_it > maxit:
                 Converged=True
                 
             if Converged:
                 if not(shush):
                     print('# Iterations for LV #'+str(a+1)+': ',str(num_it))
                 # Calculate P's for deflation p=R't/(t't) 
                 pi = _Ab_btbinv(R_.T,ti,not_Rmiss.T)
                 # Calculate v's for deflation v=Xr/(r'r) 
                 vi = _Ab_btbinv(X_.T,ri,not_Xmiss.T)
                 
                 # Deflate X leaving missing as zeros (important!)
                 R_=(R_- ti @ pi.T)*not_Rmiss
                 X_=(X_- ri @ vi.T)*not_Xmiss
                 Y_=(Y_- ti @ qi.T)*not_Ymiss
                 
                 Xhat = Xhat + ri @ vi.T
                 #print(R_.shape)
                 #print(X_.shape)
                 #print(Y_.shape)
                 if a==0:
                     T=ti
                     P=pi
                     S=si
                     U=un
                     Q=qi
                     H=hi
                     V=vi
                     Rscores = ri
                     
                     r2X   = 1-np.sum(X_**2)/TSSX
                     r2Xpv = 1-np.sum(X_**2,axis=0)/TSSXpv
                     r2Xpv = r2Xpv.reshape(-1,1)
                     r2Y   = 1-np.sum(Y_**2)/TSSY
                     r2Ypv = 1-np.sum(Y_**2,axis=0)/TSSYpv
                     r2Ypv = r2Ypv.reshape(-1,1)
                     r2R   = 1-np.sum(R_**2)/TSSR
                     r2Rpv = 1-np.sum(R_**2,axis=0)/TSSRpv
                     r2Rpv = r2Rpv.reshape(-1,1)
                                 
                 else:
                     T=np.hstack((T,ti.reshape(-1,1)))
                     U=np.hstack((U,un.reshape(-1,1)))
                     P=np.hstack((P,pi))   
                     Q=np.hstack((Q,qi))
                     S=np.hstack((S,si))
                     H=np.hstack((H,hi))
                     V=np.hstack((V,vi))
                     Rscores=np.hstack((Rscores,ri ))
                        
                     r2X_   = 1-np.sum(X_**2)/TSSX
                     r2Xpv_ = 1-np.sum(X_**2,axis=0)/TSSXpv
                     r2Xpv_ = r2Xpv_.reshape(-1,1)
                     r2X    = np.hstack((r2X,r2X_))
                     r2Xpv  = np.hstack((r2Xpv,r2Xpv_))
       
                     r2Y_   = 1-np.sum(Y_**2)/TSSY
                     r2Ypv_ = 1-np.sum(Y_**2,axis=0)/TSSYpv
                     r2Ypv_ = r2Ypv_.reshape(-1,1)
                     r2Y    = np.hstack((r2Y,r2Y_))
                     r2Ypv  = np.hstack((r2Ypv,r2Ypv_))
                     
                     r2R_   = 1-np.sum(R_**2)/TSSR
                     r2Rpv_ = 1-np.sum(R_**2,axis=0)/TSSRpv
                     r2Rpv_ = r2Rpv_.reshape(-1,1)
                     r2R    = np.hstack((r2R,r2R_))
                     r2Rpv  = np.hstack((r2Rpv,r2Rpv_))
             else:
                 num_it = num_it + 1
                 ui = un
            
        if a==0:
            numIT=num_it
        else:
            numIT=np.hstack((numIT,num_it))
    Xhat=Xhat*np.tile(x_std,(Xhat.shape[0],1))+np.tile(x_mean,(Xhat.shape[0],1) )      
    for a in list(range(A-1,0,-1)):
        r2X[a]     = r2X[a]-r2X[a-1]
        r2Xpv[:,a] = r2Xpv[:,a]-r2Xpv[:,a-1]
        r2Y[a]     = r2Y[a]-r2Y[a-1]
        r2Ypv[:,a] = r2Ypv[:,a]-r2Ypv[:,a-1]
        r2R[a]     = r2R[a]-r2R[a-1]
        r2Rpv[:,a] = r2Rpv[:,a]-r2Rpv[:,a-1]
    
    eigs = np.var(T,axis=0);
    r2xc = np.cumsum(r2X)
    r2yc = np.cumsum(r2Y)
    r2rc = np.cumsum(r2R)
    if not(shush):
        print('--------------------------------------------------------------')
        print('LV #     Eig       R2X       sum(R2X)   R2R       sum(R2R)   R2Y       sum(R2Y)')
        if A>1:    
            for a in list(range(A)):
                print("LV #"+str(a+1)+":   {:6.3f}    {:.3f}     {:.3f}      {:.3f}     {:.3f}      {:.3f}     {:.3f}".format(eigs[a], r2X[a], r2xc[a],r2R[a],r2rc[a],r2Y[a],r2yc[a]))
        else:
           d1=eigs[0]
           d2=r2xc[0]
           d3=r2rc[0]
           d4=r2yc[0]
           print("LV #"+str(a+1)+":   {:6.3f}    {:.3f}     {:.3f}      {:.3f}     {:.3f}      {:.3f}     {:.3f}".format(d1, r2X, d2,r2R,d3,r2Y,d4))
        print('--------------------------------------------------------------')   
        
    lpls_obj={'T':T,'P':P,'Q':Q,'U':U,'S':S,'H':H,'V':V,'Rscores':Rscores,
              'r2x':r2X,'r2xpv':r2Xpv,'mx':x_mean,'sx':x_std,
              'r2y':r2Y,'r2ypv':r2Ypv,'my':y_mean,'sy':y_std,
              'r2r':r2R,'r2rpv':r2Rpv,'mr':r_mean,'sr':r_std,
              'Xhat':Xhat}  
    if not isinstance(obsidX,bool):
        lpls_obj['obsidX']=obsidX
        lpls_obj['varidX']=varidX
    if not isinstance(obsidY,bool):
       lpls_obj['obsidY']=obsidY
       lpls_obj['varidY']=varidY    
    if not isinstance(obsidR,bool):
       lpls_obj['obsidR']=obsidR
       lpls_obj['varidR']=varidR  
          
    T2 = hott2(lpls_obj,Tnew=T)
    n  = T.shape[0]
    T2_lim99 = (((n-1)*(n+1)*A)/(n*(n-A)))*f99(A,(n-A))
    T2_lim95 = (((n-1)*(n+1)*A)/(n*(n-A)))*f95(A,(n-A))     
    speX = np.sum(X_**2,axis=1,keepdims=1)
    speX_lim95,speX_lim99 = spe_ci(speX)
    speY = np.sum(Y_**2,axis=1,keepdims=1)
    speY_lim95,speY_lim99 = spe_ci(speY)
    speR = np.sum(R_**2,axis=1,keepdims=1)
    speR_lim95,speR_lim99 = spe_ci(speR)
    
    lpls_obj['T2']          = T2
    lpls_obj['T2_lim99']    = T2_lim99
    lpls_obj['T2_lim95']    = T2_lim95
    lpls_obj['speX']        = speX
    lpls_obj['speX_lim99']  = speX_lim99
    lpls_obj['speX_lim95']  = speX_lim95
    lpls_obj['speY']        = speY
    lpls_obj['speY_lim99']  = speY_lim99
    lpls_obj['speY_lim95']  = speY_lim95
    lpls_obj['speR']        = speR
    lpls_obj['speR_lim99']  = speR_lim99
    lpls_obj['speR_lim95']  = speR_lim95
    
    lpls_obj['Ss'] = S @ np.linalg.pinv(V.T @ S) #trick to plot the LPLS does not really have Ws
    #Ws=W @ np.linalg.pinv(P.T @ W)
    lpls_obj['type']='lpls'
    return lpls_obj       
    
def lpls_pred(rnew,lpls_obj):
    '''
    Do a prediction with an LPLS model
    INPUTS:
        rnew: np.array, list or dataframe with elements of rew
              if multiple rows are passed, then multiple predictions are done
        
        lpls_obj: LPLS object built with pyphi.lpls routine
    
    OUTPUTS:
        pred: A dictionary {'Tnew':tnew,'Yhat':yhat,'speR':sper}   

    '''
    if isinstance(rnew,np.ndarray):
        rnew__=[rnew.copy()]
    elif isinstance(rnew,list):
        rnew__=np.array(rnew)
    elif isinstance(rnew,pd.DataFrame):
        rnew__=rnew.values[:,1:].astype(float)
    tnew=[]
    sper=[]
    for rnew_ in rnew__:
        rnew_=(rnew_-lpls_obj['mr'])/lpls_obj['sr']
        rnew_=rnew_.reshape(-1,1)
        ti=[]
        for a in np.arange(lpls_obj['T'].shape[1]):
            ti_ =( rnew_.T@lpls_obj['Rscores'][:,a]/             
                 (lpls_obj['Rscores'][:,a].T@lpls_obj['Rscores'][:,a]))
            ti.append(ti_[0])
            aux=ti_*lpls_obj['P'][:,a]
            rnew_=rnew_-aux.reshape(-1,1)
        tnew.append(np.array(ti))
        sper.append(np.sum(rnew_**2))
    tnew=np.array(tnew)
    sper=np.array(sper)
    yhat = tnew@lpls_obj['Q'].T
    yhat = (yhat * lpls_obj['sy'])+lpls_obj['my']
    preds ={'Tnew':tnew,'Yhat':yhat,'speR':sper}     
    return preds

def jrpls(Xi,Ri,Y,A,*,shush=False):
    '''
    JRPLS Algorithm per Garcia-Munoz Chemom.Intel.Lab.Syst., 133, pp.49-62.
    
     X =  Phys. Prop. dictionary of Dataframes of materials_i x mat. properties
         X = {'MatA':df_with_props_for_mat_A (one row per lot of MatA, one col per property),
              'MatB':df_with_props_for_mat_B (one row per lot of MatB, one col per property)}
         
     R = Blending ratios dictionary of Dataframes of  blends x materials_i
         R = {'MatA': df_with_ratios_of_lots_of_A_used_per_blend,
              'MatB': df_with_ratios_of_lots_of_B_used_per_blend,
              } 
     Rows of X[i] must correspond to Columns of R[i] 
         
     Y = [ b x n ]   Product characteristics dataframe of blends x prod. properties
     
     first column of all dataframes is the observation identifier
     
    '''
    X=[]
    varidX=[]
    obsidX=[]
    materials=list(Xi.keys())
    for k in Xi.keys():
        Xaux=Xi[k]   
        if isinstance(Xaux,np.ndarray):
            X_ = Xaux.copy()
            obsidX_ = False
            varidX_ = False
        elif isinstance(Xaux,pd.DataFrame):
            X_=np.array(Xaux.values[:,1:]).astype(float)
            obsidX_ = Xaux.values[:,0].astype(str)
            obsidX_ = obsidX_.tolist()
            varidX_ = Xaux.columns.values
            varidX_ = varidX_[1:]
            varidX_ = varidX_.tolist()
        X.append(X_)
        varidX.append(varidX_)
        obsidX.append(obsidX_)
        
    if isinstance(Y,np.ndarray):
        Y_=Y.copy()
        obsidY = False
        varidY = False
    elif isinstance(Y,pd.DataFrame):
        Y_=np.array(Y.values[:,1:]).astype(float)
        obsidY = Y.values[:,0].astype(str)
        obsidY = obsidY.tolist()
        varidY = Y.columns.values
        varidY = varidY[1:]
        varidY = varidY.tolist()    
 
    R=[]
    varidR=[]
    obsidR=[]
    for k in materials:  
        Raux=Ri[k] 
        if isinstance(Raux,np.ndarray):
            R_=Raux.copy()
            obsidR_ = False
            varidR_ = False
        elif isinstance(Raux,pd.DataFrame):
            R_=np.array(Raux.values[:,1:]).astype(float)
            obsidR_ = Raux.values[:,0].astype(str)
            obsidR_ = obsidR_.tolist()
            varidR_ = Raux.columns.values
            varidR_ = varidR_[1:]
            varidR_ = varidR_.tolist()
        varidR.append(varidR_)
        obsidR.append(obsidR_)
        R.append(R_)
        
    x_mean    = []
    x_std     = []
    jr_scale  = []
    r_mean    = []
    r_std     = []
    not_Xmiss = []
    not_Rmiss = []
    Xhat      = []
    TSSX      = []
    TSSXpv    = []
    TSSR      = []
    TSSRpv    = []
    X__=[]
    R__=[]
    for X_i,R_i in zip (X,R):
        X_, x_mean_, x_std_ = meancenterscale(X_i)
        R_, r_mean_, r_std_ = meancenterscale(R_i)
        
        jr_scale_=np.sqrt(X_.shape[0]*X_.shape[1])
        jr_scale_=np.sqrt(X_.shape[1])
        X_ = X_ / jr_scale_
        
        x_mean.append( x_mean_ )
        x_std.append(  x_std_  )
        jr_scale.append( jr_scale_)
        r_mean.append( r_mean_ )
        r_std.append(  r_std_  )
        
        X_nan_map  = np.isnan(X_)
        not_Xmiss_ = (np.logical_not(X_nan_map))*1
        R_nan_map  = np.isnan(R_)
        not_Rmiss_ = (np.logical_not(R_nan_map))*1
        not_Xmiss.append(not_Xmiss_)
        not_Rmiss.append(not_Rmiss_)
        
        X_,dummy=n2z(X_)
        R_,dummy=n2z(R_)
        Xhat_ = np.zeros(X_.shape)
        X__.append(X_)
        R__.append(R_)
        Xhat.append(Xhat_)
        
        TSSX.append(   np.sum(X_**2)       )
        TSSXpv.append( np.sum(X_**2,axis=0))
        TSSR.append(   np.sum(R_**2)       )
        TSSRpv.append( np.sum(R_**2,axis=0))
       
    X=X__.copy()
    R=R__.copy()
    
    Y_,y_mean,y_std = meancenterscale(Y_)
    Y_nan_map = np.isnan(Y_)
    not_Ymiss = (np.logical_not(Y_nan_map))*1
    Y_,dummy=n2z(Y_)
    TSSY   = np.sum(Y_**2)
    TSSYpv = np.sum(Y_**2,axis=0)
    
    #use nipals
    if not(shush):
        print('phi.jrpls using NIPALS executed on: '+ str(datetime.datetime.now()) )

    epsilon=1E-9
    maxit=2000    

    for a in list(range(A)):
        # Select column with largest variance in Y as initial guess
        ui = Y_[:,[np.argmax(std(Y_))]]
        Converged=False
        num_it=0
        while Converged==False:
            
     # _Ab_btbinv(A,b,A_not_nan_map):
     #  project c = Ab/b'b

             #Step 1. h=R'u/u'u
             hi=[]
             for i,R_ in enumerate(R):
                 hi_ = _Ab_btbinv(R_.T,ui,not_Rmiss[i].T)
                 hi.append(hi_)                 
             #print('step 1')
             si=[]
             for i,X_ in enumerate(X):
                 #Step 2. s = X'h/(h'h) 
                 si_ = _Ab_btbinv(X_.T, hi[i],not_Xmiss[i].T)
                 si.append(si_)
             #print('step 2')
             
             #Normalize joint s to unit length.
             js=np.array([y for x in si for y in x])  #flattening list of lists
             for i in np.arange(len(si)):
                 si[i]=si[i]/np.linalg.norm(js)      

             ri=[]
             for i,X_ in enumerate(X):
                 #Step 3. ri= (Xs)/(s's)
                 ri_ = _Ab_btbinv(X_,si[i],not_Xmiss[i])
                 ri.append(ri_)
             #print('step 3')

             #Calculating the Joint-r and Joint-R (hence the name of the method)
             jr=[y for x in ri for y in x]
             jr=np.array(jr).astype(float)
             
             for i,r_ in enumerate(R):
                 if i==0:
                     R_=r_
                 else:
                     R_=np.hstack((R_,r_))
                     
             for i,r_miss in enumerate(not_Rmiss):
                if i==0:
                    not_Rmiss_=r_miss
                else:
                    not_Rmiss_=np.hstack((not_Rmiss_,r_miss))
                             
             #Step 4. t = Rr/(r'r)
             ti = _Ab_btbinv(R_,jr,not_Rmiss_)
             #print('step 4')
             
             #Step 5 q=Y't/t't
             qi = _Ab_btbinv(Y_.T,ti,not_Ymiss.T)
             
             #Step 5 un=(Yq)/(q'q)
             un = _Ab_btbinv(Y_,qi,not_Ymiss)
             #print('step 5')
             
             if abs((np.linalg.norm(ui)-np.linalg.norm(un)))/(np.linalg.norm(ui)) < epsilon:
                 Converged=True
                 
             if num_it > maxit:
                 Converged=True
                 
             if Converged:
                 if not(shush):
                     print('# Iterations for LV #'+str(a+1)+': ',str(num_it))
                 pi=[]
                 for i,R_ in enumerate(R):
                     # Calculate P's for deflation p=R't/(t't) 
                     pi_ = _Ab_btbinv(R_.T,ti,not_Rmiss[i].T)
                     pi.append(pi_)
                 vi=[]
                 for i,X_ in enumerate(X):
                     # Calculate v's for deflation v=Xr/(r'r) 
                     vi_ = _Ab_btbinv(X_.T,ri[i],not_Xmiss[i].T)
                     vi.append(vi_)
                 
                 for i in np.arange(len(R)):    
                     # Deflate X leaving missing as zeros (important!)
                     R[i]=(R[i]- ti @ pi[i].T)*not_Rmiss[i]
                     X[i]=(X[i]- ri[i] @ vi[i].T)*not_Xmiss[i]
                     Xhat[i] = Xhat[i] + ri[i] @ vi[i].T
                 Y_=(Y_- ti @ qi.T)*not_Ymiss
                 
                 
                 #print(R_.shape)
                 #print(X_.shape)
                 #print(Y_.shape)
                 if a==0:
                     T=ti
                     P=pi
                     S=si
                     U=un
                     Q=qi
                     H=hi
                     V=vi
                     Rscores = ri
                     
                     r2X=[]
                     r2Xpv=[]
                     r2R=[]
                     r2Rpv=[]
                     for i in np.arange(len(X)):
                         r2X.append( 1-np.sum(X[i]**2)/TSSX[i])
                         r2Xpv_ = 1-np.sum(X[i]**2,axis=0)/TSSXpv[i]
                         r2Xpv.append( r2Xpv_.reshape(-1,1))
                         r2R.append( 1-np.sum(R[i]**2)/TSSR[i])
                         r2Rpv_ = 1-np.sum(R[i]**2,axis=0)/TSSRpv[i]
                         r2Rpv.append( r2Rpv_.reshape(-1,1))
                                          
                     r2Y   = 1-np.sum(Y_**2)/TSSY
                     r2Ypv = 1-np.sum(Y_**2,axis=0)/TSSYpv
                     r2Ypv = r2Ypv.reshape(-1,1)
                     

                                 
                 else:
                     T=np.hstack((T,ti.reshape(-1,1)))
                     U=np.hstack((U,un.reshape(-1,1)))
                     Q=np.hstack((Q,qi))
                     
                     for i in np.arange(len(P)):
                         P[i]=np.hstack((P[i],pi[i])) 
                     for i in np.arange(len(S)):    
                         S[i]=np.hstack((S[i],si[i]))
                     for i in np.arange(len(H)):    
                         H[i]=np.hstack((H[i],hi[i]))
                     for i in np.arange(len(V)):    
                         V[i]=np.hstack((V[i],vi[i]))
                     for i in np.arange(len(Rscores)):    
                         Rscores[i]=np.hstack((Rscores[i],ri[i] ))

                     for i in np.arange(len(X)):   
                         r2X_   = 1-np.sum(X[i]**2)/TSSX[i]
                         r2Xpv_ = 1-np.sum(X[i]**2,axis=0)/TSSXpv[i]
                         r2Xpv_ = r2Xpv_.reshape(-1,1)
                         r2X[i]    = np.hstack((r2X[i],r2X_))
                         r2Xpv[i]  = np.hstack((r2Xpv[i],r2Xpv_))
                         
                         r2R_   = 1-np.sum(R[i]**2)/TSSR[i]
                         r2Rpv_ = 1-np.sum(R[i]**2,axis=0)/TSSRpv[i]
                         r2Rpv_ = r2Rpv_.reshape(-1,1)
                         r2R[i]    = np.hstack((r2R[i],r2R_))
                         r2Rpv[i]  = np.hstack((r2Rpv[i],r2Rpv_))
                         
       
                     r2Y_   = 1-np.sum(Y_**2)/TSSY
                     r2Ypv_ = 1-np.sum(Y_**2,axis=0)/TSSYpv
                     r2Ypv_ = r2Ypv_.reshape(-1,1)
                     r2Y    = np.hstack((r2Y,r2Y_))
                     r2Ypv  = np.hstack((r2Ypv,r2Ypv_))
                     

             else:
                 num_it = num_it + 1
                 ui = un
            
        if a==0:
            numIT=num_it
        else:
            numIT=np.hstack((numIT,num_it))
    for i in np.arange(len(Xhat)):        
        Xhat[i]=Xhat[i]*np.tile(x_std[i],(Xhat[i].shape[0],1))+np.tile(x_mean[i],(Xhat[i].shape[0],1) )      
 
    for a in list(range(A-1,0,-1)):
        r2Y[a]     = r2Y[a]-r2Y[a-1]
        r2Ypv[:,a] = r2Ypv[:,a]-r2Ypv[:,a-1]
    r2xc=[]
    r2rc=[]
    for i in np.arange(len(X)):    
        for a in list(range(A-1,0,-1)):
            r2X[i][a]     = r2X[i][a]-r2X[i][a-1]
            r2Xpv[i][:,a] = r2Xpv[i][:,a]-r2Xpv[i][:,a-1]
            r2R[i][a]     = r2R[i][a]-r2R[i][a-1]
            r2Rpv[i][:,a] = r2Rpv[i][:,a]-r2Rpv[i][:,a-1]
    
    for i,r in enumerate(r2Xpv):
        if i==0:
           r2xpv_all= r
        else:
            r2xpv_all=np.vstack((r2xpv_all,r))

    
        r2xc.append(np.cumsum(r2X[i]))
        r2rc.append(np.cumsum(r2R[i]))
    
    eigs = np.var(T,axis=0);
    r2yc = np.cumsum(r2Y)
    r2rc = np.mean(np.array(r2rc),axis=0)
    r2xc = np.mean(np.array(r2xc),axis=0)
    r2x  = np.mean(np.array(r2X),axis=0 ) 
    r2r  = np.mean(np.array(r2R),axis=0 ) 
    
    if not(shush):
        print('--------------------------------------------------------------')
        print('LV #     Eig       R2X       sum(R2X)   R2R       sum(R2R)   R2Y       sum(R2Y)')
        if A>1:    
            for a in list(range(A)):
                print("LV #"+str(a+1)+":   {:6.3f}    {:.3f}     {:.3f}      {:.3f}     {:.3f}      {:.3f}     {:.3f}".format(eigs[a], r2x[a], r2xc[a],r2r[a],r2rc[a],r2Y[a],r2yc[a]))
        else:
           d1=eigs[0]
           d2=r2xc[0]
           d3=r2rc[0]
           d4=r2yc[0]
           print("LV #"+str(a+1)+":   {:6.3f}    {:.3f}     {:.3f}      {:.3f}     {:.3f}      {:.3f}     {:.3f}".format(d1, r2x, d2,r2r,d3,r2Y,d4))
        print('--------------------------------------------------------------')   
        
    jrpls_obj={'T':T,'P':P,'Q':Q,'U':U,'S':S,'H':H,'V':V,'Rscores':Rscores,
              'r2xi':r2X,'r2xpvi':r2Xpv,'r2xpv':r2xpv_all,
              'mx':x_mean,'sx':x_std,
              'r2y':r2Y,'r2ypv':r2Ypv,'my':y_mean,'sy':y_std,
              'r2ri':r2R,'r2rpvi':r2Rpv,'mr':r_mean,'sr':r_std,
              'Xhat':Xhat,'materials':materials}  
    if not isinstance(obsidX,bool):
        jrpls_obj['obsidXi']=obsidX
        jrpls_obj['varidXi']=varidX
    
    varidXall=[]
    for i in np.arange(len(materials)):
        for j in np.arange(len( varidX[i])):
            varidXall.append( materials[i]+':'+varidX[i][j])
    jrpls_obj['varidX']=varidXall    
    
    if not isinstance(obsidR,bool):
       jrpls_obj['obsidRi']=obsidR
       jrpls_obj['varidRi']=varidR  
    
    if not isinstance(obsidY,bool):
       jrpls_obj['obsidY']=obsidY
       jrpls_obj['varidY']=varidY    
       
    T2 = hott2(jrpls_obj,Tnew=T)
    n  = T.shape[0]
    T2_lim99 = (((n-1)*(n+1)*A)/(n*(n-A)))*f99(A,(n-A))
    T2_lim95 = (((n-1)*(n+1)*A)/(n*(n-A)))*f95(A,(n-A))    
    
    speX=[]
    speR=[]
    speX_lim95=[]
    speX_lim99=[]
    speR_lim95=[]
    speR_lim99=[]
    for i in np.arange(len(X)):
        speX .append(  np.sum(X[i]**2,axis=1,keepdims=1))
        aux_=np.sum(X[i]**2,axis=1,keepdims=1)
        speX_lim95_,speX_lim99_ = spe_ci(aux_)
        speX_lim95.append(speX_lim95_)
        speX_lim99.append(speX_lim99_)
        
        speR.append( np.sum(R[i]**2,axis=1,keepdims=1))
        aux_=np.sum(R[i]**2,axis=1,keepdims=1)
        speR_lim95_,speR_lim99_ = spe_ci(aux_)
        speR_lim95.append(speR_lim95_)
        speR_lim99.append(speR_lim99_)
        
    speY = np.sum(Y_**2,axis=1,keepdims=1)
    speY_lim95,speY_lim99 = spe_ci(speY)
    
    jrpls_obj['T2']          = T2
    jrpls_obj['T2_lim99']    = T2_lim99
    jrpls_obj['T2_lim95']    = T2_lim95
    jrpls_obj['speX']        = speX
    jrpls_obj['speX_lim99']  = speX_lim99
    jrpls_obj['speX_lim95']  = speX_lim95
    jrpls_obj['speY']        = speY
    jrpls_obj['speY_lim99']  = speY_lim99
    jrpls_obj['speY_lim95']  = speY_lim95
    jrpls_obj['speR']        = speR
    jrpls_obj['speR_lim99']  = speR_lim99
    jrpls_obj['speR_lim95']  = speR_lim95
    
    Wsi= []
    Ws = []
    for i in np.arange(len(S)):
        Wsi.append (S[i] @ np.linalg.pinv(V[i].T @ S[i]) )
        if i==0:
            Ws=S[i] @ np.linalg.pinv(V[i].T @ S[i])
        else:
            Ws=np.vstack((Ws,S[i] @ np.linalg.pinv(V[i].T @ S[i])))    
    jrpls_obj['Ssi'] = Wsi #trick to plot the JRPLS/LPLS does not really have Ws
    jrpls_obj['Ss'] = Ws #trick to plot the JRPLS/LPLS does not really have Ws
    #Ws=W @ np.linalg.pinv(P.T @ W)
    jrpls_obj['type']='jrpls'
    return jrpls_obj       

     
def jrpls_pred(rnew,jrplsobj):
    '''
    Routine to produce the prediction for a new observation of Ri
    using a JRPLS model
    preds = (rnew,jrplsobj)
    
    Inputs:
        rnew: A dictionary with the format: 
            rnew={ 
                'matid':[(lotid,rvalue )],
                
                }
            
            for example, a prediction for the scenario:
                
        material    lot to use  rvalue       
        API	        A0129	    0.5
        Lactose	    Lac0003   	0.1
        Lactose     Lac1010     0.2
        MgSt	        M0012	    0.02
        MCC      	MCC0017	    0.18
        
        use:
            
        rnew={
            'API':[('A0129',0.5)],
            'Lactose':[('Lac0003',0.1 ),('Lac1010',0.2 )],
            'MgSt':[('M0012',0.02)],
            'MCC':[('MCC0017',0.18)],
            }    
        
    Outputs:
        preds a dictionary of the form:
            
            preds ={'Tnew':tnew,'Yhat':yhat,'speR':sper} 
            
            where speR has the speR per each material

    '''
    
    ok=True
    if isinstance(rnew,list):
        #check dimensions
        i=0     
        for r,mr,sr in zip(rnew,jrplsobj['mr'],jrplsobj['sr']):
            if not(len(r)==len(mr[0])):
                ok=False
            np.ones(len(r))
            if i==0:
                rnew_=r
                mr_=mr
                sr_=sr
                Rscores = jrplsobj['Rscores'][i]
                P       = jrplsobj['P'][i]
            else:
                rnew_   = np.hstack((rnew_,r))    
                mr_     = np.hstack((mr_,mr))
                sr_     = np.hstack((sr_,sr))
                Rscores = np.vstack((Rscores,jrplsobj['Rscores'][i] ))
                P       = np.vstack((P,jrplsobj['P'][i] ))
            i+=1
                
    elif isinstance(rnew,dict):
        #re-arrange 
        ok=True
        rnew_=[['*']]*len(jrplsobj['materials'])
        for k in list(rnew.keys()):
            i = jrplsobj['materials'].index(k)
            ri=np.zeros((jrplsobj['mr'][i].shape[1]) )
            for m,r in rnew[k]:
                e = jrplsobj['varidRi'][i].index(m)
                ri[e]=r
            rnew_[i]=ri
        
        preds=jrpls_pred(rnew_,jrplsobj)
        return preds
         
    if ok:  
        bkzeros=0
        selmat=[]
        for i,r in enumerate(jrplsobj['Rscores']):
            frontzeros=Rscores.shape[0]-bkzeros-r.shape[0]
            row=np.vstack((np.zeros((bkzeros,1)),
                           np.ones((r.shape[0],1)),
                           np.zeros(( frontzeros,1))))
            bkzeros+=r.shape[0]
            selmat.append(row)
            
        tnew=[]
        sper=[]
       
        rnew_=(rnew_-mr_)/sr_
        rnew_=rnew_.reshape(-1,1)
        ti=[]
        for a in np.arange(jrplsobj['T'].shape[1]):

            ti_ =( rnew_.T@Rscores[:,a]/             
                 (Rscores[:,a].T@Rscores[:,a]))
            ti.append(ti_[0])
            aux=ti_*P[:,a]
            rnew_=rnew_-aux.reshape(-1,1)
        tnew=np.array(ti)
        sper=[]
        for row in selmat:
            sper.append(np.sum(rnew_[row==1]**2))
        
        yhat = tnew@jrplsobj['Q'].T
        yhat = (yhat * jrplsobj['sy'])+jrplsobj['my']
        preds ={'Tnew':tnew,'Yhat':yhat,'speR':sper}     
        return preds    
    else:
        return 'dimensions of rnew did not macth model'
    
def tpls(Xi,Ri,Z,Y,A,*,shush=False):
    '''
     TPLS Algorithm per Garcia-Munoz Chemom.Intel.Lab.Syst., 133, pp.49-62.
    
     X = Phys. Prop. dictionary of Dataframes of materials_i x mat. properties
         X = {'MatA':df_with_props_for_mat_A (one row per lot of MatA, one col per property),
              'MatB':df_with_props_for_mat_B (one row per lot of MatB, one col per property)}
         
     R =  Blending ratios dictionary of Dataframes of  blends x materials_i
         R = {'MatA': df_with_ratios_of_lots_of_A_used_per_blend,
              'MatB': df_with_ratios_of_lots_of_B_used_per_blend,
              } 
     Rows of X[i] must correspond to Columns of R[i] 
         
     Y = [ b x n ]   Product characteristics dataframe of blends x prod. properties
     
     Z = [b x p]  Process conditions dataframe of  blends x process variables
     
     first column of all dataframes is the observation identifier
     
    '''
    X=[]
    varidX=[]
    obsidX=[]
    materials=list(Xi.keys())
    for k in Xi.keys():
        Xaux=Xi[k]   
        if isinstance(Xaux,np.ndarray):
            X_ = Xaux.copy()
            obsidX_ = False
            varidX_ = False
        elif isinstance(Xaux,pd.DataFrame):
            X_=np.array(Xaux.values[:,1:]).astype(float)
            obsidX_ = Xaux.values[:,0].astype(str)
            obsidX_ = obsidX_.tolist()
            varidX_ = Xaux.columns.values
            varidX_ = varidX_[1:]
            varidX_ = varidX_.tolist()
        X.append(X_)
        varidX.append(varidX_)
        obsidX.append(obsidX_)
        
    if isinstance(Y,np.ndarray):
        Y_=Y.copy()
        obsidY = False
        varidY = False
    elif isinstance(Y,pd.DataFrame):
        Y_=np.array(Y.values[:,1:]).astype(float)
        obsidY = Y.values[:,0].astype(str)
        obsidY = obsidY.tolist()
        varidY = Y.columns.values
        varidY = varidY[1:]
        varidY = varidY.tolist()    

    if isinstance(Z,np.ndarray):
        Z_=Z.copy()
        obsidZ = False
        varidZ = False
    elif isinstance(Z,pd.DataFrame):
        Z_=np.array(Z.values[:,1:]).astype(float)
        obsidZ = Z.values[:,0].astype(str)
        obsidZ = obsidZ.tolist()
        varidZ = Z.columns.values
        varidZ = varidZ[1:]
        varidZ = varidZ.tolist()      
    
    R=[]
    varidR=[]
    obsidR=[]
    for k in materials:  
        Raux=Ri[k] 
        if isinstance(Raux,np.ndarray):
            R_=Raux.copy()
            obsidR_ = False
            varidR_ = False
        elif isinstance(Raux,pd.DataFrame):
            R_=np.array(Raux.values[:,1:]).astype(float)
            obsidR_ = Raux.values[:,0].astype(str)
            obsidR_ = obsidR_.tolist()
            varidR_ = Raux.columns.values
            varidR_ = varidR_[1:]
            varidR_ = varidR_.tolist()
        varidR.append(varidR_)
        obsidR.append(obsidR_)
        R.append(R_)
        
    x_mean    = []
    x_std     = []
    jr_scale  = []
    r_mean    = []
    r_std     = []
    not_Xmiss = []
    not_Rmiss = []
    Xhat      = []
    TSSX      = []
    TSSXpv    = []
    TSSR      = []
    TSSRpv    = []
    X__=[]
    R__=[]
    for X_i,R_i in zip (X,R):
        X_, x_mean_, x_std_ = meancenterscale(X_i)
        R_, r_mean_, r_std_ = meancenterscale(R_i)
        
        jr_scale_=np.sqrt(X_.shape[0]*X_.shape[1])
        jr_scale_=np.sqrt(X_.shape[1])
        X_ = X_ / jr_scale_
        
        x_mean.append( x_mean_ )
        x_std.append(  x_std_  )
        jr_scale.append( jr_scale_)
        r_mean.append( r_mean_ )
        r_std.append(  r_std_  )
        
        X_nan_map  = np.isnan(X_)
        not_Xmiss_ = (np.logical_not(X_nan_map))*1
        R_nan_map  = np.isnan(R_)
        not_Rmiss_ = (np.logical_not(R_nan_map))*1
        not_Xmiss.append(not_Xmiss_)
        not_Rmiss.append(not_Rmiss_)
        
        X_,dummy=n2z(X_)
        R_,dummy=n2z(R_)
        Xhat_ = np.zeros(X_.shape)
        X__.append(X_)
        R__.append(R_)
        Xhat.append(Xhat_)
        
        TSSX.append(   np.sum(X_**2)       )
        TSSXpv.append( np.sum(X_**2,axis=0))
        TSSR.append(   np.sum(R_**2)       )
        TSSRpv.append( np.sum(R_**2,axis=0))
       
    X=X__.copy()
    R=R__.copy()
    
    Y_,y_mean,y_std = meancenterscale(Y_)
    Y_nan_map = np.isnan(Y_)
    not_Ymiss = (np.logical_not(Y_nan_map))*1
    Y_,dummy=n2z(Y_)
    TSSY   = np.sum(Y_**2)
    TSSYpv = np.sum(Y_**2,axis=0)
    
    Z_,z_mean,z_std = meancenterscale(Z_)
    Z_nan_map = np.isnan(Z_)
    not_Zmiss = (np.logical_not(Z_nan_map))*1
    Z_,dummy=n2z(Z_)
    TSSZ   = np.sum(Z_**2)
    TSSZpv = np.sum(Z_**2,axis=0)
    
    #use nipals
    if not(shush):
        print('phi.tpls using NIPALS executed on: '+ str(datetime.datetime.now()) )

    epsilon=1E-9
    maxit=2000    

    for a in list(range(A)):
        # Select column with largest variance in Y as initial guess
        ui = Y_[:,[np.argmax(std(Y_))]]
        Converged=False
        num_it=0
        while Converged==False:
            
     # _Ab_btbinv(A,b,A_not_nan_map):
     #  project c = Ab/b'b

             #Step 2. h=R'u/u'u
             hi=[]
             for i,R_ in enumerate(R):
                 hi_ = _Ab_btbinv(R_.T,ui,not_Rmiss[i].T)
                 hi.append(hi_)                 
             #print('step 2')
             si=[]
             for i,X_ in enumerate(X):
                 #Step 3. s = X'h/(h'h) 
                 si_ = _Ab_btbinv(X_.T, hi[i],not_Xmiss[i].T)
                 si.append(si_)
             #print('step 3')
             
             #Step 4 Normalize joint s to unit length.
             js=np.array([y for x in si for y in x])  #flattening list of lists
             for i in np.arange(len(si)):
                 si[i]=si[i]/np.linalg.norm(js)      

             #Step 5
             ri=[]
             for i,X_ in enumerate(X):
                 #Step 5. ri= (Xs)/(s's)
                 ri_ = _Ab_btbinv(X_,si[i],not_Xmiss[i])
                 ri.append(ri_)
             
             #Step 6  
             #Calculating the Joint-r and Joint-R (hence the name of the method)
             jr=[y for x in ri for y in x]
             jr=np.array(jr).astype(float)
             
             for i,r_ in enumerate(R):
                 if i==0:
                     R_=r_
                 else:
                     R_=np.hstack((R_,r_))
                     
             for i,r_miss in enumerate(not_Rmiss):
                if i==0:
                    not_Rmiss_=r_miss
                else:
                    not_Rmiss_=np.hstack((not_Rmiss_,r_miss))
                             
             t_rx = _Ab_btbinv(R_,jr,not_Rmiss_)
             #print('step 6')
             
             #Step 7
             #Now the process matrix
             wi = _Ab_btbinv(Z_.T,ui,not_Zmiss.T)
             
             #Step 8
             wi = wi / np.linalg.norm(wi)
        
             #Step 9
             t_z =  _Ab_btbinv(Z_,wi,not_Zmiss)
             
             #Step 10
             Taux=np.hstack((t_rx,t_z))
             plsobj_=pls(Taux, Y_,1,mcsX=False,mcsY=False,shush=True,force_nipals=True)
             wt_i = plsobj_['W']
             qi   = plsobj_['Q']
             un   = plsobj_['U']   
             ti   = plsobj_['T']
             
             
             if abs((np.linalg.norm(ui)-np.linalg.norm(un)))/(np.linalg.norm(ui)) < epsilon:
                 Converged=True
                 
             if num_it > maxit:
                 Converged=True
                 
             if Converged:
                 if not(shush):
                     print('# Iterations for LV #'+str(a+1)+': ',str(num_it))
                 pi=[]
                 for i,R_ in enumerate(R):
                     # Calculate P's for deflation p=R't/(t't) 
                     pi_ = _Ab_btbinv(R_.T,ti,not_Rmiss[i].T)
                     pi.append(pi_)
                 vi=[]
                 for i,X_ in enumerate(X):
                     # Calculate v's for deflation v=Xr/(r'r) 
                     vi_ = _Ab_btbinv(X_.T,ri[i],not_Xmiss[i].T)
                     vi.append(vi_)
                 
                 pzi = _Ab_btbinv(Z_.T,ti,not_Zmiss.T)
                 
                 for i in np.arange(len(R)):    
                     # Deflate X leaving missing as zeros (important!)
                     R[i]=(R[i]- ti @ pi[i].T)*not_Rmiss[i]
                     X[i]=(X[i]- ri[i] @ vi[i].T)*not_Xmiss[i]
                     Xhat[i] = Xhat[i] + ri[i] @ vi[i].T
                     
                 Y_=(Y_- ti @ qi.T)*not_Ymiss
                 Z_=(Z_- ti @ pzi.T)*not_Zmiss 
                 
                 #print(R_.shape)
                 #print(X_.shape)
                 #print(Y_.shape)
                 if a==0:
                     T=ti
                     P=pi
                     Pz=pzi
                     S=si
                     U=un
                     Q=qi
                     H=hi
                     V=vi
                     Rscores = ri
                     W=wi
                     Wt=wt_i
                     
                     r2X=[]
                     r2Xpv=[]
                     r2R=[]
                     r2Rpv=[]
                     for i in np.arange(len(X)):
                         r2X.append( 1-np.sum(X[i]**2)/TSSX[i])
                         r2Xpv_ = 1-np.sum(X[i]**2,axis=0)/TSSXpv[i]
                         r2Xpv.append( r2Xpv_.reshape(-1,1))
                         r2R.append( 1-np.sum(R[i]**2)/TSSR[i])
                         r2Rpv_ = 1-np.sum(R[i]**2,axis=0)/TSSRpv[i]
                         r2Rpv.append( r2Rpv_.reshape(-1,1))
                                          
                     r2Y   = 1-np.sum(Y_**2)/TSSY
                     r2Ypv = 1-np.sum(Y_**2,axis=0)/TSSYpv
                     r2Ypv = r2Ypv.reshape(-1,1)
                     
                     r2Z   = 1-np.sum(Z_**2)/TSSZ
                     r2Zpv = 1-np.sum(Z_**2,axis=0)/TSSZpv
                     r2Zpv = r2Zpv.reshape(-1,1)              
                 else:
                     T=np.hstack((T,ti.reshape(-1,1)))
                     U=np.hstack((U,un.reshape(-1,1)))
                     Q=np.hstack((Q,qi))
                     W=np.hstack((W,wi))
                     Wt=np.hstack((Wt,wt_i))
                     Pz=np.hstack((Pz,pzi ))
                     
                     for i in np.arange(len(P)):
                         P[i]=np.hstack((P[i],pi[i])) 
                     for i in np.arange(len(S)):    
                         S[i]=np.hstack((S[i],si[i]))
                     for i in np.arange(len(H)):    
                         H[i]=np.hstack((H[i],hi[i]))
                     for i in np.arange(len(V)):    
                         V[i]=np.hstack((V[i],vi[i]))
                     for i in np.arange(len(Rscores)):    
                         Rscores[i]=np.hstack((Rscores[i],ri[i] ))

                     for i in np.arange(len(X)):   
                         r2X_   = 1-np.sum(X[i]**2)/TSSX[i]
                         r2Xpv_ = 1-np.sum(X[i]**2,axis=0)/TSSXpv[i]
                         r2Xpv_ = r2Xpv_.reshape(-1,1)
                         r2X[i]    = np.hstack((r2X[i],r2X_))
                         r2Xpv[i]  = np.hstack((r2Xpv[i],r2Xpv_))
                         
                         r2R_   = 1-np.sum(R[i]**2)/TSSR[i]
                         r2Rpv_ = 1-np.sum(R[i]**2,axis=0)/TSSRpv[i]
                         r2Rpv_ = r2Rpv_.reshape(-1,1)
                         r2R[i]    = np.hstack((r2R[i],r2R_))
                         r2Rpv[i]  = np.hstack((r2Rpv[i],r2Rpv_))
                         
       
                     r2Y_   = 1-np.sum(Y_**2)/TSSY
                     r2Ypv_ = 1-np.sum(Y_**2,axis=0)/TSSYpv
                     r2Ypv_ = r2Ypv_.reshape(-1,1)
                     r2Y    = np.hstack((r2Y,r2Y_))
                     r2Ypv  = np.hstack((r2Ypv,r2Ypv_))
                     
                     r2Z_   = 1-np.sum(Z_**2)/TSSZ
                     r2Zpv_ = 1-np.sum(Z_**2,axis=0)/TSSZpv
                     r2Zpv_ = r2Zpv_.reshape(-1,1)  
                     r2Z    = np.hstack((r2Z,r2Z_))
                     r2Zpv  = np.hstack((r2Zpv,r2Zpv_))                     

             else:
                 num_it = num_it + 1
                 ui = un       
        if a==0:
            numIT=num_it
        else:
            numIT=np.hstack((numIT,num_it))
    for i in np.arange(len(Xhat)):        
        Xhat[i]=Xhat[i]*np.tile(x_std[i],(Xhat[i].shape[0],1))+np.tile(x_mean[i],(Xhat[i].shape[0],1) )      
 
    for a in list(range(A-1,0,-1)):
        r2Y[a]     = r2Y[a]-r2Y[a-1]
        r2Ypv[:,a] = r2Ypv[:,a]-r2Ypv[:,a-1]
        r2Z[a]     = r2Z[a]-r2Z[a-1]
        r2Zpv[:,a] = r2Zpv[:,a]-r2Zpv[:,a-1]

    r2xc=[]
    r2rc=[]
    for i in np.arange(len(X)):    
        for a in list(range(A-1,0,-1)):
            r2X[i][a]     = r2X[i][a]-r2X[i][a-1]
            r2Xpv[i][:,a] = r2Xpv[i][:,a]-r2Xpv[i][:,a-1]
            r2R[i][a]     = r2R[i][a]-r2R[i][a-1]
            r2Rpv[i][:,a] = r2Rpv[i][:,a]-r2Rpv[i][:,a-1]
    
    for i,r in enumerate(r2Xpv):
        if i==0:
           r2xpv_all= r
        else:
            r2xpv_all=np.vstack((r2xpv_all,r))

    
        r2xc.append(np.cumsum(r2X[i]))
        r2rc.append(np.cumsum(r2R[i]))
    

    r2yc = np.cumsum(r2Y)
    r2zc = np.cumsum(r2Z)
    r2rc = np.mean(np.array(r2rc),axis=0)
    r2xc = np.mean(np.array(r2xc),axis=0)
    r2x  = np.mean(np.array(r2X),axis=0 ) 
    r2r  = np.mean(np.array(r2R),axis=0 ) 
    
    if not(shush):
        print('--------------------------------------------------------------')
        print('LV #     R2X       sum(R2X)   R2R       sum(R2R)   R2Z       sum(R2Z)   R2Y       sum(R2Y)')
        if A>1:    
            for a in list(range(A)):
                print("LV #"+str(a+1)+":   {:.3f}     {:.3f}      {:.3f}     {:.3f}      {:.3f}     {:.3f}      {:.3f}     {:.3f}".format(r2x[a],r2xc[a],r2r[a],r2rc[a],r2Z[a],r2zc[a],r2Y[a],r2yc[a]))
        else:
         
           d1=r2xc[0]
           d2=r2rc[0]
           d3=r2zc[0]
           d4=r2yc[0]
           print("LV #"+str(a+1)+":   {:.3f}     {:.3f}      {:.3f}     {:.3f}      {:.3f}     {:.3f}      {:.3f}     {:.3f}".format(r2x, d1,r2r,d2,r2Z,d3,r2Y,d4))
        print('--------------------------------------------------------------')   
        
    tpls_obj={'T':T,'P':P,'Q':Q,'U':U,'S':S,'H':H,'V':V,'Rscores':Rscores,
              'r2xi':r2X,'r2xpvi':r2Xpv,'r2xpv':r2xpv_all,
              'mx':x_mean,'sx':x_std,
              'r2y':r2Y,'r2ypv':r2Ypv,
              'my':y_mean,'sy':y_std,
              'r2ri':r2R,'r2rpvi':r2Rpv,
              'mr':r_mean,'sr':r_std,
              'r2z':r2Z,'r2zpv':r2Zpv,
              'mz':z_mean,'sz':z_std,
              'Xhat':Xhat,'materials':materials,'Wt':Wt,'W':W,'Pz':Pz}  
    if not isinstance(obsidX,bool):
        tpls_obj['obsidXi']=obsidX
        tpls_obj['varidXi']=varidX
    
    varidXall=[]
    for i in np.arange(len(materials)):
        for j in np.arange(len( varidX[i])):
            varidXall.append( materials[i]+':'+varidX[i][j])
    tpls_obj['varidX']=varidXall    
    
    if not isinstance(obsidR,bool):
       tpls_obj['obsidRi']=obsidR
       tpls_obj['varidRi']=varidR  
    
    if not isinstance(obsidY,bool):
       tpls_obj['obsidY']=obsidY
       tpls_obj['varidY']=varidY    

    if not isinstance(obsidZ,bool):
       tpls_obj['obsidZ']=obsidZ
       tpls_obj['varidZ']=varidZ  
       
    T2 = hott2(tpls_obj,Tnew=T)
    n  = T.shape[0]
    T2_lim99 = (((n-1)*(n+1)*A)/(n*(n-A)))*f99(A,(n-A))
    T2_lim95 = (((n-1)*(n+1)*A)/(n*(n-A)))*f95(A,(n-A))    
    
    speX=[]
    speR=[]
    speX_lim95=[]
    speX_lim99=[]
    speR_lim95=[]
    speR_lim99=[]
    for i in np.arange(len(X)):
        speX .append(  np.sum(X[i]**2,axis=1,keepdims=1))
        aux_=np.sum(X[i]**2,axis=1,keepdims=1)
        speX_lim95_,speX_lim99_ = spe_ci(aux_)
        speX_lim95.append(speX_lim95_)
        speX_lim99.append(speX_lim99_)
        
        speR.append( np.sum(R[i]**2,axis=1,keepdims=1))
        aux_=np.sum(R[i]**2,axis=1,keepdims=1)
        speR_lim95_,speR_lim99_ = spe_ci(aux_)
        speR_lim95.append(speR_lim95_)
        speR_lim99.append(speR_lim99_)
        
    speY = np.sum(Y_**2,axis=1,keepdims=1)
    speY_lim95,speY_lim99 = spe_ci(speY)
    
    speZ = np.sum(Z_**2,axis=1,keepdims=1)
    speZ_lim95,speZ_lim99 = spe_ci(speZ)
    
    tpls_obj['T2']          = T2
    tpls_obj['T2_lim99']    = T2_lim99
    tpls_obj['T2_lim95']    = T2_lim95
    tpls_obj['speX']        = speX
    tpls_obj['speX_lim99']  = speX_lim99
    tpls_obj['speX_lim95']  = speX_lim95
    tpls_obj['speY']        = speY
    tpls_obj['speY_lim99']  = speY_lim99
    tpls_obj['speY_lim95']  = speY_lim95
    tpls_obj['speR']        = speR
    tpls_obj['speR_lim99']  = speR_lim99
    tpls_obj['speR_lim95']  = speR_lim95
    tpls_obj['speZ']        = speZ
    tpls_obj['speZ_lim99']  = speZ_lim99
    tpls_obj['speZ_lim95']  = speZ_lim95
    
    Wsi= []
    Ws = []
    for i in np.arange(len(S)):
        Wsi.append (S[i] @ np.linalg.pinv(V[i].T @ S[i]) )
        if i==0:
            Ws=S[i] @ np.linalg.pinv(V[i].T @ S[i])
        else:
            Ws=np.vstack((Ws,S[i] @ np.linalg.pinv(V[i].T @ S[i])))    
    tpls_obj['Ssi'] = Wsi #trick to plot the JRPLS/LPLS does not really have Ws
    tpls_obj['Ss'] = Ws #trick to plot the JRPLS/LPLS does not really have Ws
    #Ws=W @ np.linalg.pinv(P.T @ W)
    tpls_obj['type']='tpls'
    
    Ws=W @ np.linalg.pinv(Pz.T @ W)
    tpls_obj['Ws']=Ws
    return tpls_obj       

def tpls_pred(rnew,znew,tplsobj):
    '''
    Routine to produce the prediction for a new observation of Ri
    using a TPLS model
    preds = (rnew,znew,tplsobj)
    
    Inputs:
        rnew: A dictionary with the format: 
            rnew={ 
                'matid':[(lotid,rvalue )],
                
                }
            
            for example, a prediction for the scenario:
                
        material    lot to use  rvalue       
        API	        A0129	    0.5
        Lactose	    Lac0003   	0.1
        Lactose     Lac1010     0.2
        MgSt	        M0012	    0.02
        MCC      	MCC0017	    0.18
        
        use:
            
        rnew={
            'API':[('A0129',0.5)],
            'Lactose':[('Lac0003',0.1 ),('Lac1010',0.2 )],
            'MgSt':[('M0012',0.02)],
            'MCC':[('MCC0017',0.18)],
            }    
        
        znew: Dataframe or numpy with new observation
        
    Outputs:
        preds a dictionary of the form:
            
            preds ={'Tnew':tnew,'Yhat':yhat,'speR':sper,'speZ':spez} 
            
            where speR has the speR per each material
            

    '''
    
    ok=True
    if isinstance(rnew,list):
        #check dimensions
        i=0     
        for r,mr,sr in zip(rnew,tplsobj['mr'],tplsobj['sr']):
            if not(len(r)==len(mr[0])):
                ok=False
            np.ones(len(r))
            if i==0:
                rnew_=r
                mr_=mr
                sr_=sr
                Rscores = tplsobj['Rscores'][i]
                P       = tplsobj['P'][i]
            else:
                rnew_   = np.hstack((rnew_,r))    
                mr_     = np.hstack((mr_,mr))
                sr_     = np.hstack((sr_,sr))
                Rscores = np.vstack((Rscores,tplsobj['Rscores'][i] ))
                P       = np.vstack((P,tplsobj['P'][i] ))
            i+=1
                
    elif isinstance(rnew,dict):
        #re-arrange 
        ok=True
        rnew_=[['*']]*len(tplsobj['materials'])
        for k in list(rnew.keys()):
            i = tplsobj['materials'].index(k)
            ri=np.zeros((tplsobj['mr'][i].shape[1]) )
            for m,r in rnew[k]:
                e = tplsobj['varidRi'][i].index(m)
                ri[e]=r
            rnew_[i]=ri
        
        preds=tpls_pred(rnew_,znew,tplsobj)
        return preds
    if isinstance(znew,pd.DataFrame):
        znew_=znew.values.reshape(-1)[1:].astype(float)
    elif isinstance(znew,list):
        znew_=np.array(znew)
    elif isinstance(znew,np.ndarray):
        znew_=znew.copy()
        
    if not(len(znew_)==tplsobj['mz'].shape[1]):
        ok = False
   
    if ok:  
        bkzeros=0
        selmat=[]
        for i,r in enumerate(tplsobj['Rscores']):
            frontzeros=Rscores.shape[0]-bkzeros-r.shape[0]
            row=np.vstack((np.zeros((bkzeros,1)),
                           np.ones((r.shape[0],1)),
                           np.zeros(( frontzeros,1))))
            bkzeros+=r.shape[0]
            selmat.append(row)
            
        tnew=[]
        sper=[]
        spez=[]
       
        rnew_=(rnew_-mr_)/sr_
        rnew_=rnew_.reshape(-1,1)

        znew_=(znew_-tplsobj['mz'])/tplsobj['sz']
        znew_=znew_.reshape(-1,1)

        
        tnew=[]
        for a in np.arange(tplsobj['T'].shape[1]):

            ti_rx_ =( rnew_.T@Rscores[:,a]/             
                 (Rscores[:,a].T@Rscores[:,a]))
            
            ti_z_ = znew_.T@tplsobj['W'][:,a]
            
            
            ti_ = np.array([ti_rx_,ti_z_]).reshape(1,-1)@tplsobj['Wt'][:,a]
            
            tnew.append(ti_[0])
            aux=ti_*P[:,a]
            rnew_=rnew_-aux.reshape(-1,1)
            
            auxz=ti_*tplsobj['Pz'][:,a]
            znew_=znew_-auxz.reshape(-1,1)
            
        
        sper=[]
        for row in selmat:
            sper.append(np.sum(rnew_[row==1]**2))
        spez=np.sum(znew_**2)
        tnew=np.array(tnew)
        yhat = tnew@tplsobj['Q'].T
        yhat = (yhat * tplsobj['sy'])+tplsobj['my']
        preds ={'Tnew':tnew,'Yhat':yhat,'speR':sper,'speZ':spez}     
        return preds    
    else:
        return 'dimensions of rnew or znew did not macth model'
    
def varimax_(X, gamma = 1.0, q = 20, tol = 1e-6):
    p,k = X.shape
    R = eye(k)
    d=0
    for i in range(q):
        d_ = d
        Lambda = dot(X, R)
        u,s,vh = svd(dot(X.T,asarray(Lambda)**3 - (gamma/p) * dot(Lambda, diag(diag(dot(Lambda.T,Lambda))))))
        R = dot(u,vh)
        d = sum(s)
        if d_!=0 and d/d_ < 1 + tol: break
    return dot(X, R)

def varimax_rotation(mvm_obj,X,*,Y=False):
    mvmobj=mvm_obj.copy()
    if isinstance(X,np.ndarray):
        X_=X.copy
    if isinstance(X,pd.DataFrame):
        X_=X.values[:,1:].astype(float)
        
    if isinstance(Y,np.ndarray):
        Y_=Y.copy
    if isinstance(Y,pd.DataFrame):
        Y_=Y.values[:,1:].astype(float)
            
    X_ = (X_ - np.tile(mvmobj['mx'],(X_.shape[0],1)))/np.tile(mvmobj['sx'],(X_.shape[0],1))
    not_Xmiss = ~(np.isnan(X_))*1
    X_,Xmap=n2z(X_)
    TSSX   = np.sum(X_**2)
    TSSXpv = np.sum(X_**2,axis=0)
    if not(isinstance(Y,bool)):
        Y_ = (Y_ - np.tile(mvmobj['my'],(Y_.shape[0],1)))/np.tile(mvmobj['sy'],(Y_.shape[0],1))
        not_Ymiss = ~(np.isnan(Y_))*1
        Y_,Ymap=n2z(Y_)
        TSSY   = np.sum(Y_**2)
        TSSYpv = np.sum(Y_**2,axis=0)
        
    A=mvmobj['T'].shape[1]
    if 'Q' in mvmobj:
        Wrot=varimax_(mvmobj['W'])
        Trot=[]
        Prot=[]
        Qrot=[]
        Urot=[]
        for a in np.arange(A):
            ti=_Ab_btbinv(X_, Wrot[:,a], not_Xmiss)
            pi=_Ab_btbinv(X_.T,ti,not_Xmiss.T )
            qi=_Ab_btbinv(Y_.T,ti,not_Ymiss.T )
            ui=_Ab_btbinv(Y_,qi,not_Ymiss )
            X_ = (X_ - ti@pi.T)*not_Xmiss
            Y_ = (Y_ - ti@qi.T)*not_Ymiss
            if a==0:
                
                r2Xpv = np.zeros((1,len(TSSXpv))).reshape(-1)                
                r2X   = 1-np.sum(X_**2)/TSSX
                r2Xpv[TSSXpv>0] = 1-(np.sum(X_**2,axis=0)[TSSXpv>0]/TSSXpv[TSSXpv>0])
                r2Xpv = r2Xpv.reshape(-1,1)   
                                
                r2Ypv = np.zeros((1,len(TSSYpv))).reshape(-1)                
                r2Y   = 1-np.sum(Y_**2)/TSSY
                r2Ypv[TSSYpv>0] = 1-(np.sum(Y_**2,axis=0)[TSSYpv>0]/TSSYpv[TSSYpv>0])
                r2Ypv = r2Ypv.reshape(-1,1)                   
                
            else:
                
                r2X   = np.hstack((r2X,1-np.sum(X_**2)/TSSX))
                aux_  = np.zeros((1,len(TSSXpv))).reshape(-1)
                aux_[TSSXpv>0]  = 1- (np.sum(X_**2,axis=0)[TSSXpv>0]/TSSXpv[TSSXpv>0])
                aux_  = aux_.reshape(-1,1)
                r2Xpv = np.hstack((r2Xpv,aux_))
                                
                r2Y   = np.hstack((r2Y,1-np.sum(Y_**2)/TSSY))
                aux_  = np.zeros((1,len(TSSYpv))).reshape(-1)
                aux_[TSSYpv>0]  = 1- (np.sum(Y_**2,axis=0)[TSSYpv>0]/TSSYpv[TSSYpv>0])
                aux_  = aux_.reshape(-1,1)
                r2Ypv = np.hstack((r2Ypv,aux_))


            Trot.append(ti)
            Prot.append(pi)
            Qrot.append(qi)
            Urot.append(ui)
                            
        for a in list(range(A-1,0,-1)):
            r2X[a]     = r2X[a]-r2X[a-1]
            r2Xpv[:,a] = r2Xpv[:,a]-r2Xpv[:,a-1]
            r2Y[a]     = r2Y[a]-r2Y[a-1]
            r2Ypv[:,a] = r2Ypv[:,a]-r2Ypv[:,a-1]
        Trot=np.array(Trot).T
        Prot=np.array(Prot).T
        Qrot=np.array(Qrot).T
        Urot=np.array(Urot).T
        Trot=Trot[0]
        Prot=Prot[0]
        Qrot=Qrot[0]
        Urot=Urot[0]
        Wsrot=Wrot @ np.linalg.pinv(Prot.T @ Wrot)
        mvmobj['W']=Wrot
        mvmobj['T']=Trot
        mvmobj['P']=Prot
        mvmobj['Q']=Qrot
        mvmobj['U']=Urot
        mvmobj['Ws']=Wsrot
        mvmobj['r2x']   = r2X
        mvmobj['r2xpv'] = r2Xpv
        mvmobj['r2y']   = r2Y
        mvmobj['r2ypv'] = r2Ypv
              
    else:
        Prot=varimax_( mvmobj['P'])
        Trot=[]
        for a in np.arange(A):
            ti=_Ab_btbinv(X_, Prot[:,a], not_Xmiss)
            Trot.append(ti)
            pi=Prot[:,[a]]
            X_ = (X_ - ti@pi.T)*not_Xmiss
            if a==0:                
                r2Xpv = np.zeros((1,len(TSSXpv))).reshape(-1)                
                r2X   = 1-np.sum(X_**2)/TSSX
                r2Xpv[TSSXpv>0] = 1-(np.sum(X_**2,axis=0)[TSSXpv>0]/TSSXpv[TSSXpv>0])
                r2Xpv = r2Xpv.reshape(-1,1)                
            else:
                r2X   = np.hstack((r2X,1-np.sum(X_**2)/TSSX))
                aux_ = np.zeros((1,len(TSSXpv))).reshape(-1)
                aux_[TSSXpv>0]  = 1- (np.sum(X_**2,axis=0)[TSSXpv>0]/TSSXpv[TSSXpv>0])
                aux_  = aux_.reshape(-1,1)
                r2Xpv = np.hstack((r2Xpv,aux_))
            
        Trot=np.array(Trot).T
        for a in list(range(A-1,0,-1)):
            r2X[a]     = r2X[a]-r2X[a-1]
            r2Xpv[:,a] = r2Xpv[:,a]-r2Xpv[:,a-1]            
        mvmobj['P']=Prot
        mvmobj['T']=Trot[0]
        mvmobj['r2x']   = r2X
        mvmobj['r2xpv'] = r2Xpv
    return mvmobj
        
        