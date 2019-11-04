"""
Phi for Python (pyPhi)

by Salvador Garcia (sgarciam@ic.ac.uk salvadorgarciamunoz@gmail.com)

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
             
        md_algorithm: 'nipals' *default if not sent*
                      'nlp'    To be implemented
                      
        force_nipals: If = True and if X is complete, will use NIPALS.
                           Otherwise, if X is complete will use SVD.
                         = False *default if not sent*
                      
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
             P          = X.T @ T
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
                          if not(shush):
                              print('# Iterations for PC #'+str(a+1)+': ',str(num_it))
                          if a==0:
                              T=ti
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
        elif md_algorithm=='nlp':
            #use NLP per Lopez-Negrete et al. J. Chemometrics 2010; 24: 301–311
            pca_obj=1
            return pca_obj
  
def pls(X,Y,A,*,mcsX=True,mcsY=True,md_algorithm='nipals',force_nipals=False,shush=False,cross_val=0,cross_val_X=False):
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
                      'nlp'    To be implemented
                      
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
            #use NLP per Eranda's paper
            pls_obj=1
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
#        yhat =yhat + tnew @ q
        yhat  = yhat + q @ tnew         
        Xi    = Xi - t @ p.T
        Yi    = Yi - t @ q.T
        xnewi = xnewi - p @ tnew
    return yhat.T
    
    
def pca_pred(Xnew,pcaobj,*,algorithm='p2mp'):
    if isinstance(Xnew,np.ndarray):
        X_=Xnew.copy()
        if X_.ndim==1:
            X_=np.reshape(X_,(1,-1))
    elif isinstance(Xnew,pd.DataFrame):
        X_=np.array(Xnew.values[:,1:]).astype(float)

    X_nan_map = np.isnan(X_)
    #not_Xmiss = (np.logical_not(X_nan_map))*1
    if not(X_nan_map.any()):
        X_mcs= X_- np.tile(pcaobj['mx'],(X_.shape[0],1))
        X_mcs= X_mcs/(np.tile(pcaobj['sx'],(X_.shape[0],1)))      
        tnew =  X_mcs @ pcaobj['P']
        xhat = (tnew @ pcaobj['P'].T) * np.tile(pcaobj['sx'],(X_.shape[0],1)) + np.tile(pcaobj['mx'],(X_.shape[0],1))
        var_t = (pcaobj['T'].T @ pcaobj['T'])/pcaobj['T'].shape[0]
        htt2 = np.sum((tnew @ np.linalg.inv(var_t)) * tnew,axis=1)
        spe  = ((X_-np.tile(pcaobj['mx'],(X_.shape[0],1)))/(np.tile(pcaobj['sx'],(X_.shape[0],1))))-(tnew @ pcaobj['P'].T)
        spe  = np.sum(spe**2,axis=1,keepdims=True) 
        xpred={'Xhat':xhat,'Tnew':tnew, 'speX':spe,'T2':htt2}
    elif algorithm=='p2mp':  # Using Projection to the model plane method for missing data    
        X_nan_map = np.isnan(Xnew)
        not_Xmiss = (np.logical_not(X_nan_map))*1
        X_,dummy=n2z(X_)
        Xmcs=((X_-np.tile(pcaobj['mx'],(X_.shape[0],1)))/(np.tile(pcaobj['sx'],(X_.shape[0],1))))
        for i in list(range(Xmcs.shape[0])):
            row_missing_map=not_Xmiss[[i],:]
            tempP = pcaobj['P'] * np.tile(row_missing_map.T,(1,pcaobj['P'].shape[1]))
            PTP = tempP.T @ tempP
            tnew_ = np.linalg.inv(PTP) @ tempP.T  @ Xmcs[[i],:].T
            if i==0:
                tnew = tnew_.T
            else:
                tnew = np.vstack((tnew,tnew_.T))
        xhat = (tnew @ pcaobj['P'].T) * np.tile(pcaobj['sx'],(X_.shape[0],1)) + np.tile(pcaobj['mx'],(X_.shape[0],1))        
        var_t = (pcaobj['T'].T @ pcaobj['T'])/pcaobj['T'].shape[0]
        htt2 = np.sum((tnew @ np.linalg.inv(var_t)) * tnew,axis=1)
        spe  = ((X_-np.tile(pcaobj['mx'],(X_.shape[0],1)))/(np.tile(pcaobj['sx'],(X_.shape[0],1))))-(tnew @ pcaobj['P'].T)
        spe  = np.sum(spe**2,axis=1,keepdims=True) 
        xpred={'Xhat':xhat,'Tnew':tnew, 'speX':spe,'T2':htt2}
    return xpred

def pls_pred(Xnew,plsobj,*,algorithm='p2mp'):
    if isinstance(Xnew,np.ndarray):
        X_=Xnew.copy()
        if X_.ndim==1:
            X_=np.reshape(X_,(1,-1))
    elif isinstance(Xnew,pd.DataFrame):
        X_=np.array(Xnew.values[:,1:]).astype(float)
    X_nan_map = np.isnan(X_)    
    if not(X_nan_map.any()):
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
        X_,dummy=n2z(X_)
        Xmcs=((X_-np.tile(plsobj['mx'],(X_.shape[0],1)))/(np.tile(plsobj['sx'],(X_.shape[0],1))))
        for i in list(range(Xmcs.shape[0])):
            row_missing_map=not_Xmiss[[i],:]
            tempWs = plsobj['Ws'] * np.tile(row_missing_map.T,(1,plsobj['Ws'].shape[1]))
            
            for a in list(range(plsobj['Ws'].shape[1])):
                WsTWs    = tempWs[:,[a]].T @ tempWs[:,[a]]
                tnew_aux = np.linalg.inv(WsTWs) @ tempWs[:,[a]].T  @ Xmcs[[i],:].T
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
        speX  = ((X_-np.tile(plsobj['mx'],(X_.shape[0],1)))/(np.tile(plsobj['sx'],(X_.shape[0],1))))-(tnew @ plsobj['P'].T)
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
        #Calculate mean without accounting for NaN's
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
    if isinstance(mcs,bool):
        if mcs:
            x_mean = mean(X)
            x_std  = std(X)
            X      = X-np.tile(x_mean,(X.shape[0],1))
            X      = X/np.tile(x_std,(X.shape[0],1))
    elif mcs=='center':
         x_mean = mean(X)
         X      = X-np.tile(x_mean,(X.shape[0],1))
         x_std  = np.ones((1,X.shape[1]))
    elif mcs=='autoscale':
         x_std  = std(X) 
         X      = X/np.tile(x_std,(X.shape[0],1))
         x_mean = np.zeros((1,X.shape[1]))
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

def np2D2pyomo(arr):
    output=dict(((i+1,j+1), arr[i][j]) for i in range(arr.shape[0]) for j in range(arr.shape[1]))
    return output

def np1D2pyomo(arr,*,indexes=False):
    if arr.ndim==2:
        arr=arr[0]
    if isinstance(indexes,bool):
        output=dict(((j+1), arr[j]) for j in range(len(arr)))
    elif isinstance(indexes,list):
        output=dict((indexes[j], arr[j]) for j in range(len(arr)))
    return output
       
def conv_eiot(plsobj,*,r_length=False):
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
    plsobj_['pyo_S_I']        = np.nan
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

        
        
def clean_low_variances(X,*,shush=False):
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
    std_x=std(X_)
    std_x=std_x.flatten()
    
    indx = find(std_x, lambda x: x<1E-10)
    if len(indx)>0:
        for i in indx:
            if not(shush):
                print('Removing variable ', varidX[i], ' due to low variance')
        if isinstance(X,pd.DataFrame):
            indx = np.array(indx)
            indx = indx +1
            X_=X.drop(X.columns[indx],axis=1)
        else:
            X_=np.delete(X_,indx,1)
        return X_    
    else:
        return X
    
def find(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]
    
        

    


