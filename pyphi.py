import numpy as np
import pandas as pd
import datetime
from scipy.special import factorial
  
def pca(X,A,*,mcs=True,md_algorithm='nipals',force_nipals=False):
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
    elif mcs=='center':
        X_,x_mean,x_std = meancenterscale(X_,mcs='center')
        #only center
    elif mcs=='autoscale':
        #only autoscale
        X_,x_mean,x_std = meancenterscale(X_,mcs='autoscale')
        
    #Generate Missing Data Map    
    X_nan_map = np.isnan(X_)
    not_Xmiss = (np.logical_not(X_nan_map))*1
    
    if not(X_nan_map.any()) and not(force_nipals):
        #no missing elements
        print('phi.pca using SVD executed on: '+ str(datetime.datetime.now()) )
        TSS   = np.sum(X_**2)
        TSSpv = np.sum(X_**2,axis=0)
        if X_.shape[1]>X_.shape[0]:
             [U,S,Th]   = np.linalg.svd(X_ @ X_.T)
             T          = Th.T 
             T          = T[:,0:A]
             P          = T.T @ X
             for a in list(range(A)):
                 P[:,a] = P[:,a]/np.linalg.norm(P[:,a])
        elif X_.shape[0]>=X_.shape[1]:
             [U,S,Ph]   = np.linalg.svd(X_.T @ X_)
             P          = Ph.T
             P          = P[:,0:A]
             T          = X_ @ P
        for a in list(range(A)):
            X_ = X_-T[:,[a]]@P[:,[a]].T
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
        return pca_obj
    else:
        if md_algorithm=='nipals':
             #use nipals
             print('phi.pca using NIPALS executed on: '+ str(datetime.datetime.now()) )
             X_,dummy=n2z(X_)
             epsilon=1E-10
             maxit=10000
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
             print('--------------------------------------------------------------')
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
             return pca_obj                            
        elif md_algorithm=='nlp':
            #use NLP per Lopez-Negrete et al. J. Chemometrics 2010; 24: 301–311
            pca_obj=1
            return pca_obj
  
def pls(X,Y,A,*,mcsX=True,mcsY=True,md_algorithm='nipals',force_nipals=False):
    ''' Partial Least Squares routine by Sal Garcia sgarciam@ic.ac.uk
        import pyphi as phi
        #import your data into X and Y#
        #simplest use for a 2LV model
        [mvm] = phi.pls(X,Y,2)    
    
        Options:   
        mcsX / mcsY  = 'center' | 'autoscale' | True  <does both and is the default>
        md_algorithm = 'nipals' <default>  (NLP options to come soon)
        force_nipals = True | False <default> (If data has is complete and force_nipals='False' then uses SVD)
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
        
   
    if isinstance(mcsX,bool):
        if mcsX:
            #Mean center and autoscale  
            X_,x_mean,x_std = meancenterscale(X_)
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
        return pls_obj
    else:
        if md_algorithm=='nipals':
             #use nipals
             print('phi.pls using NIPALS executed on: '+ str(datetime.datetime.now()) )
             X_,dummy=n2z(X_)
             Y_,dummy=n2z(Y_)
             epsilon=1E-10
             maxit=10000

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
             return pls_obj   
                         
        elif md_algorithm=='nlp':
            #use NLP per Eranda's paper
            pls_obj=1
            return pls_obj

def lwpls(xnew,loc_par,mvm,X,Y):
    vip=np.sum(np.abs(mvm['Ws'] * np.tile(mvm['r2y'],(mvm['Ws'].shape[0],1)) ),axis=1)
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
    
    for a in list(range(0,mvm['T'].shape[1])):
        [U_,S,Wh]=np.linalg.svd(Xi.T @ OMEGA @ Yi @ Yi.T @ OMEGA @ Xi)
        w           = Wh.T
        w           = w[:,[0]]
        t = Xi @ w
        p = Xi.T @ OMEGA @ t / (t.T @ OMEGA @ t)        
        q = Yi.T @ OMEGA @ t / (t.T @ OMEGA @ t)
        
        tnew = xnewi.T @ w
        yhat =yhat + tnew @ q
        
        Xi    = Xi - t @ p.T
        Yi    = Yi - t @ q.T
        xnewi = xnewi - p @ tnew
    return yhat
    
    
def pca_pred(Xnew,pcaobj,*,algorithm='p2mp'):
    X_=Xnew.copy()
    X_nan_map = np.isnan(X_)
    #not_Xmiss = (np.logical_not(X_nan_map))*1
    if not(X_nan_map.any()):
        tnew = ((X_-np.tile(pcaobj['mx'],(X_.shape[0],1)))/(np.tile(pcaobj['sx'],(X_.shape[0],1)))) @ pcaobj['P']
        xhat = (tnew @ pcaobj['P'].T) * np.tile(pcaobj['sx'],(X_.shape[0],1)) + np.tile(pcaobj['mx'],(X_.shape[0],1))
        xpred={'Xhat':xhat,'Tnew':tnew}
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
        xpred={'Xhat':xhat,'Tnew':tnew}
    return xpred

def pls_pred(Xnew,plsobj,algorithm='p2mp'):
    X_=Xnew.copy()
    X_nan_map = np.isnan(X_)    
    if not(X_nan_map.any()):
        tnew = ((X_-np.tile(plsobj['mx'],(X_.shape[0],1)))/(np.tile(plsobj['sx'],(X_.shape[0],1)))) @ plsobj['Ws']
        yhat = (tnew @ plsobj['Q'].T) * np.tile(plsobj['sy'],(X_.shape[0],1)) + np.tile(plsobj['my'],(X_.shape[0],1))
        xhat = (tnew @ plsobj['P'].T) * np.tile(plsobj['sx'],(X_.shape[0],1)) + np.tile(plsobj['mx'],(X_.shape[0],1))
        ypred ={'Yhat':yhat,'Xhat':xhat,'Tnew':tnew}
    elif algorithm=='p2mp':
        X_nan_map = np.isnan(Xnew)
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
        ypred ={'Yhat':yhat,'Xhat':xhat,'Tnew':tnew}     
    return ypred



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
        Dm_sg= M @ Dm
    else:
        for i in np.arange(1,Dm.shape[0]+1):
            dm_ = M @ Dm[i-1,:]
            if i==1:
                Dm_sg=dm_
            else:
                Dm_sg=np.vstack((Dm_sg,dm_))
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
    h=(2*spem^2)/spev

    lim95=np.interp(h,chi[:,[1]],chi[:,[2]])
    lim99=np.interp(h,chi[:,[1]],chi[:,[3]]);
    lim95= g*lim95
    lim99= g*lim99
    return lim95,lim99