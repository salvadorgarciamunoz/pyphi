import numpy as np

def phi_z2n(X,X_nan_map):
    X[X_nan_map==1] = np.nan
    return X

def phi_n2z(X):
    X_nan_map = np.isnan(X)            
    if X_nan_map.any():
        X_nan_map       = X_nan_map*1
        X[X_nan_map==1] = 0
    else:
        X_nan_map       = X_nan_map*1
    return X,X_nan_map

def phi_mean(X):
    X_nan_map = np.isnan(X)
    if X_nan_map.any():
        X_nan_map       = X_nan_map*1
        X[X_nan_map==1] = 0
        aux             = np.sum(X_nan_map,axis=0)
        #Calculate mean without accounting for NaN's
        x_mean = np.sum(X,axis=0,keepdims=1)/(np.ones((1,X.shape[1]))*X.shape[0]-aux)
    else:
        x_mean = np.mean(X,axis=0,keepdims=1)
    return x_mean

def phi_std(X):
    x_mean=phi_mean(X)
    x_mean=np.tile(x_mean,(X.shape[0],1))
    X_nan_map = np.isnan(X)
    if X_nan_map.any():
        X_nan_map       = X_nan_map*1
        X[X_nan_map==1] = 0
        aux             = np.sum(X_nan_map,axis=0)
        #Calculate mean without accounting for NaN's
        x_std = np.sqrt(np.sum((X-x_mean)**2,axis=0,keepdims=1)/(np.ones((1,X.shape[1]))*(X.shape[0]-1)-aux))
    else:
        x_std = np.sqrt(np.sum((X-x_mean)**2,axis=0,keepdims=1)/(np.ones((1,X.shape[1]))*(X.shape[0]-1)))
    return x_std
    

def phi_pca(X,A,*,mcs=True):
    if isinstance(mcs,bool):
        if mcs:
            #Mean center and autoscale  
            pca_obj=1
    elif mcs=='center':
        #only center
        pca_obj=1
    elif mcs=='autoscale':
        #only autoscale
        pca_obj=1
    return pca_obj
