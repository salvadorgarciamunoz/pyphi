"""
Phi for Python (pyPhi)  —  Version 2.0

By Sal Garcia (sgarciam@ic.ac.uk salvadorgarciamunoz@gmail.com)

Added Feb 23 2026
    * Added _validate_inputs function for input validation and observation reconciliation
    * Integrated validation into pca, pls, lpls entry points
    * Replaced np.tile with numpy broadcasting throughout
    * Optimized _Ab_btbinv with fast path for complete data
    * var_t (score covariance matrix) stored in model objects to avoid recalculation
    * Added _extract_array and _calc_r2 helper functions to reduce duplication
    * Replaced hardcoded F-distribution and chi2 lookup tables with scipy.stats
    * Replaced hardcoded t-distribution with scipy.stats

Added Feb 07 2026
        * fixed cat_2_matrix for the output to be consistent with MBPLS
Added Jan 30 2025
        * Added a pinv alternative protection in spectra_savgol for the case where
          inv fails
Added Jan 20 2025
        * Added the 'cca' flag to the pls routine to calculate CCA between
          the Ts and each of the Ys (one by one), calculating loadings and scores 
          equivalent to a perfectly orthogonalized OPLS model. The covariant scores (Tcv)
          the covariant Loadings (Pcv) and predictive weights (Wcv) are added
          as keys to the model object.
          [The covariant loadings(Pcv) are equivalent to the predictive loadings in OPLS]
          
        * Added cca and cca-multi routines to perform PLS-CCA (alternative to OPLS)
          as of now cca-multi remains unused.

Added Nov 18th, 2024
        * replaced interp2d with RectBivariateSpline 
        * Protected SPE lim calculations for near zero residuals
        * Added build_polynomial function to create linear regression
          models with variable selection assited by PLS

by merge from James
        * Added spectra preprocessing methods
        * bootstrap PLS
        
by Salvador Garcia (sgarciam@ic.ac.uk salvadorgarciamunoz@gmail.com)
Added Dec 19th 2023
        * phi.clean_htmls  removes all html files in the working directory
        * clean_empty_rows returns also the names of the rows removed
Added May 1st
        * YMB is now added in the same structure as the XMB
        * Corrected the dimensionality of the lwpls prediction, it was a double-nested array.
        
Added Apr 30
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
        *Fixed access to NEOS server and use of GAMS instead of IPOPT
        
Release as of Aug 12 2022       
        * Fixed the SPE calculations in pls_pred and pca_pred
        * Changed to a more efficient inversion in pca_pred (=pls_pred)
        * Added a pseudo-inverse option in pmp for pca_pred
        
Release as of Aug 2 2022
        *Added replicate_data

"""

import numpy as np
import pandas as pd
import datetime
from scipy.special import factorial
from scipy.stats import norm, f as f_dist, chi2
from scipy.optimize import fsolve
from scipy.interpolate import RectBivariateSpline
from scipy.stats import t as t_dist
from shutil import which
import os
from numpy import eye, asarray, dot, sum, diag
from numpy.linalg import svd
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF

os.environ['NEOS_EMAIL'] = 'pyphisoftware@gmail.com' 

try:
    from pyomo.environ import *
    pyomo_ok = True
except ImportError:
    pyomo_ok = False

if bool(which('gams')):
    gams_ok = True    
else:
    gams_ok = False

ipopt_ok = bool(which('ipopt'))

if pyomo_ok and gams_ok:
    from pyomo.solvers.plugins.solvers.GAMS import GAMSDirect, GAMSShell
    gams_ok = (GAMSDirect().available(exception_flag=False)
                or GAMSShell().available(exception_flag=False))

def ma57_dummy_check():
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

# =============================================================================
# Helper functions (new in v1.0)
# =============================================================================

def _extract_array(X):
    """Extract numpy array, observation IDs, and variable IDs from DataFrame or ndarray.
    
    Args:
        X: numpy array or pandas DataFrame (first column = obs IDs)
    Returns:
        (array, obsid, varid) where obsid/varid are False if input is ndarray
    """
    if isinstance(X, np.ndarray):
        return X.copy(), False, False
    elif isinstance(X, pd.DataFrame):
        arr = np.array(X.values[:, 1:]).astype(float)
        obsid = X.values[:, 0].astype(str).tolist()
        varid = X.columns.values[1:].tolist()
        return arr, obsid, varid

def _calc_r2(residual, TSS, TSSpv, prev_r2=None, prev_r2pv=None):
    """Calculate cumulative R2 and R2 per variable, optionally appending to previous.
    
    Args:
        residual: residual matrix after deflation
        TSS: total sum of squares (scalar)
        TSSpv: total sum of squares per variable (1D array)
        prev_r2: previous cumulative R2 array (None for first component)
        prev_r2pv: previous cumulative R2pv matrix (None for first component)
    Returns:
        (r2, r2pv) updated arrays
    """
    r2_new = 1 - np.sum(residual**2) / TSS
    r2pv_new = (1 - np.sum(residual**2, axis=0) / TSSpv).reshape(-1, 1)
    if prev_r2 is None:
        return r2_new, r2pv_new
    return np.hstack((prev_r2, r2_new)), np.hstack((prev_r2pv, r2pv_new))

def _r2_cumulative_to_per_component(r2, r2pv, A):
    """Convert cumulative R2 to per-component R2."""
    for a in range(A-1, 0, -1):
        r2[a]     = r2[a] - r2[a-1]
        r2pv[:, a] = r2pv[:, a] - r2pv[:, a-1]
    return r2, r2pv

def _Ab_btbinv(A, b, A_not_nan_map):
    """Project c = Ab / (b'b), accounting for missing data.
    
    Optimized: uses broadcasting instead of np.tile, with fast path
    when there is no missing data.
    """
    b_flat = b.ravel()
    numer = A @ b_flat
    # Fast path: no missing data
    if A_not_nan_map.all():
        return (numer / np.dot(b_flat, b_flat)).reshape(-1, 1)
    denom = np.sum((A_not_nan_map * b_flat)**2, axis=1)
    return (numer / denom).reshape(-1, 1)

# =============================================================================
# Input validation (new in v2.0)
# =============================================================================

def _validate_inputs(X, Y=None, A=None, mcs=None):
    """Validate and reconcile inputs for PCA and PLS model building.
    
    Checks performed:
        - X (and Y if provided) are DataFrame or ndarray
        - DataFrames have a string/object first column (observation IDs)
        - No duplicate observation IDs
        - X and Y observation IDs match one-to-one (reorders Y if needed)
        - A does not exceed rank limits
        - mcs flag is a recognized value
    
    Args:
        X:   pandas DataFrame or numpy ndarray
        Y:   pandas DataFrame, numpy ndarray, or None (PCA case)
        A:   int, number of components requested, or None to skip check
        mcs: mcs flag value(s) to validate, or None to skip check
             For PLS, pass as tuple: (mcsX, mcsY)
    
    Returns:
        X, Y  — potentially reordered so observations align.
                If Y is None, returns (X, None).
    
    Raises:
        ValueError: on validation failures that cannot be auto-corrected
    """
    _valid_mcs = {True, False, 'center', 'autoscale'}
    
    # --- Validate types ---
    if not isinstance(X, (np.ndarray, pd.DataFrame)):
        raise ValueError(
            f"X must be a pandas DataFrame or numpy ndarray, got {type(X).__name__}")
    if Y is not None and not isinstance(Y, (np.ndarray, pd.DataFrame)):
        raise ValueError(
            f"Y must be a pandas DataFrame or numpy ndarray, got {type(Y).__name__}")
    
    # --- Validate mcs flags ---
    if mcs is not None:
        mcs_values = mcs if isinstance(mcs, tuple) else (mcs,)
        for m in mcs_values:
            if m not in _valid_mcs:
                raise ValueError(
                    f"mcs value '{m}' not recognized. Must be one of: True, False, 'center', 'autoscale'")
    
    # --- Validate A ---
    if A is not None:
        if not isinstance(A, (int, np.integer)) or A < 1:
            raise ValueError(f"A must be a positive integer, got {A}")
    
    # --- Helper: check DataFrame structure ---
    def _check_df_structure(df, name):
        if df.shape[1] < 2:
            raise ValueError(
                f"{name} DataFrame must have at least 2 columns "
                f"(1 obs ID column + 1 data column), got {df.shape[1]}")
        data_cols = df.iloc[:, 1:]
        non_numeric = []
        for col in data_cols.columns:
            try:
                data_cols[col].astype(float)
            except (ValueError, TypeError):
                non_numeric.append(col)
        if non_numeric:
            raise ValueError(
                f"{name} has non-numeric data columns: {non_numeric[:5]}"
                + (f" ... and {len(non_numeric)-5} more" if len(non_numeric) > 5 else ""))
    
    # --- Helper: check for duplicate obs IDs ---
    def _check_duplicates(df, name):
        obs_col = df.iloc[:, 0].astype(str)
        dupes = obs_col[obs_col.duplicated(keep=False)]
        if len(dupes) > 0:
            unique_dupes = dupes.unique().tolist()
            raise ValueError(
                f"{name} has duplicate observation IDs: {unique_dupes[:10]}"
                + (f" ... and {len(unique_dupes)-10} more" if len(unique_dupes) > 10 else ""))
    
    # --- DataFrame-specific validation ---
    if isinstance(X, pd.DataFrame):
        _check_df_structure(X, "X")
        _check_duplicates(X, "X")
    
    if Y is not None and isinstance(Y, pd.DataFrame):
        _check_df_structure(Y, "Y")
        _check_duplicates(Y, "Y")
    
    # --- Validate A against dimensions ---
    if A is not None:
        if isinstance(X, pd.DataFrame):
            n_rows = X.shape[0]
            n_cols = X.shape[1] - 1
        else:
            n_rows = X.shape[0]
            n_cols = X.shape[1]
        max_A = min(n_rows, n_cols)
        if A > max_A:
            raise ValueError(
                f"A={A} exceeds max allowable components for X "
                f"({n_rows} observations x {n_cols} variables, max A={max_A})")
    
    # --- Reconcile X and Y observations ---
    if Y is None:
        return X, None
    
    # Both numpy: can only check row count
    if isinstance(X, np.ndarray) and isinstance(Y, np.ndarray):
        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"X and Y must have the same number of rows. "
                f"X has {X.shape[0]}, Y has {Y.shape[0]}")
        return X, Y
    
    # One numpy, one DataFrame: check row count only
    if isinstance(X, np.ndarray) != isinstance(Y, np.ndarray):
        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"X and Y must have the same number of rows. "
                f"X has {X.shape[0]}, Y has {Y.shape[0]}. "
                f"Cannot reconcile observation order when mixing ndarray and DataFrame.")
        print("Warning: Cannot verify observation alignment when mixing "
              "ndarray and DataFrame. Ensure rows correspond.")
        return X, Y
    
    # Both DataFrames: full reconciliation
    x_obs = X.iloc[:, 0].astype(str).tolist()
    y_obs = Y.iloc[:, 0].astype(str).tolist()
    
    x_set = set(x_obs)
    y_set = set(y_obs)
    
    in_x_only = x_set - y_set
    in_y_only = y_set - x_set
    common    = x_set & y_set
    
    if len(common) == 0:
        raise ValueError(
            "X and Y share no common observation IDs. "
            f"X IDs sample: {x_obs[:5]}, Y IDs sample: {y_obs[:5]}. "
            "Check that the first column contains matching observation identifiers.")
    
    if in_x_only:
        dropped = sorted(in_x_only)
        print(f"Warning: {len(in_x_only)} observation(s) in X not found in Y — dropped: "
              + (', '.join(dropped[:10]) + (f' ... and {len(dropped)-10} more' if len(dropped) > 10 else '')))
        X = X[X.iloc[:, 0].astype(str).isin(common)].reset_index(drop=True)
    
    if in_y_only:
        dropped = sorted(in_y_only)
        print(f"Warning: {len(in_y_only)} observation(s) in Y not found in X — dropped: "
              + (', '.join(dropped[:10]) + (f' ... and {len(dropped)-10} more' if len(dropped) > 10 else '')))
        Y = Y[Y.iloc[:, 0].astype(str).isin(common)].reset_index(drop=True)
    
    x_obs = X.iloc[:, 0].astype(str).tolist()
    y_obs = Y.iloc[:, 0].astype(str).tolist()
    
    if x_obs == y_obs:
        return X, Y
    
    print("Warning: Observation order in Y does not match X. Reordering Y to align with X.")
    y_obs_to_idx = {obs: idx for idx, obs in enumerate(y_obs)}
    new_order = [y_obs_to_idx[obs] for obs in x_obs]
    Y = Y.iloc[new_order].reset_index(drop=True)
    
    return X, Y

# =============================================================================
# Statistical functions (v1.0: scipy.stats replaces lookup tables)
# =============================================================================

def f99(i, j):
    """F-distribution critical value at 99% confidence."""
    if i <= 0 or j <= 0:
        return 1.0
    return float(f_dist.ppf(0.99, i, j))

def f95(i, j):
    """F-distribution critical value at 95% confidence."""
    if i <= 0 or j <= 0:
        return 1.0
    return float(f_dist.ppf(0.95, i, j))

def spe_ci(spe):
    """SPE confidence intervals using chi-squared approximation."""
    spem = np.mean(spe)
    if spem > 1E-16:
        spev = np.var(spe, ddof=1)
        g = spev / (2 * spem)
        h = (2 * spem**2) / spev
        lim95 = g * chi2.ppf(0.95, h)
        lim99 = g * chi2.ppf(0.99, h)
    else:
        lim95 = 0
        lim99 = 0
    return lim95, lim99

def single_score_conf_int(t):
    n = t.shape[0]
    st = np.var(t, ddof=1)
    lim95 = t_dist.ppf(0.975, df=n) * np.sqrt(st)
    lim99 = t_dist.ppf(0.995, df=n) * np.sqrt(st)
    return lim95, lim99

def scores_conf_int_calc(st, N):
    n_points = 100
    cte2 = ((N-1)*(N+1)*(2)) / (N*(N-2))
    f95_ = cte2 * f95(2, N-2)
    f99_ = cte2 * f99(2, N-2)
    xd95 = np.sqrt(f95_ * st[0,0])
    xd99 = np.sqrt(f99_ * st[0,0])
    xd95 = np.linspace(-xd95, xd95, num=n_points)
    xd99 = np.linspace(-xd99, xd99, num=n_points)
    
    st = np.linalg.inv(st)
    s11, s22, s12, s21 = st[0,0], st[1,1], st[0,1], st[1,0]

    a = np.tile(s22, n_points)
    b_ = xd95 * np.tile(s12, n_points) + xd95 * np.tile(s21, n_points)
    c = (xd95**2) * np.tile(s11, n_points) - f95_
    safe_chk = b_**2 - 4*a*c
    safe_chk[safe_chk < 0] = 0
    yd95p = (-b_ + np.sqrt(safe_chk)) / (2*a)
    yd95n = (-b_ - np.sqrt(safe_chk)) / (2*a)
    
    a = np.tile(s22, n_points)
    b_ = xd99 * np.tile(s12, n_points) + xd99 * np.tile(s21, n_points)
    c = (xd99**2) * np.tile(s11, n_points) - f99_
    safe_chk = b_**2 - 4*a*c
    safe_chk[safe_chk < 0] = 0
    yd99p = (-b_ + np.sqrt(safe_chk)) / (2*a)
    yd99n = (-b_ - np.sqrt(safe_chk)) / (2*a)
    
    return xd95, xd99, yd95p, yd95n, yd99p, yd99n

# =============================================================================
# Core utility functions
# =============================================================================

def clean_htmls():
    '''Deletes all .html files in the current directory.'''
    for f in os.listdir('.'):
        if 'html' in f:
            os.remove(f)      
    return

def z2n(X, X_nan_map):
    X[X_nan_map == 1] = np.nan
    return X

def n2z(X):
    X_nan_map = np.isnan(X)            
    if X_nan_map.any():
        X_nan_map       = X_nan_map * 1
        X[X_nan_map==1] = 0
    else:
        X_nan_map       = X_nan_map * 1
    return X, X_nan_map

def mean(X):
    X_nan_map = np.isnan(X)
    X_ = X.copy()
    if X_nan_map.any():
        X_nan_map       = X_nan_map * 1
        X_[X_nan_map==1] = 0
        aux             = np.sum(X_nan_map, axis=0)
        x_mean = np.sum(X_, axis=0, keepdims=1) / (np.ones((1, X_.shape[1])) * X_.shape[0] - aux)
    else:
        x_mean = np.mean(X_, axis=0, keepdims=1)
    return x_mean

def std(X):
    x_mean = mean(X)
    X_nan_map = np.isnan(X)
    if X_nan_map.any():
        X_nan_map             = X_nan_map * 1
        X_                    = X.copy()
        X_[X_nan_map==1]      = 0
        aux_mat               = (X_ - x_mean)**2
        aux_mat[X_nan_map==1] = 0
        aux                   = np.sum(X_nan_map, axis=0)
        x_std = np.sqrt((np.sum(aux_mat, axis=0, keepdims=1)) / (np.ones((1, X_.shape[1])) * (X_.shape[0]-1-aux)))
    else:
        x_std = np.sqrt(np.sum((X - x_mean)**2, axis=0, keepdims=1) / (np.ones((1, X.shape[1])) * (X.shape[0]-1)))
    return x_std
   
def meancenterscale(X, *, mcs=True):
    '''Mean center and scale a matrix. Only works with Numpy matrices.'''
    if isinstance(mcs, bool):
        if mcs:
            x_mean = mean(X)
            x_std  = std(X)
            X      = X - x_mean
            X      = X / x_std
        else:
            x_mean = np.nan
            x_std  = np.nan
    elif mcs == 'center':
         x_mean = mean(X)
         X      = X - x_mean
         x_std  = np.ones((1, X.shape[1]))
    elif mcs == 'autoscale':
         x_std  = std(X) 
         X      = X / x_std
         x_mean = np.zeros((1, X.shape[1]))
    else:
        x_mean = np.nan
        x_std  = np.nan
    return X, x_mean, x_std

def find(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]

# =============================================================================
# PCA
# =============================================================================

def pca(X, A, *, mcs=True, md_algorithm='nipals', force_nipals=False, shush=False, cross_val=0):
    '''Principal Components Analysis.
    
    pca_object = pyphi.pca(X, A, ...)
    
    Args:
        X: DataFrame or Numpy array
        A: Number of Principal Components
    Returns:
        Dictionary with PCA loadings, scores, diagnostics.
    '''      
    X, _ = _validate_inputs(X, Y=None, A=A, mcs=mcs)
    
    if cross_val == 0:
        pcaobj = pca_(X, A, mcs=mcs, md_algorithm=md_algorithm, force_nipals=force_nipals, shush=shush)
        pcaobj['type'] = 'pca'
    elif 0 < cross_val < 100:
        if isinstance(X, np.ndarray):
            X_ = X.copy()
        elif isinstance(X, pd.DataFrame):
            X_ = np.array(X.values[:,1:]).astype(float)
        if isinstance(mcs, bool):
            if mcs:
                X_, x_mean, x_std = meancenterscale(X_)
            else:    
                x_mean = np.zeros((1, X_.shape[1]))
                x_std  = np.ones((1, X_.shape[1]))
        elif mcs == 'center':
            X_, x_mean, x_std = meancenterscale(X_, mcs='center')
        elif mcs == 'autoscale':
            X_, x_mean, x_std = meancenterscale(X_, mcs='autoscale')
            
        X_nan_map = np.isnan(X_)
        not_Xmiss = (~X_nan_map) * 1

        X_, Xnanmap = n2z(X_)
        TSS   = np.sum(X_**2)
        TSSpv = np.sum(X_**2, axis=0)
        cols = X_.shape[1]
        rows = X_.shape[0]
        X_ = z2n(X_, Xnanmap)
        
        for a in range(A):
            if not shush:
                print('Cross validating PC #' + str(a+1))
            not_removed_map = not_Xmiss.copy().reshape(rows*cols, -1)
            Xrnd = np.random.random(X_.shape) * not_Xmiss
            indx = np.argsort(Xrnd.ravel())
            elements_to_remove = int(np.ceil(rows * cols * (cross_val/100)))
            error = np.zeros((rows*cols, 1))
            rounds = 1
            while np.sum(not_removed_map) > 0:
                rounds += 1          
                X_copy = X_.copy()
                if indx.size > elements_to_remove:
                    indx_this = indx[:elements_to_remove]
                    indx = indx[elements_to_remove:]
                else: 
                    indx_this = indx
                X_copy = X_copy.reshape(rows*cols, 1)
                elements_out = X_copy[indx_this]
                X_copy[indx_this] = np.nan
                X_copy = X_copy.reshape(rows, cols)
                not_removed_map[indx_this] = 0
                auxmap = np.sum(np.isnan(X_copy) * 1, axis=1)
                indx2 = np.where(auxmap == X_copy.shape[1])[0].tolist()
                if len(indx2) > 0:
                    X_copy = np.delete(X_copy, indx2, 0)
                pcaobj_ = pca_(X_copy, 1, mcs=False, shush=True)
                xhat = pcaobj_['T'] @ pcaobj_['P'].T
                xhat = np.insert(xhat, indx2, np.nan, axis=0)
                xhat = xhat.reshape(rows*cols, 1)
                error[indx_this] = elements_out - xhat[indx_this]
            error = error.reshape(rows, cols)
            error, dummy = n2z(error)
            PRESSpv = np.sum(error**2, axis=0)
            PRESS   = np.sum(error**2)
            
            if a == 0:
                q2   = 1 - PRESS/TSS
                q2pv = (1 - PRESSpv/TSSpv).reshape(-1, 1)
            else:
                q2   = np.hstack((q2, 1 - PRESS/TSS))
                q2pv = np.hstack((q2pv, (1 - PRESSpv/TSSpv).reshape(-1, 1)))
            
            X_copy = X_.copy()
            pcaobj_ = pca_(X_copy, 1, mcs=False, shush=True)
            xhat = pcaobj_['T'] @ pcaobj_['P'].T
            X_, Xnanmap = n2z(X_)
            X_ = (X_ - xhat) * not_Xmiss
            r2, r2pv = _calc_r2(X_, TSS, TSSpv, 
                                 None if a == 0 else r2, 
                                 None if a == 0 else r2pv)
            X_ = z2n(X_, Xnanmap)
            
        pcaobj = pca_(X, A, mcs=mcs, force_nipals=True, shush=True)
        r2, r2pv = _r2_cumulative_to_per_component(r2, r2pv, A)
        q2, q2pv = _r2_cumulative_to_per_component(q2, q2pv, A)
        r2xc = np.cumsum(r2)
        q2xc = np.cumsum(q2)
        eigs = np.var(pcaobj['T'], axis=0)
        pcaobj['q2']   = q2
        pcaobj['q2pv'] = q2pv
        
        if not shush:   
            print('phi.pca using NIPALS and cross validation (' + str(cross_val) + '%) executed on: ' + str(datetime.datetime.now()))            
            print('--------------------------------------------------------------')
            print('PC #          Eig      R2X     sum(R2X)      Q2X     sum(Q2X)')
            if A > 1:
                for a in range(A):
                    print("PC #"+str(a+1)+":   {:8.3f}    {:.3f}     {:.3f}       {:.3f}     {:.3f}".format(eigs[a], r2[a], r2xc[a], q2[a], q2xc[a]))
            else:
                print("PC #1:   {:8.3f}    {:.3f}     {:.3f}       {:.3f}     {:.3f}".format(eigs[0], r2, r2xc[0], q2, q2xc[0]))
            print('--------------------------------------------------------------')     
        pcaobj['type'] = 'pca'    
    else:
        pcaobj = 'Cannot cross validate with those options'
    return pcaobj

def pca_(X, A, *, mcs=True, md_algorithm='nipals', force_nipals=False, shush=False):
    X_, obsidX, varidX = _extract_array(X)
            
    if isinstance(mcs, bool):
        if mcs:
            X_, x_mean, x_std = meancenterscale(X_)  
        else:    
            x_mean = np.zeros((1, X_.shape[1]))
            x_std  = np.ones((1, X_.shape[1]))
    elif mcs == 'center':
        X_, x_mean, x_std = meancenterscale(X_, mcs='center')
    elif mcs == 'autoscale':
        X_, x_mean, x_std = meancenterscale(X_, mcs='autoscale')
        
    X_nan_map = np.isnan(X_)
    not_Xmiss = (~X_nan_map) * 1
    
    if not X_nan_map.any() and not force_nipals and ((X_.shape[1]/X_.shape[0] >= 10) or (X_.shape[0]/X_.shape[1] >= 10)):
        if not shush:
            print('phi.pca using SVD executed on: ' + str(datetime.datetime.now()))
        TSS   = np.sum(X_**2)
        TSSpv = np.sum(X_**2, axis=0)
        if X_.shape[1]/X_.shape[0] >= 10:
             [U, S, Th] = np.linalg.svd(X_ @ X_.T)
             T = Th.T[:, :A]
             P = X_.T @ T
             for a in range(A):
                 P[:, a] = P[:, a] / np.linalg.norm(P[:, a])
             T = X_ @ P
        elif X_.shape[0]/X_.shape[1] >= 10:
             [U, S, Ph] = np.linalg.svd(X_.T @ X_)
             P = Ph.T[:, :A]
             T = X_ @ P
        r2 = None; r2pv = None
        for a in range(A):
            X_ = X_ - T[:,[a]] @ P[:,[a]].T
            r2, r2pv = _calc_r2(X_, TSS, TSSpv, r2, r2pv)
        r2, r2pv = _r2_cumulative_to_per_component(r2, r2pv, A)
        
        var_t = (T.T @ T) / T.shape[0]
        pca_obj = {'T':T, 'P':P, 'r2x':r2, 'r2xpv':r2pv, 'mx':x_mean, 'sx':x_std, 'var_t':var_t}
        if not isinstance(obsidX, bool):
            pca_obj['obsidX'] = obsidX
            pca_obj['varidX'] = varidX
        eigs = np.var(T, axis=0)
        r2xc = np.cumsum(r2)
        if not shush:
            print('--------------------------------------------------------------')
            print('PC #      Eig        R2X       sum(R2X) ')
            if A > 1:
                for a in range(A):
                    print("PC #"+str(a+1)+":   {:8.3f}    {:.3f}     {:.3f}".format(eigs[a], r2[a], r2xc[a]))
            else:
                print("PC #1:   {:8.3f}    {:.3f}     {:.3f}".format(eigs[0], r2, r2xc[0]))
            print('--------------------------------------------------------------')      
        T2 = hott2(pca_obj, Tnew=T)
        n = T.shape[0]
        T2_lim99 = (((n-1)*(n+1)*A)/(n*(n-A))) * f99(A, (n-A))
        T2_lim95 = (((n-1)*(n+1)*A)/(n*(n-A))) * f95(A, (n-A))     
        speX = np.sum(X_**2, axis=1, keepdims=1)
        speX_lim95, speX_lim99 = spe_ci(speX)
        pca_obj['T2']          = T2
        pca_obj['T2_lim99']    = T2_lim99
        pca_obj['T2_lim95']    = T2_lim95
        pca_obj['speX']        = speX
        pca_obj['speX_lim99']  = speX_lim99
        pca_obj['speX_lim95']  = speX_lim95
        return pca_obj
    else:
        if md_algorithm == 'nipals':
             if not shush:
                 print('phi.pca using NIPALS executed on: ' + str(datetime.datetime.now()))
             X_, dummy = n2z(X_)
             epsilon = 1E-10
             maxit = 5000
             TSS   = np.sum(X_**2)
             TSSpv = np.sum(X_**2, axis=0)
             r2 = None; r2pv = None
             for a in range(A):
                 ti = X_[:, [np.argmax(std(X_))]]
                 Converged = False
                 num_it = 0
                 while not Converged:
                      # p = X't / (t't) with missing data handling
                      pi = np.sum(X_ * ti, axis=0) / np.sum((ti * not_Xmiss)**2, axis=0)
                      pi = pi / np.linalg.norm(pi)
                      # t_new = Xp / (p'p)
                      tn = X_ @ pi
                      ptp = np.sum((pi * not_Xmiss)**2, axis=1)
                      tn = tn / ptp
                      pi = pi.reshape(-1, 1)
                      if abs((np.linalg.norm(ti) - np.linalg.norm(tn))) / np.linalg.norm(ti) < epsilon:
                          Converged = True
                      if num_it > maxit:
                          Converged = True
                      if Converged:
                          if (len(ti[ti<0]) > 0) and (len(ti[ti>0]) > 0):
                              if np.var(ti[ti<0]) > np.var(ti[ti>=0]):
                                 tn = -tn; ti = -ti; pi = -pi 
                          if not shush:
                              print('# Iterations for PC #'+str(a+1)+': ', str(num_it))
                          if a == 0:
                              T = tn.reshape(-1, 1)
                              P = pi
                          else:
                              T = np.hstack((T, tn.reshape(-1, 1)))
                              P = np.hstack((P, pi))                           
                          X_ = (X_ - ti @ pi.T) * not_Xmiss
                          r2, r2pv = _calc_r2(X_, TSS, TSSpv, r2, r2pv)
                      else:
                          num_it += 1
                          ti = tn.reshape(-1, 1)
                 if a == 0:
                     numIT = num_it
                 else:
                     numIT = np.hstack((numIT, num_it))
                     
             r2, r2pv = _r2_cumulative_to_per_component(r2, r2pv, A)
             eigs = np.var(T, axis=0)
             r2xc = np.cumsum(r2)
             if not shush:               
                 print('--------------------------------------------------------------')
                 print('PC #      Eig        R2X       sum(R2X) ')
                 if A > 1:
                     for a in range(A):
                         print("PC #"+str(a+1)+":   {:8.3f}    {:.3f}     {:.3f}".format(eigs[a], r2[a], r2xc[a]))
                 else:
                     print("PC #1:   {:8.3f}    {:.3f}     {:.3f}".format(eigs[0], r2, r2xc[0]))
                 print('--------------------------------------------------------------')      
        
             var_t = (T.T @ T) / T.shape[0]
             pca_obj = {'T':T, 'P':P, 'r2x':r2, 'r2xpv':r2pv, 'mx':x_mean, 'sx':x_std, 'var_t':var_t}    
             if not isinstance(obsidX, bool):
                 pca_obj['obsidX'] = obsidX
                 pca_obj['varidX'] = varidX
                 
             T2 = hott2(pca_obj, Tnew=T)
             n = T.shape[0]
             T2_lim99 = (((n-1)*(n+1)*A)/(n*(n-A))) * f99(A, (n-A))
             T2_lim95 = (((n-1)*(n+1)*A)/(n*(n-A))) * f95(A, (n-A))     
             speX = np.sum(X_**2, axis=1, keepdims=1)
             speX_lim95, speX_lim99 = spe_ci(speX)
             pca_obj['T2']          = T2
             pca_obj['T2_lim99']    = T2_lim99
             pca_obj['T2_lim95']    = T2_lim95
             pca_obj['speX']        = speX
             pca_obj['speX_lim99']  = speX_lim99
             pca_obj['speX_lim95']  = speX_lim95
             return pca_obj                            
        elif md_algorithm == 'nlp' and pyomo_ok:
            if not shush:
                 print('phi.pca using NLP with Ipopt executed on: ' + str(datetime.datetime.now()))
            pcaobj_ = pca_(X, A, mcs=mcs, md_algorithm='nipals', shush=True)
            pcaobj_ = prep_pca_4_MDbyNLP(pcaobj_, X_)
          
            TSS   = np.sum(X_**2)
            TSSpv = np.sum(X_**2, axis=0)
            
            model             = ConcreteModel()
            model.A           = Set(initialize=pcaobj_['pyo_A'])
            model.N           = Set(initialize=pcaobj_['pyo_N'])
            model.O           = Set(initialize=pcaobj_['pyo_O'])
            model.P           = Var(model.N, model.A, within=Reals, initialize=pcaobj_['pyo_P_init'])
            model.T           = Var(model.O, model.A, within=Reals, initialize=pcaobj_['pyo_T_init'])
            model.psi         = Param(model.O, model.N, initialize=pcaobj_['pyo_psi'])
            model.X           = Param(model.O, model.N, initialize=pcaobj_['pyo_X'])
            model.delta       = Param(model.A, model.A, initialize=lambda model, a1, a2: 1.0 if a1==a2 else 0)
            
            def _c20b_con(model, a1, a2):
                return sum(model.P[j, a1] * model.P[j, a2] for j in model.N) == model.delta[a1, a2]
            model.c20b = Constraint(model.A, model.A, rule=_c20b_con)
    
            def _20c_con(model, a1, a2):
                if a2 < a1:
                    return sum(model.T[o, a1] * model.T[o, a2] for o in model.O) == 0
                else:
                    return Constraint.Skip
            model.c20c = Constraint(model.A, model.A, rule=_20c_con)

            def mean_zero(model, i):
                return sum(model.T[o,i] for o in model.O) == 0
            model.eq3 = Constraint(model.A, rule=mean_zero)

            def _eq_20a_obj(model):
                return sum(sum((model.X[o,n] - model.psi[o,n] * sum(model.T[o,a] * model.P[n,a] for a in model.A))**2 for n in model.N) for o in model.O)
            model.obj = Objective(rule=_eq_20a_obj)            

            if ipopt_ok:
                print("Solving NLP using local IPOPT executable")
                solver = SolverFactory('ipopt')
                if ma57_ok:
                    solver.options['linear_solver'] = 'ma57'
                results = solver.solve(model, tee=True)
            elif gams_ok:
                print("Solving NLP using GAMS/IPOPT interface")
                solver = SolverFactory('gams:ipopt')
                results = solver.solve(model, tee=True)
            else:                
                print("Solving NLP using IPOPT on remote NEOS server")
                solver_manager = SolverManagerFactory('neos')
                results = solver_manager.solve(model, opt='ipopt', tee=True)

            T = np.array([[value(model.T[o,a]) for a in model.A] for o in model.O])
            P = np.array([[value(model.P[n,a]) for a in model.A] for n in model.N])
            
            r2 = None; r2pv = None
            for a in range(A):
                 ti = T[:,[a]]; pi = P[:,[a]]
                 if np.var(ti[ti<0]) > np.var(ti[ti>=0]):
                    T[:,[a]] = -T[:,[a]]; P[:,[a]] = -P[:,[a]]
                    ti = -ti; pi = -pi
                 X_ = (X_ - ti @ pi.T) * not_Xmiss
                 r2, r2pv = _calc_r2(X_, TSS, TSSpv, r2, r2pv)
                    
            r2, r2pv = _r2_cumulative_to_per_component(r2, r2pv, A)
            eigs = np.var(T, axis=0)
            r2xc = np.cumsum(r2)
            if not shush:               
                 print('--------------------------------------------------------------')
                 print('PC #      Eig        R2X       sum(R2X) ')
                 if A > 1:
                     for a in range(A):
                         print("PC #"+str(a+1)+":   {:8.3f}    {:.3f}     {:.3f}".format(eigs[a], r2[a], r2xc[a]))
                 else:
                     print("PC #1:   {:8.3f}    {:.3f}     {:.3f}".format(eigs[0], r2, r2xc[0]))
                 print('--------------------------------------------------------------')      
        
            var_t = (T.T @ T) / T.shape[0]
            pca_obj = {'T':T, 'P':P, 'r2x':r2, 'r2xpv':r2pv, 'mx':x_mean, 'sx':x_std, 'var_t':var_t}    
            if not isinstance(obsidX, bool):
                 pca_obj['obsidX'] = obsidX
                 pca_obj['varidX'] = varidX
                 
            T2 = hott2(pca_obj, Tnew=T)
            n = T.shape[0]
            T2_lim99 = (((n-1)*(n+1)*A)/(n*(n-A))) * f99(A, (n-A))
            T2_lim95 = (((n-1)*(n+1)*A)/(n*(n-A))) * f95(A, (n-A))     
            speX = np.sum(X_**2, axis=1, keepdims=1)
            speX_lim95, speX_lim99 = spe_ci(speX)
            pca_obj['T2']          = T2
            pca_obj['T2_lim99']    = T2_lim99
            pca_obj['T2_lim95']    = T2_lim95
            pca_obj['speX']        = speX
            pca_obj['speX_lim99']  = speX_lim99
            pca_obj['speX_lim95']  = speX_lim95
            return pca_obj                  
            
        elif md_algorithm == 'nlp' and not pyomo_ok:
            print('Pyomo was not found in your system sorry')
            print('visit  http://www.pyomo.org/ ')
            return 1

# =============================================================================
# PLS
# =============================================================================

def pls_cca(pls_obj, Xmcs, Ymcs, not_Xmiss):
    Tcv=[]; Pcv=[]; Wcv=[]; Betacv=[]
    firstone = True
    for i in np.arange(Ymcs.shape[1]):
        y_ = Ymcs[:, i].reshape(-1, 1)
        corr, wt, wy = cca(pls_obj['T'], y_)
        t_cv = (pls_obj['T'] @ wt).reshape(-1, 1)
        beta = np.linalg.lstsq(t_cv, y_, rcond=None)[0]
        w_cv = (pls_obj['Ws'] @ wt).reshape(-1, 1)
        p_cv = np.sum(Xmcs * t_cv, axis=0) / np.sum((t_cv * not_Xmiss)**2, axis=0)
        p_cv = p_cv.reshape(-1, 1)
        if firstone:
            Tcv = t_cv; Pcv = p_cv; Wcv = w_cv; Betacv = beta[0][0]
            firstone = False
        else:
            Tcv = np.hstack((Tcv, t_cv))
            Pcv = np.hstack((Pcv, p_cv))
            Wcv = np.hstack((Wcv, w_cv))
            Betacv = np.vstack((Betacv, beta[0][0]))
    return Tcv, Pcv, Wcv, Betacv


def pls(X, Y, A, *, mcsX=True, mcsY=True, md_algorithm='nipals', force_nipals=True, shush=False,
        cross_val=0, cross_val_X=False, cca=False):
    '''Projection to Latent Structures (PLS).
    
    pls_object = pyphi.pls(X, Y, A, ...)
    '''
    X, Y = _validate_inputs(X, Y, A=A, mcs=(mcsX, mcsY))
    
    if cross_val == 0:
        plsobj = pls_(X, Y, A, mcsX=mcsX, mcsY=mcsY, md_algorithm=md_algorithm, force_nipals=force_nipals, shush=shush, cca=cca)  
        plsobj['type'] = 'pls' 
    elif 0 < cross_val < 100:
        if isinstance(X, np.ndarray):
            X_ = X.copy()
        elif isinstance(X, pd.DataFrame):
            X_ = np.array(X.values[:,1:]).astype(float)
        if isinstance(mcsX, bool):
            if mcsX:
                X_, x_mean, x_std = meancenterscale(X_)
            else:    
                x_mean = np.zeros((1, X_.shape[1]))
                x_std  = np.ones((1, X_.shape[1]))
        elif mcsX == 'center':
            X_, x_mean, x_std = meancenterscale(X_, mcs='center')
        elif mcsX == 'autoscale':
            X_, x_mean, x_std = meancenterscale(X_, mcs='autoscale')
        X_nan_map = np.isnan(X_)
        not_Xmiss = (~X_nan_map) * 1
        
        if isinstance(Y, np.ndarray):
            Y_ = Y.copy()
        elif isinstance(Y, pd.DataFrame):
            Y_ = np.array(Y.values[:,1:]).astype(float)
        if isinstance(mcsY, bool):
            if mcsY:
                Y_, y_mean, y_std = meancenterscale(Y_)
            else:    
                y_mean = np.zeros((1, Y_.shape[1]))
                y_std  = np.ones((1, Y_.shape[1]))
        elif mcsY == 'center':
            Y_, y_mean, y_std = meancenterscale(Y_, mcs='center')
        elif mcsY == 'autoscale':
            Y_, y_mean, y_std = meancenterscale(Y_, mcs='autoscale')
        Y_nan_map = np.isnan(Y_)
        not_Ymiss = (~Y_nan_map) * 1
        
        X_, Xnanmap = n2z(X_)
        TSSX   = np.sum(X_**2)
        TSSXpv = np.sum(X_**2, axis=0)
        colsX = X_.shape[1]; rowsX = X_.shape[0]
        X_ = z2n(X_, Xnanmap)
        
        Y_, Ynanmap = n2z(Y_)
        TSSY   = np.sum(Y_**2)
        TSSYpv = np.sum(Y_**2, axis=0)
        colsY = Y_.shape[1]; rowsY = Y_.shape[0]
        Y_ = z2n(Y_, Ynanmap)
        
        r2X = None; r2Xpv = None; r2Y = None; r2Ypv = None
        q2Y = None; q2Ypv = None; q2X = None; q2Xpv = None
        
        for a in range(A):
            if not shush:
                print('Cross validating LV #' + str(a+1))
            not_removed_mapY = not_Ymiss.copy().reshape(rowsY*colsY, -1)
            Yrnd = np.random.random(Y_.shape) * not_Ymiss
            indxY = np.argsort(Yrnd.ravel())
            elements_to_remove_Y = int(np.ceil(rowsY * colsY * (cross_val/100)))
            errorY = np.zeros((rowsY*colsY, 1))
                
            if cross_val_X:
                not_removed_mapX = not_Xmiss.copy().reshape(rowsX*colsX, -1)
                Xrnd = np.random.random(X_.shape) * not_Xmiss
                indxX = np.argsort(Xrnd.ravel())
                elements_to_remove_X = int(np.ceil(rowsX * colsX * (cross_val/100)))
                errorX = np.zeros((rowsX*colsX, 1))
            else:
                not_removed_mapX = 0
                
            number_of_rounds = 1    
            while np.sum(not_removed_mapX) > 0 or np.sum(not_removed_mapY) > 0:
                number_of_rounds += 1
                X_copy = X_.copy()
                if cross_val_X:
                    if indxX.size > elements_to_remove_X:
                        indx_this_roundX = indxX[:elements_to_remove_X]
                        indxX = indxX[elements_to_remove_X:]
                    else: 
                        indx_this_roundX = indxX
                    X_copy = X_copy.reshape(rowsX*colsX, 1)
                    elements_outX = X_copy[indx_this_roundX]
                    X_copy[indx_this_roundX] = np.nan
                    X_copy = X_copy.reshape(rowsX, colsX)
                    not_removed_mapX[indx_this_roundX] = 0
                    auxmap = np.sum(np.isnan(X_copy)*1, axis=1)
                    indx2 = np.where(auxmap == X_copy.shape[1])[0].tolist()
                else:
                    indx2 = []
                        
                Y_copy = Y_.copy()        
                if indxY.size > elements_to_remove_Y:
                    indx_this_roundY = indxY[:elements_to_remove_Y]
                    indxY = indxY[elements_to_remove_Y:]
                else:                      
                    indx_this_roundY = indxY
                Y_copy = Y_copy.reshape(rowsY*colsY, 1)
                elements_outY = Y_copy[indx_this_roundY]
                Y_copy[indx_this_roundY] = np.nan
                Y_copy = Y_copy.reshape(rowsY, colsY)
                not_removed_mapY[indx_this_roundY] = 0
                auxmap = np.sum(np.isnan(Y_copy)*1, axis=1)
                indx3  = np.where(auxmap == Y_copy.shape[1])[0].tolist()
                indx4  = np.unique(indx3 + indx2).tolist()
                if len(indx4) > 0:
                    X_copy = np.delete(X_copy, indx4, 0)
                    Y_copy = np.delete(Y_copy, indx4, 0)
                plsobj_ = pls_(X_copy, Y_copy, 1, mcsX=False, mcsY=False, shush=True)
                plspred = pls_pred(X_, plsobj_)
                
                if cross_val_X:
                    xhat = plspred['Tnew'] @ plsobj_['P'].T
                    xhat = xhat.reshape(rowsX*colsX, 1)
                    errorX[indx_this_roundX] = elements_outX - xhat[indx_this_roundX]
                
                yhat = plspred['Tnew'] @ plsobj_['Q'].T
                yhat = yhat.reshape(rowsY*colsY, 1)
                errorY[indx_this_roundY] = elements_outY - yhat[indx_this_roundY]
                
            if cross_val_X:
                errorX = errorX.reshape(rowsX, colsX)
                errorX, dummy = n2z(errorX)
                PRESSXpv = np.sum(errorX**2, axis=0)
                PRESSX   = np.sum(errorX**2)
            
            errorY = errorY.reshape(rowsY, colsY)
            errorY, dummy = n2z(errorY)
            PRESSYpv = np.sum(errorY**2, axis=0)
            PRESSY   = np.sum(errorY**2)
            
            if a == 0:
                q2Y   = 1 - PRESSY/TSSY
                q2Ypv = (1 - PRESSYpv/TSSYpv).reshape(-1, 1)
                if cross_val_X:
                    q2X   = 1 - PRESSX/TSSX
                    q2Xpv = (1 - PRESSXpv/TSSXpv).reshape(-1, 1)
            else:
                q2Y   = np.hstack((q2Y, 1 - PRESSY/TSSY))
                q2Ypv = np.hstack((q2Ypv, (1 - PRESSYpv/TSSYpv).reshape(-1, 1)))
                if cross_val_X:
                    q2X   = np.hstack((q2X, 1 - PRESSX/TSSX))
                    q2Xpv = np.hstack((q2Xpv, (1 - PRESSXpv/TSSXpv).reshape(-1, 1)))
            
            X_copy = X_.copy(); Y_copy = Y_.copy()
            plsobj_ = pls_(X_copy, Y_copy, 1, mcsX=False, mcsY=False, shush=True)
            xhat = plsobj_['T'] @ plsobj_['P'].T
            yhat = plsobj_['T'] @ plsobj_['Q'].T
            X_, Xnanmap = n2z(X_)
            Y_, Ynanmap = n2z(Y_)
            X_ = (X_ - xhat) * not_Xmiss
            Y_ = (Y_ - yhat) * not_Ymiss
            r2X, r2Xpv = _calc_r2(X_, TSSX, TSSXpv,
                                    None if a == 0 else r2X,
                                    None if a == 0 else r2Xpv)
            r2Y, r2Ypv = _calc_r2(Y_, TSSY, TSSYpv,
                                    None if a == 0 else r2Y,
                                    None if a == 0 else r2Ypv)
            X_ = z2n(X_, Xnanmap)
            Y_ = z2n(Y_, Ynanmap)
            
        plsobj = pls_(X, Y, A, mcsX=mcsX, mcsY=mcsY, shush=True, cca=cca)
        r2X, r2Xpv = _r2_cumulative_to_per_component(r2X, r2Xpv, A)
        r2Y, r2Ypv = _r2_cumulative_to_per_component(r2Y, r2Ypv, A)
        if cross_val_X:
            q2X, q2Xpv = _r2_cumulative_to_per_component(q2X, q2Xpv, A)
        else:
            q2X = False; q2Xpv = False
        q2Y, q2Ypv = _r2_cumulative_to_per_component(q2Y, q2Ypv, A)
             
        r2xc = np.cumsum(r2X); r2yc = np.cumsum(r2Y)
        q2xc = np.cumsum(q2X) if cross_val_X else False
        q2yc = np.cumsum(q2Y)    
        eigs = np.var(plsobj['T'], axis=0)
        
        plsobj['q2Y']   = q2Y
        plsobj['q2Ypv'] = q2Ypv
        if cross_val_X:
            plsobj['q2X']   = q2X
            plsobj['q2Xpv'] = q2Xpv
        
        if not shush:
            print('phi.pls using NIPALS and cross-validation (' + str(cross_val) + '%) executed on: ' + str(datetime.datetime.now()))
            if not cross_val_X:
                print('---------------------------------------------------------------------------------')
                print('PC #       Eig      R2X     sum(R2X)      R2Y     sum(R2Y)      Q2Y     sum(Q2Y)')
                if A > 1:
                    for a in range(A):
                        print("PC #"+str(a+1)+":{:8.3f}    {:.3f}     {:.3f}       {:.3f}     {:.3f}       {:.3f}     {:.3f}".format(eigs[a], r2X[a], r2xc[a], r2Y[a], r2yc[a], q2Y[a], q2yc[a]))
                else:
                    print("PC #1:{:8.3f}    {:.3f}     {:.3f}       {:.3f}        {:.3f}    {:.3f}     {:.3f}".format(eigs[0], r2X, r2xc[0], r2Y, r2yc[0], q2Y, q2yc[0]))
                print('---------------------------------------------------------------------------------')     
            else:
                print('-------------------------------------------------------------------------------------------------------')
                print('PC #       Eig      R2X     sum(R2X)      Q2X     sum(Q2X)      R2Y     sum(R2Y)      Q2Y     sum(Q2Y)')
                if A > 1:
                    for a in range(A):
                        print("PC #"+str(a+1)+":{:8.3f}    {:.3f}     {:.3f}       {:.3f}     {:.3f}       {:.3f}     {:.3f}       {:.3f}     {:.3f}".format(eigs[a], r2X[a], r2xc[a], q2X[a], q2xc[a], r2Y[a], r2yc[a], q2Y[a], q2yc[a]))
                else:
                    print("PC #1:{:8.3f}    {:.3f}     {:.3f}       {:.3f}     {:.3f}       {:.3f}     {:.3f}       {:.3f}     {:.3f}".format(eigs[0], r2X, r2xc[0], q2X, q2xc[0], r2Y, r2yc[0], q2Y, q2yc[0]))
                print('-------------------------------------------------------------------------------------------------------')   
        plsobj['type'] = 'pls'

    elif cross_val == 100:
        # LOO cross-validation  
        if isinstance(X, np.ndarray):
            X_ = X.copy()
        elif isinstance(X, pd.DataFrame):
            X_ = np.array(X.values[:,1:]).astype(float)
        if isinstance(mcsX, bool):
            if mcsX:
                X_, x_mean, x_std = meancenterscale(X_)
            else:    
                x_mean = np.zeros((1, X_.shape[1]))
                x_std  = np.ones((1, X_.shape[1]))
        elif mcsX == 'center':
            X_, x_mean, x_std = meancenterscale(X_, mcs='center')
        elif mcsX == 'autoscale':
            X_, x_mean, x_std = meancenterscale(X_, mcs='autoscale')
        X_nan_map = np.isnan(X_)
        not_Xmiss = (~X_nan_map) * 1
        
        if isinstance(Y, np.ndarray):
            Y_ = Y.copy()
        elif isinstance(Y, pd.DataFrame):
            Y_ = np.array(Y.values[:,1:]).astype(float)
        if isinstance(mcsY, bool):
            if mcsY:
                Y_, y_mean, y_std = meancenterscale(Y_)
            else:    
                y_mean = np.zeros((1, Y_.shape[1]))
                y_std  = np.ones((1, Y_.shape[1]))
        elif mcsY == 'center':
            Y_, y_mean, y_std = meancenterscale(Y_, mcs='center')
        elif mcsY == 'autoscale':
            Y_, y_mean, y_std = meancenterscale(Y_, mcs='autoscale')
        Y_nan_map = np.isnan(Y_)
        not_Ymiss = (~Y_nan_map) * 1
        
        X_, Xnanmap = n2z(X_)
        TSSX   = np.sum(X_**2)
        TSSXpv = np.sum(X_**2, axis=0)
        colsX = X_.shape[1]; rowsX = X_.shape[0]
        X_ = z2n(X_, Xnanmap)
        
        Y_, Ynanmap = n2z(Y_)
        TSSY   = np.sum(Y_**2)
        TSSYpv = np.sum(Y_**2, axis=0)
        colsY = Y_.shape[1]; rowsY = Y_.shape[0]
        Y_ = z2n(Y_, Ynanmap)
        
        r2X = None; r2Xpv = None; r2Y = None; r2Ypv = None
        
        for a in range(A):
            errorY = np.zeros((rowsY*colsY, 1))               
            if cross_val_X:
                errorX = np.zeros((rowsX*colsX, 1))
                
            for o in range(X_.shape[0]):
                X_copy = X_.copy(); Y_copy = Y_.copy()   
                elements_outX = X_copy[o, :].copy()
                elements_outY = Y_copy[o, :].copy()
                X_copy = np.delete(X_copy, o, 0)
                Y_copy = np.delete(Y_copy, o, 0)
                plsobj_ = pls_(X_copy, Y_copy, 1, mcsX=False, mcsY=False, shush=True)
                plspred = pls_pred(elements_outX, plsobj_)
                if o == 0:
                    if cross_val_X:
                        errorX_mat = elements_outX - plspred['Xhat']
                    errorY_mat = elements_outY - plspred['Yhat']
                else:
                    if cross_val_X:
                        errorX_mat = np.vstack((errorX_mat, elements_outX - plspred['Xhat']))
                    errorY_mat = np.vstack((errorY_mat, elements_outY - plspred['Yhat']))
                  
            if cross_val_X:
                errorX_mat, dummy = n2z(errorX_mat)
                PRESSXpv = np.sum(errorX_mat**2, axis=0)
                PRESSX   = np.sum(errorX_mat**2)
            
            errorY_mat, dummy = n2z(errorY_mat)
            PRESSYpv = np.sum(errorY_mat**2, axis=0)
            PRESSY   = np.sum(errorY_mat**2)
            
            if a == 0:
                q2Y   = 1 - PRESSY/TSSY
                q2Ypv = (1 - PRESSYpv/TSSYpv).reshape(-1, 1)
                if cross_val_X:
                    q2X   = 1 - PRESSX/TSSX
                    q2Xpv = (1 - PRESSXpv/TSSXpv).reshape(-1, 1)
            else:
                q2Y   = np.hstack((q2Y, 1 - PRESSY/TSSY))
                q2Ypv = np.hstack((q2Ypv, (1 - PRESSYpv/TSSYpv).reshape(-1, 1)))
                if cross_val_X:
                    q2X   = np.hstack((q2X, 1 - PRESSX/TSSX))
                    q2Xpv = np.hstack((q2Xpv, (1 - PRESSXpv/TSSXpv).reshape(-1, 1)))
            
            X_copy = X_.copy(); Y_copy = Y_.copy()
            plsobj_ = pls_(X_copy, Y_copy, 1, mcsX=False, mcsY=False, shush=True)
            xhat = plsobj_['T'] @ plsobj_['P'].T
            yhat = plsobj_['T'] @ plsobj_['Q'].T
            X_, Xnanmap = n2z(X_)
            Y_, Ynanmap = n2z(Y_)
            X_ = (X_ - xhat) * not_Xmiss
            Y_ = (Y_ - yhat) * not_Ymiss
            r2X, r2Xpv = _calc_r2(X_, TSSX, TSSXpv,
                                    None if a == 0 else r2X,
                                    None if a == 0 else r2Xpv)
            r2Y, r2Ypv = _calc_r2(Y_, TSSY, TSSYpv,
                                    None if a == 0 else r2Y,
                                    None if a == 0 else r2Ypv)
            X_ = z2n(X_, Xnanmap)
            Y_ = z2n(Y_, Ynanmap)
            
        plsobj = pls_(X, Y, A, mcsX=mcsX, mcsY=mcsY, shush=True, cca=cca)
        r2X, r2Xpv = _r2_cumulative_to_per_component(r2X, r2Xpv, A)
        r2Y, r2Ypv = _r2_cumulative_to_per_component(r2Y, r2Ypv, A)
        if cross_val_X:
            q2X, q2Xpv = _r2_cumulative_to_per_component(q2X, q2Xpv, A)
        else:
            q2X = False; q2Xpv = False
        q2Y, q2Ypv = _r2_cumulative_to_per_component(q2Y, q2Ypv, A)

        r2xc = np.cumsum(r2X); r2yc = np.cumsum(r2Y)
        q2xc = np.cumsum(q2X) if cross_val_X else False
        q2yc = np.cumsum(q2Y)    
        eigs = np.var(plsobj['T'], axis=0)
        
        plsobj['q2Y']   = q2Y
        plsobj['q2Ypv'] = q2Ypv
        plsobj['type']  = 'pls'
        if cross_val_X:
            plsobj['q2X']   = q2X
            plsobj['q2Xpv'] = q2Xpv
        
        if not shush:
            if not cross_val_X:
                print('---------------------------------------------------------------------------------')
                print('PC #       Eig      R2X     sum(R2X)      R2Y     sum(R2Y)      Q2Y     sum(Q2Y)')
                if A > 1:
                    for a in range(A):
                        print("PC #"+str(a+1)+":{:8.3f}    {:.3f}     {:.3f}       {:.3f}     {:.3f}       {:.3f}     {:.3f}".format(eigs[a], r2X[a], r2xc[a], r2Y[a], r2yc[a], q2Y[a], q2yc[a]))
                else:
                    print("PC #1:{:8.3f}    {:.3f}     {:.3f}       {:.3f}     {:.3f}       {:.3f}     {:.3f}".format(eigs[0], r2X, r2xc[0], r2Y, r2yc[0], q2Y, q2yc[0]))
                print('---------------------------------------------------------------------------------')     
            else:
                print('-------------------------------------------------------------------------------------------------------')
                print('PC #       Eig      R2X     sum(R2X)      Q2X     sum(Q2X)      R2Y     sum(R2Y)      Q2Y     sum(Q2Y)')
                if A > 1:
                    for a in range(A):
                        print("PC #"+str(a+1)+":{:8.3f}    {:.3f}     {:.3f}       {:.3f}     {:.3f}       {:.3f}     {:.3f}       {:.3f}     {:.3f}".format(eigs[a], r2X[a], r2xc[a], q2X[a], q2xc[a], r2Y[a], r2yc[a], q2Y[a], q2yc[a]))
                else:
                    print("PC #1:{:8.3f}    {:.3f}     {:.3f}       {:.3f}     {:.3f}       {:.3f}     {:.3f}       {:.3f}     {:.3f}".format(eigs[0], r2X, r2xc[0], q2X, q2xc[0], r2Y, r2yc[0], q2Y, q2yc[0]))
                print('-------------------------------------------------------------------------------------------------------')   
    else:
        plsobj = 'Cannot cross validate with those options'
    return plsobj    


def pls_(X, Y, A, *, mcsX=True, mcsY=True, md_algorithm='nipals', force_nipals=True, shush=False, cca=False):
    X_, obsidX, varidX = _extract_array(X)
    Y_, obsidY, varidY = _extract_array(Y)

    if isinstance(mcsX, bool):
        if mcsX:
            X_, x_mean, x_std = meancenterscale(X_)
        else:    
            x_mean = np.zeros((1, X_.shape[1]))
            x_std  = np.ones((1, X_.shape[1]))
    elif mcsX == 'center':
        X_, x_mean, x_std = meancenterscale(X_, mcs='center')
    elif mcsX == 'autoscale':
        X_, x_mean, x_std = meancenterscale(X_, mcs='autoscale')
        
    if isinstance(mcsY, bool):
        if mcsY:
            Y_, y_mean, y_std = meancenterscale(Y_)
        else:    
            y_mean = np.zeros((1, Y_.shape[1]))
            y_std  = np.ones((1, Y_.shape[1]))
    elif mcsY == 'center':
        Y_, y_mean, y_std = meancenterscale(Y_, mcs='center')
    elif mcsY == 'autoscale':
        Y_, y_mean, y_std = meancenterscale(Y_, mcs='autoscale')
   
    X_nan_map = np.isnan(X_)
    not_Xmiss = (~X_nan_map) * 1
    Y_nan_map = np.isnan(Y_)
    not_Ymiss = (~Y_nan_map) * 1
    
    if (not X_nan_map.any() and not Y_nan_map.any()) and not force_nipals:
        if cca:    
           Xmcs = X_.copy(); Ymcs = Y_.copy() 
        if not shush:
            print('phi.pls using SVD executed on: ' + str(datetime.datetime.now()))
        TSSX   = np.sum(X_**2); TSSXpv = np.sum(X_**2, axis=0)
        TSSY   = np.sum(Y_**2); TSSYpv = np.sum(Y_**2, axis=0)
        
        r2X = None; r2Xpv = None; r2Y = None; r2Ypv = None
        for a in range(A):
            [U_, S, Wh] = np.linalg.svd((X_.T @ Y_) @ (Y_.T @ X_))
            w = Wh.T[:, [0]]
            t = X_ @ w
            q = Y_.T @ t / (t.T @ t)
            u = Y_ @ q / (q.T @ q)
            p = X_.T @ t / (t.T @ t)
            X_ = X_ - t @ p.T
            Y_ = Y_ - t @ q.T
            
            if a == 0:
                W = w; T = t; Q = q; U = u; P = p
            else:
                W = np.hstack((W, w)); T = np.hstack((T, t))
                Q = np.hstack((Q, q)); U = np.hstack((U, u)); P = np.hstack((P, p))
            r2X, r2Xpv = _calc_r2(X_, TSSX, TSSXpv, r2X, r2Xpv)
            r2Y, r2Ypv = _calc_r2(Y_, TSSY, TSSYpv, r2Y, r2Ypv)
                
        r2X, r2Xpv = _r2_cumulative_to_per_component(r2X, r2Xpv, A)
        r2Y, r2Ypv = _r2_cumulative_to_per_component(r2Y, r2Ypv, A)
        Ws = W @ np.linalg.pinv(P.T @ W)
        Ws[:, 0] = W[:, 0]
        eigs = np.var(T, axis=0)
        r2xc = np.cumsum(r2X); r2yc = np.cumsum(r2Y)
        if not shush:
            print('--------------------------------------------------------------')
            print('LV #     Eig       R2X       sum(R2X)   R2Y       sum(R2Y)')
            if A > 1:    
                for a in range(A):
                    print("LV #"+str(a+1)+":   {:6.3f}    {:.3f}     {:.3f}      {:.3f}     {:.3f}".format(eigs[a], r2X[a], r2xc[a], r2Y[a], r2yc[a]))
            else:
                print("LV #1:   {:6.3f}    {:.3f}     {:.3f}      {:.3f}     {:.3f}".format(eigs[0], r2X, r2xc[0], r2Y, r2yc[0]))
            print('--------------------------------------------------------------')   
        
        var_t = (T.T @ T) / T.shape[0]
        pls_obj = {'T':T, 'P':P, 'Q':Q, 'W':W, 'Ws':Ws, 'U':U,
                   'r2x':r2X, 'r2xpv':r2Xpv, 'mx':x_mean, 'sx':x_std,
                   'r2y':r2Y, 'r2ypv':r2Ypv, 'my':y_mean, 'sy':y_std, 'var_t':var_t}  
        if not isinstance(obsidX, bool):
            pls_obj['obsidX'] = obsidX; pls_obj['varidX'] = varidX
        if not isinstance(obsidY, bool):
            pls_obj['obsidY'] = obsidY; pls_obj['varidY'] = varidY
            
        T2 = hott2(pls_obj, Tnew=T)
        n  = T.shape[0]
        T2_lim99 = (((n-1)*(n+1)*A)/(n*(n-A))) * f99(A, (n-A))
        T2_lim95 = (((n-1)*(n+1)*A)/(n*(n-A))) * f95(A, (n-A))     
        speX = np.sum(X_**2, axis=1, keepdims=1)
        speX_lim95, speX_lim99 = spe_ci(speX)
        speY = np.sum(Y_**2, axis=1, keepdims=1)
        speY_lim95, speY_lim99 = spe_ci(speY)
        pls_obj['T2'] = T2; pls_obj['T2_lim99'] = T2_lim99; pls_obj['T2_lim95'] = T2_lim95
        pls_obj['speX'] = speX; pls_obj['speX_lim99'] = speX_lim99; pls_obj['speX_lim95'] = speX_lim95
        pls_obj['speY'] = speY; pls_obj['speY_lim99'] = speY_lim99; pls_obj['speY_lim95'] = speY_lim95
        if cca:
            Tcv, Pcv, Wcv, Betacv = pls_cca(pls_obj, Xmcs, Ymcs, not_Xmiss)
            pls_obj['Tcv'] = Tcv; pls_obj['Pcv'] = Pcv; pls_obj['Wcv'] = Wcv; pls_obj['Betacv'] = Betacv
        return pls_obj
    else:
        if md_algorithm == 'nipals':
             if not shush:
                 print('phi.pls using NIPALS executed on: ' + str(datetime.datetime.now()))
             X_, dummy = n2z(X_)
             Y_, dummy = n2z(Y_)
             epsilon = 1E-9; maxit = 2000
             if cca:    
                 Xmcs = X_.copy(); Ymcs = Y_.copy() 

             TSSX = np.sum(X_**2); TSSXpv = np.sum(X_**2, axis=0)
             TSSY = np.sum(Y_**2); TSSYpv = np.sum(Y_**2, axis=0)
             
             r2X = None; r2Xpv = None; r2Y = None; r2Ypv = None
             for a in range(A):
                 ui = Y_[:, [np.argmax(std(Y_))]]
                 Converged = False; num_it = 0
                 while not Converged:
                      # w = X'u / (u'u)
                      wi = np.sum(X_ * ui, axis=0) / np.sum((ui * not_Xmiss)**2, axis=0)
                      wi = wi / np.linalg.norm(wi)
                      # t = Xw / (w'w)
                      ti = X_ @ wi
                      wtw = np.sum((wi * not_Xmiss)**2, axis=1)
                      ti = ti / wtw
                      ti = ti.reshape(-1, 1); wi = wi.reshape(-1, 1)
                      # q = Y't / (t't)
                      qi = np.sum(Y_ * ti, axis=0) / np.sum((ti * not_Ymiss)**2, axis=0)
                      # u = Yq / (q'q)
                      qi_1d = qi.copy()
                      qi = qi.reshape(-1, 1)
                      un = Y_ @ qi
                      qtq = np.sum((qi_1d * not_Ymiss)**2, axis=1).reshape(-1, 1)
                      un = un / qtq
                      
                      if abs((np.linalg.norm(ui) - np.linalg.norm(un))) / np.linalg.norm(ui) < epsilon:
                          Converged = True
                      if num_it > maxit:
                          Converged = True
                      if Converged:
                          if np.var(ti[ti<0]) > np.var(ti[ti>=0]):
                             ti = -ti; wi = -wi; un = -un; qi = -qi
                          if not shush:
                              print('# Iterations for LV #'+str(a+1)+': ', str(num_it))
                          # p = X't / (t't)
                          pi = np.sum(X_ * ti, axis=0) / np.sum((ti * not_Xmiss)**2, axis=0)
                          pi = pi.reshape(-1, 1)
                          X_ = (X_ - ti @ pi.T) * not_Xmiss
                          Y_ = (Y_ - ti @ qi.T) * not_Ymiss
                          
                          if a == 0:
                              T = ti; P = pi; W = wi; U = un; Q = qi
                          else:
                              T = np.hstack((T, ti)); U = np.hstack((U, un))
                              P = np.hstack((P, pi)); W = np.hstack((W, wi)); Q = np.hstack((Q, qi))
                          r2X, r2Xpv = _calc_r2(X_, TSSX, TSSXpv, r2X, r2Xpv)
                          r2Y, r2Ypv = _calc_r2(Y_, TSSY, TSSYpv, r2Y, r2Ypv)
                      else:
                          num_it += 1
                          ui = un
                 if a == 0:
                     numIT = num_it
                 else:
                     numIT = np.hstack((numIT, num_it))
                     
             r2X, r2Xpv = _r2_cumulative_to_per_component(r2X, r2Xpv, A)
             r2Y, r2Ypv = _r2_cumulative_to_per_component(r2Y, r2Ypv, A)
             
             Ws = W @ np.linalg.pinv(P.T @ W)
             Ws[:, 0] = W[:, 0]
             eigs = np.var(T, axis=0)
             r2xc = np.cumsum(r2X); r2yc = np.cumsum(r2Y)
             if not shush:
                 print('--------------------------------------------------------------')
                 print('LV #     Eig       R2X       sum(R2X)   R2Y       sum(R2Y)')
                 if A > 1:    
                     for a in range(A):
                         print("LV #"+str(a+1)+":   {:6.3f}    {:.3f}     {:.3f}      {:.3f}     {:.3f}".format(eigs[a], r2X[a], r2xc[a], r2Y[a], r2yc[a]))
                 else:
                    print("LV #1:   {:6.3f}    {:.3f}     {:.3f}      {:.3f}     {:.3f}".format(eigs[0], r2X, r2xc[0], r2Y, r2yc[0]))
                 print('--------------------------------------------------------------')   
                       
             var_t = (T.T @ T) / T.shape[0]
             pls_obj = {'T':T, 'P':P, 'Q':Q, 'W':W, 'Ws':Ws, 'U':U,
                        'r2x':r2X, 'r2xpv':r2Xpv, 'mx':x_mean, 'sx':x_std,
                        'r2y':r2Y, 'r2ypv':r2Ypv, 'my':y_mean, 'sy':y_std, 'var_t':var_t}  
             if not isinstance(obsidX, bool):
                 pls_obj['obsidX'] = obsidX; pls_obj['varidX'] = varidX
             if not isinstance(obsidY, bool):
                pls_obj['obsidY'] = obsidY; pls_obj['varidY'] = varidY
                
             T2 = hott2(pls_obj, Tnew=T)
             n  = T.shape[0]
             T2_lim99 = (((n-1)*(n+1)*A)/(n*(n-A))) * f99(A, (n-A))
             T2_lim95 = (((n-1)*(n+1)*A)/(n*(n-A))) * f95(A, (n-A))     
             speX = np.sum(X_**2, axis=1, keepdims=1)
             speX_lim95, speX_lim99 = spe_ci(speX)
             speY = np.sum(Y_**2, axis=1, keepdims=1)
             speY_lim95, speY_lim99 = spe_ci(speY)
             pls_obj['T2'] = T2; pls_obj['T2_lim99'] = T2_lim99; pls_obj['T2_lim95'] = T2_lim95
             pls_obj['speX'] = speX; pls_obj['speX_lim99'] = speX_lim99; pls_obj['speX_lim95'] = speX_lim95
             pls_obj['speY'] = speY; pls_obj['speY_lim99'] = speY_lim99; pls_obj['speY_lim95'] = speY_lim95
             if cca:
                 Tcv, Pcv, Wcv, Betacv = pls_cca(pls_obj, Xmcs, Ymcs, not_Xmiss)
                 pls_obj['Tcv'] = Tcv; pls_obj['Pcv'] = Pcv; pls_obj['Wcv'] = Wcv; pls_obj['Betacv'] = Betacv
             return pls_obj   
                         
        elif md_algorithm == 'nlp':
            shush = False         
            if not shush:
                 print('phi.pls using NLP with Ipopt executed on: ' + str(datetime.datetime.now()))
            X_, dummy = n2z(X_)
            Y_, dummy = n2z(Y_)
            plsobj_ = pls_(X, Y, A, mcsX=mcsX, mcsY=mcsY, md_algorithm='nipals', shush=True)
            plsobj_ = prep_pls_4_MDbyNLP(plsobj_, X_, Y_)
              
            TSSX = np.sum(X_**2); TSSXpv = np.sum(X_**2, axis=0)
            TSSY = np.sum(Y_**2); TSSYpv = np.sum(Y_**2, axis=0)
            
            model = ConcreteModel()
            model.A     = Set(initialize=plsobj_['pyo_A'])
            model.N     = Set(initialize=plsobj_['pyo_N'])
            model.M     = Set(initialize=plsobj_['pyo_M'])
            model.O     = Set(initialize=plsobj_['pyo_O'])
            model.P     = Var(model.N, model.A, within=Reals, initialize=plsobj_['pyo_P_init'])
            model.T     = Var(model.O, model.A, within=Reals, initialize=plsobj_['pyo_T_init'])
            model.psi   = Param(model.O, model.N, initialize=plsobj_['pyo_psi'])
            model.X     = Param(model.O, model.N, initialize=plsobj_['pyo_X'])
            model.theta = Param(model.O, model.M, initialize=plsobj_['pyo_theta'])
            model.Y     = Param(model.O, model.M, initialize=plsobj_['pyo_Y'])           
            model.delta = Param(model.A, model.A, initialize=lambda model, a1, a2: 1.0 if a1==a2 else 0)
            
            def _c27bc_con(model, a1, a2):
                return sum(model.P[j, a1] * model.P[j, a2] for j in model.N) == model.delta[a1, a2]
            model.c27bc = Constraint(model.A, model.A, rule=_c27bc_con)
            def _27d_con(model, a1, a2):
                if a2 < a1:
                    return sum(model.T[o, a1] * model.T[o, a2] for o in model.O) == 0
                else:
                    return Constraint.Skip
            model.c27d = Constraint(model.A, model.A, rule=_27d_con)
            def _27e_con(model, i):
                return sum(model.T[o,i] for o in model.O) == 0
            model.c27e = Constraint(model.A, rule=_27e_con)
            def _eq_27a_obj(model):
                return sum(sum(sum((model.theta[o,m]*model.Y[o,m]) * (model.X[o,n] - model.psi[o,n] * sum(model.T[o,a] * model.P[n,a] for a in model.A)) for o in model.O)**2 for n in model.N) for m in model.M)
            model.obj = Objective(rule=_eq_27a_obj)
            
            if ipopt_ok:
                solver = SolverFactory('ipopt')
                if ma57_ok: solver.options['linear_solver'] = 'ma57'
                results = solver.solve(model, tee=True)
            elif gams_ok:
                solver = SolverFactory('gams:ipopt')
                results = solver.solve(model, tee=True)
            else:
                solver_manager = SolverManagerFactory('neos')
                results = solver_manager.solve(model, opt='ipopt', tee=True)
            
            T = np.array([[value(model.T[o,a]) for a in model.A] for o in model.O])
            P = np.array([[value(model.P[n,a]) for a in model.A] for n in model.N])
            
            # Obtain Ws via NLP
            Taux = np2D2pyomo(T)
            modelb       = ConcreteModel()
            modelb.A     = Set(initialize=plsobj_['pyo_A'])
            modelb.N     = Set(initialize=plsobj_['pyo_N'])
            modelb.O     = Set(initialize=plsobj_['pyo_O'])
            modelb.Ws    = Var(model.N, model.A, within=Reals, initialize=plsobj_['pyo_Ws_init'])
            modelb.T     = Param(model.O, model.A, within=Reals, initialize=Taux)
            modelb.psi   = Param(model.O, model.N, initialize=plsobj_['pyo_psi'])
            modelb.X     = Param(model.O, model.N, initialize=plsobj_['pyo_X'])
            def _eq_obj(model):
                return sum(sum((model.T[o,a] - sum(model.psi[o,n] * model.X[o,n] * model.Ws[n,a] for n in model.N))**2 for a in model.A) for o in model.O)
            modelb.obj = Objective(rule=_eq_obj)
            if ipopt_ok:
                solver = SolverFactory('ipopt')
                if ma57_ok: solver.options['linear_solver'] = 'ma57'
                results = solver.solve(modelb, tee=True)
            elif gams_ok:
                solver = SolverFactory('gams:ipopt')
                results = solver.solve(modelb, tee=True)
            else:
                solver_manager = SolverManagerFactory('neos')
                results = solver_manager.solve(modelb, opt='ipopt', tee=True)

            Ws = np.array([[value(modelb.Ws[n,a]) for a in modelb.A] for n in modelb.N])
            
            # Obtain Q via NLP
            Xhat = T @ P.T
            Xaux = X_.copy(); Xaux[X_nan_map] = Xhat[X_nan_map]
            Xaux = np2D2pyomo(Xaux); Taux = np2D2pyomo(T)
            
            model2       = ConcreteModel()
            model2.A     = Set(initialize=plsobj_['pyo_A'])
            model2.N     = Set(initialize=plsobj_['pyo_N'])
            model2.M     = Set(initialize=plsobj_['pyo_M'])
            model2.O     = Set(initialize=plsobj_['pyo_O'])
            model2.T     = Param(model.O, model.A, within=Reals, initialize=Taux)
            model2.Q     = Var(model.M, model.A, within=Reals, initialize=plsobj_['pyo_Q_init'])
            model2.X     = Param(model.O, model.N, initialize=plsobj_['pyo_X'])
            model2.theta = Param(model.O, model.M, initialize=plsobj_['pyo_theta'])
            model2.Y     = Param(model.O, model.M, initialize=plsobj_['pyo_Y'])           
            model2.delta = Param(model.A, model.A, initialize=lambda model, a1, a2: 1.0 if a1==a2 else 0)
            
            def _eq_36a_mod_obj(model):
                return sum(sum(sum((model.X[o,n]) * (model.Y[o,m] - model.theta[o,m] * sum(model.T[o,a] * model.Q[m,a] for a in model.A)) for o in model.O)**2 for n in model.N) for m in model.M)
            model2.obj = Objective(rule=_eq_36a_mod_obj)
            
            if ipopt_ok:
                solver = SolverFactory('ipopt')
                if ma57_ok: solver.options['linear_solver'] = 'ma57'
                results = solver.solve(model2, tee=True)
            elif gams_ok:
                solver = SolverFactory('gams:ipopt')
                results = solver.solve(model2, tee=True)
            else:
                solver_manager = SolverManagerFactory('neos')
                results = solver_manager.solve(model2, opt='ipopt', tee=True)
               
            Q = np.array([[value(model2.Q[m,a]) for a in model2.A] for m in model2.M])
            
            r2X = None; r2Xpv = None; r2Y = None; r2Ypv = None
            for a in range(A):
                 ti = T[:,[a]]; pi = P[:,[a]]; qi = Q[:,[a]]
                 X_ = (X_ - ti @ pi.T) * not_Xmiss
                 Y_ = (Y_ - ti @ qi.T) * not_Ymiss
                 r2X, r2Xpv = _calc_r2(X_, TSSX, TSSXpv, r2X, r2Xpv)
                 r2Y, r2Ypv = _calc_r2(Y_, TSSY, TSSYpv, r2Y, r2Ypv)
                    
            r2X, r2Xpv = _r2_cumulative_to_per_component(r2X, r2Xpv, A)
            r2Y, r2Ypv = _r2_cumulative_to_per_component(r2Y, r2Ypv, A)
                 
            eigs = np.var(T, axis=0)
            r2xc = np.cumsum(r2X); r2yc = np.cumsum(r2Y)
            if not shush:
                print('--------------------------------------------------------------')
                print('LV #     Eig       R2X       sum(R2X)   R2Y       sum(R2Y)')
                if A > 1:    
                    for a in range(A):
                        print("LV #"+str(a+1)+":   {:6.3f}    {:.3f}     {:.3f}      {:.3f}     {:.3f}".format(eigs[a], r2X[a], r2xc[a], r2Y[a], r2yc[a]))
                else:
                    print("LV #1:   {:6.3f}    {:.3f}     {:.3f}      {:.3f}     {:.3f}".format(eigs[0], r2X, r2xc[0], r2Y, r2yc[0]))
                print('--------------------------------------------------------------')   
            W = 1; U = 1
            var_t = (T.T @ T) / T.shape[0]
            pls_obj = {'T':T, 'P':P, 'Q':Q, 'W':W, 'Ws':Ws, 'U':U,
                       'r2x':r2X, 'r2xpv':r2Xpv, 'mx':x_mean, 'sx':x_std,
                       'r2y':r2Y, 'r2ypv':r2Ypv, 'my':y_mean, 'sy':y_std, 'var_t':var_t}  
            if not isinstance(obsidX, bool):
                pls_obj['obsidX'] = obsidX; pls_obj['varidX'] = varidX
            if not isinstance(obsidY, bool):
                pls_obj['obsidY'] = obsidY; pls_obj['varidY'] = varidY
                
            T2 = hott2(pls_obj, Tnew=T)
            n  = T.shape[0]
            T2_lim99 = (((n-1)*(n+1)*A)/(n*(n-A))) * f99(A, (n-A))
            T2_lim95 = (((n-1)*(n+1)*A)/(n*(n-A))) * f95(A, (n-A))     
            speX = np.sum(X_**2, axis=1, keepdims=1)
            speX_lim95, speX_lim99 = spe_ci(speX)
            speY = np.sum(Y_**2, axis=1, keepdims=1)
            speY_lim95, speY_lim99 = spe_ci(speY)
            pls_obj['T2'] = T2; pls_obj['T2_lim99'] = T2_lim99; pls_obj['T2_lim95'] = T2_lim95
            pls_obj['speX'] = speX; pls_obj['speX_lim99'] = speX_lim99; pls_obj['speX_lim95'] = speX_lim95
            pls_obj['speY'] = speY; pls_obj['speY_lim99'] = speY_lim99; pls_obj['speY_lim95'] = speY_lim95
            return pls_obj 

# =============================================================================
# Prediction functions (v1.0: use stored var_t, broadcasting)
# =============================================================================

def hott2(mvmobj, *, Xnew=False, Tnew=False):
    var_t = mvmobj.get('var_t', (mvmobj['T'].T @ mvmobj['T']) / mvmobj['T'].shape[0])
    var_t_inv = np.linalg.inv(var_t)
    if isinstance(Xnew, bool) and not isinstance(Tnew, bool):
        hott2_ = np.sum((Tnew @ var_t_inv) * Tnew, axis=1)
    elif isinstance(Tnew, bool) and not isinstance(Xnew, bool):
        if 'Q' in mvmobj:  
            xpred = pls_pred(Xnew, mvmobj)  
        else:
            xpred = pca_pred(Xnew, mvmobj)  
        Tnew = xpred['Tnew']    
        hott2_ = np.sum((Tnew @ var_t_inv) * Tnew, axis=1)
    elif isinstance(Xnew, bool) and isinstance(Tnew, bool):
        Tnew = mvmobj['T']
        hott2_ = np.sum((Tnew @ var_t_inv) * Tnew, axis=1)
    return hott2_

def pca_pred(Xnew, pcaobj, *, algorithm='p2mp'):
    '''Evaluate new data with a PCA model.'''
    if isinstance(Xnew, np.ndarray):
        X_ = Xnew.copy()
        if X_.ndim == 1:
            X_ = X_.reshape(1, -1)
    elif isinstance(Xnew, pd.DataFrame):
        X_ = np.array(Xnew.values[:,1:]).astype(float)

    var_t = pcaobj.get('var_t', (pcaobj['T'].T @ pcaobj['T']) / pcaobj['T'].shape[0])
    
    X_nan_map = np.isnan(X_)    
    if not X_nan_map.any():
        X_mcs = (X_ - pcaobj['mx']) / pcaobj['sx']
        tnew = X_mcs @ pcaobj['P']
        xhat = tnew @ pcaobj['P'].T * pcaobj['sx'] + pcaobj['mx']
        htt2 = np.sum((tnew @ np.linalg.inv(var_t)) * tnew, axis=1)
        spe_ = X_mcs - tnew @ pcaobj['P'].T
        spe_ = np.sum(spe_**2, axis=1, keepdims=True) 
        xpred = {'Xhat':xhat, 'Tnew':tnew, 'speX':spe_, 'T2':htt2}
    elif algorithm == 'p2mp':
        not_Xmiss = (~X_nan_map) * 1
        Xmcs = (X_ - pcaobj['mx']) / pcaobj['sx']
        Xmcs, dummy = n2z(Xmcs)
        for i in range(Xmcs.shape[0]):
            row_map = not_Xmiss[[i], :]
            tempP = pcaobj['P'] * row_map.T
            PTP = tempP.T @ tempP  
            try:
                tnew_, resid, rank, s = np.linalg.lstsq(PTP, tempP.T @ Xmcs[[i], :].T, rcond=None)
            except:
                tnew_ = np.linalg.pinv(PTP) @ tempP.T @ Xmcs[[i], :].T
            if i == 0:
                tnew = tnew_.T
            else:
                tnew = np.vstack((tnew, tnew_.T))
        xhat = tnew @ pcaobj['P'].T * pcaobj['sx'] + pcaobj['mx']
        htt2 = np.sum((tnew @ np.linalg.inv(var_t)) * tnew, axis=1)
        spe_ = (Xmcs - tnew @ pcaobj['P'].T) * not_Xmiss
        spe_ = np.sum(spe_**2, axis=1, keepdims=True) 
        xpred = {'Xhat':xhat, 'Tnew':tnew, 'speX':spe_, 'T2':htt2}
    return xpred

def pls_pred(Xnew, plsobj):
    '''Evaluate new data with a PLS model.'''
    algorithm = 'p2mp'; force_deflation = False
    
    if isinstance(Xnew, np.ndarray):
        X_ = Xnew.copy()
        if X_.ndim == 1:
            X_ = X_.reshape(1, -1)
    elif isinstance(Xnew, pd.DataFrame):
        X_ = np.array(Xnew.values[:,1:]).astype(float)
    elif isinstance(Xnew, dict):
        data_ = []; names_ = []
        for k in Xnew.keys():
            data_.append(Xnew[k]); names_.append(k)
        c = 0
        for i, x in enumerate(data_):        
            x_ = x.values[:,1:].astype(float)                     
            if c == 0:
                X_ = x_.copy() 
            else:    
                X_ = np.hstack((X_, x_))
            c += 1        
       
    var_t = plsobj.get('var_t', (plsobj['T'].T @ plsobj['T']) / plsobj['T'].shape[0])
    
    X_nan_map = np.isnan(X_)    
    if not X_nan_map.any() and not force_deflation:
        X_mcs = (X_ - plsobj['mx']) / plsobj['sx']
        tnew = X_mcs @ plsobj['Ws']
        yhat = tnew @ plsobj['Q'].T * plsobj['sy'] + plsobj['my']
        xhat = tnew @ plsobj['P'].T * plsobj['sx'] + plsobj['mx']
        htt2 = np.sum((tnew @ np.linalg.inv(var_t)) * tnew, axis=1)
        speX = X_mcs - tnew @ plsobj['P'].T
        speX = np.sum(speX**2, axis=1, keepdims=True) 
        ypred = {'Yhat':yhat, 'Xhat':xhat, 'Tnew':tnew, 'speX':speX, 'T2':htt2}
        if 'Wcv' in plsobj:
            ypred['Tcv'] = X_mcs @ plsobj['Wcv']
    elif algorithm == 'p2mp':
        not_Xmiss = (~X_nan_map) * 1
        Xmcs = (X_ - plsobj['mx']) / plsobj['sx']
        Xmcs, dummy = n2z(Xmcs)        
        for i in range(Xmcs.shape[0]):
            row_map = not_Xmiss[[i], :]
            tempW = plsobj['W'] * row_map.T
            for a in range(plsobj['W'].shape[1]):
                WTW = tempW[:,[a]].T @ tempW[:,[a]]
                tnew_aux, resid, rank, s = np.linalg.lstsq(WTW, tempW[:,[a]].T @ Xmcs[[i],:].T, rcond=None)
                Xmcs[[i], :] = (Xmcs[[i], :] - tnew_aux @ plsobj['P'][:,[a]].T) * row_map
                if a == 0:
                    tnew_ = tnew_aux
                else:
                    tnew_ = np.vstack((tnew_, tnew_aux))
            if i == 0:
                tnew = tnew_.T
            else:
                tnew = np.vstack((tnew, tnew_.T))
        yhat = tnew @ plsobj['Q'].T * plsobj['sy'] + plsobj['my']
        xhat = tnew @ plsobj['P'].T * plsobj['sx'] + plsobj['mx']
        htt2 = np.sum((tnew @ np.linalg.inv(var_t)) * tnew, axis=1)
        X_, dummy = n2z(X_)
        speX = ((X_ - plsobj['mx']) / plsobj['sx']) - tnew @ plsobj['P'].T
        speX = speX * not_Xmiss
        speX = np.sum(speX**2, axis=1, keepdims=True) 
        ypred = {'Yhat':yhat, 'Xhat':xhat, 'Tnew':tnew, 'speX':speX, 'T2':htt2} 
        if 'Wcv' in plsobj:
            ypred['Tcv'] = ((X_ - plsobj['mx']) / plsobj['sx']) @ plsobj['Wcv']
    return ypred

def spe(mvmobj, Xnew, *, Ynew=False):
    if 'Q' in mvmobj:  
        xpred = pls_pred(Xnew, mvmobj)  
    else:
        xpred = pca_pred(Xnew, mvmobj)  
    Tnew = xpred['Tnew']
    if isinstance(Xnew, np.ndarray):
        X_ = Xnew.copy()
    elif isinstance(Xnew, pd.DataFrame):
        X_ = np.array(Xnew.values[:,1:]).astype(float)
    if isinstance(Ynew, np.ndarray):
        Y_ = Ynew.copy()
    elif isinstance(Ynew, pd.DataFrame):
        Y_ = np.array(Ynew.values[:,1:]).astype(float)
    
    Xnewhat = Tnew @ mvmobj['P'].T
    Xres = (X_ - mvmobj['mx']) / mvmobj['sx'] - Xnewhat
    spex_ = np.sum(Xres**2, axis=1, keepdims=True)
    
    if not isinstance(Ynew, bool) and ('Q' in mvmobj):
        Ynewhat = Tnew @ mvmobj['Q'].T
        Yres = (Y_ - mvmobj['my']) / mvmobj['sy'] - Ynewhat
        spey_ = np.sum(Yres**2, axis=1, keepdims=True)
        return spex_, spey_
    else:      
        return spex_

# =============================================================================
# LWPLS
# =============================================================================

def lwpls(xnew, loc_par, mvmobj, X, Y, *, shush=False):
    '''LWPLS algorithm per Int. J. Pharmaceutics 421 (2011) 269-274.'''
    if not shush:
        print('phi.lwpls executed on: ' + str(datetime.datetime.now()))
    xnew = np.reshape(xnew, (-1, 1))     
    if isinstance(X, pd.DataFrame):
        X = np.array(X.values[:,1:]).astype(float)
    if isinstance(Y, pd.DataFrame):
        Y = np.array(Y.values[:,1:]).astype(float)
        
    vip = np.sum(np.abs(mvmobj['Ws'] * mvmobj['r2y']), axis=1).reshape(-1, 1)
    theta = vip

    D = X - xnew.T
    d2 = D * theta.T * D
    d2 = np.sqrt(np.sum(d2, axis=1))
    omega = np.exp(-d2 / (np.var(d2, ddof=1) * loc_par))
    OMEGA = np.diag(omega)
    omega = omega.reshape(-1, 1)
    
    X_weighted_mean = (np.sum(omega * X, axis=0) / np.sum(omega)).reshape(-1, 1)
    Y_weighted_mean = (np.sum(omega * Y, axis=0) / np.sum(omega)).reshape(-1, 1)
    
    Xi = X - X_weighted_mean.T
    Yi = Y - Y_weighted_mean.T
    xnewi = xnew - X_weighted_mean
    yhat = Y_weighted_mean
    
    for a in range(mvmobj['T'].shape[1]):
        [U_, S, Wh] = np.linalg.svd(Xi.T @ OMEGA @ Yi @ Yi.T @ OMEGA @ Xi)
        w = Wh.T[:, [0]]
        t = Xi @ w
        p = Xi.T @ OMEGA @ t / (t.T @ OMEGA @ t)        
        q = Yi.T @ OMEGA @ t / (t.T @ OMEGA @ t)
        tnew = xnewi.T @ w
        yhat = yhat + q @ tnew         
        Xi = Xi - t @ p.T
        Yi = Yi - t @ q.T
        xnewi = xnewi - p @ tnew
    return yhat[0].T

# =============================================================================
# Contributions
# =============================================================================

def contributions(mvmobj, X, cont_type, *, Y=False, from_obs=False, to_obs=False, lv_space=False):
    '''Calculate contributions to diagnostics.'''
    if isinstance(lv_space, bool):
        lv_space = list(range(mvmobj['T'].shape[1]))
    elif isinstance(lv_space, int):
        lv_space = (np.array([lv_space]) - 1).tolist()
    elif isinstance(lv_space, list):
        lv_space = (np.array(lv_space) - 1).tolist()
    
    if isinstance(to_obs, int):
        to_obs = [to_obs]
    if not isinstance(from_obs, bool):    
        if isinstance(from_obs, int):
           from_obs = [from_obs]
        
    if isinstance(X, np.ndarray):
        X_ = X.copy()
    elif isinstance(X, pd.DataFrame):
        X_ = np.array(X.values[:,1:]).astype(float)      
    if not isinstance(Y, bool):
        if isinstance(Y, np.ndarray):
            Y_ = Y.copy()
        elif isinstance(Y, pd.DataFrame):
            Y_ = np.array(Y.values[:,1:]).astype(float)
            
    if cont_type == 'ht2' or cont_type == 'scores':
        X_ = (X_ - mvmobj['mx']) / mvmobj['sx']
        X_, dummy = n2z(X_)   
        t_stdevs = np.std(mvmobj['T'], axis=0, ddof=1)     
        loadings = mvmobj['Ws'] if 'Q' in mvmobj else mvmobj['P']
        to_obs_mean = np.mean(X_[to_obs, :], axis=0, keepdims=True)   
        to_cont = np.zeros((1, X_.shape[1]))
        for a in lv_space:    
                aux_ = (to_obs_mean * np.abs(loadings[:, a].T)) / t_stdevs[a]
                to_cont = to_cont + (aux_ if cont_type == 'scores' else aux_**2)
        if not isinstance(from_obs, bool):
            from_obs_mean = np.mean(X_[from_obs, :], axis=0, keepdims=1)    
            from_cont = np.zeros((1, X_.shape[1]))
            for a in lv_space:    
                    aux_ = (from_obs_mean * np.abs(loadings[:, a].T)) / t_stdevs[a]
                    from_cont = from_cont + (aux_ if cont_type == 'scores' else aux_**2)
            return to_cont - from_cont           
        else: 
            return to_cont

    elif cont_type == 'spe':
        X_ = X_[to_obs, :]
        if 'Q' in mvmobj:
            pred = pls_pred(X_, mvmobj)
        else:
            pred = pca_pred(X_, mvmobj)
        Xhat = pred['Xhat']       
        Xhatmcs = (Xhat - mvmobj['mx']) / mvmobj['sx']
        X_ = (X_ - mvmobj['mx']) / mvmobj['sx']
        Xerror = X_ - Xhatmcs
        Xerror, dummy = n2z(Xerror)
        contsX = (Xerror**2) * np.sign(Xerror)
        contsX = np.mean(contsX, axis=0, keepdims=True)
        
        if not isinstance(Y, bool):
            Y_ = Y_[to_obs, :]
            Yhat = pred['Yhat']
            Yhatmcs = (Yhat - mvmobj['my']) / mvmobj['sy']
            Y_ = (Y_ - mvmobj['my']) / mvmobj['sy']
            Yerror = Y_ - Yhatmcs
            Yerror, dummy = n2z(Yerror)
            contsY = (Yerror**2) * np.sign(Yerror)
            contsY = np.mean(contsY, axis=0, keepdims=True)
            return contsX, contsY
        else:
            return contsX

# =============================================================================
# Data cleaning utilities
# =============================================================================

def clean_empty_rows(X, *, shush=False):
    '''Remove rows with all missing data.'''
    if isinstance(X, np.ndarray):
        X_ = X.copy()
        ObsID_ = ['Obs #'+str(n) for n in np.arange(X.shape[0])+1]
    elif isinstance(X, pd.DataFrame):
        X_ = np.array(X.values[:,1:]).astype(float)
        ObsID_ = X.values[:,0].astype(str).tolist()
                     
    X_nan_map = np.isnan(X_)
    Xmiss = np.sum(X_nan_map * 1, axis=1)
    indx = find(Xmiss, lambda x: x == X_.shape[1])
    rows_rem = []   
    if len(indx) > 0:
        for i in indx:
            if not shush:
                print('Removing row ', ObsID_[i], ' due to 100% missing data')
            rows_rem.append(ObsID_[i])
        if isinstance(X, pd.DataFrame):
            X_ = X.drop(X.index.values[indx].tolist())
        else:
            X_ = np.delete(X_, indx, 0)
        return X_, rows_rem
    else:
        return X, rows_rem
    
def clean_low_variances(X, *, shush=False, min_var=1E-10):
    '''Remove columns of negligible variance.'''
    cols_removed = []
    if isinstance(X, pd.DataFrame):
        X_ = np.array(X.values[:,1:]).astype(float)
        varidX = X.columns.values[1:].tolist()
    else:
        X_ = X.copy()
        varidX = ['Var #'+str(n) for n in np.arange(X.shape[1])+1]
            
    X_nan_map = np.isnan(X_)
    Xmiss = np.sum(X_nan_map * 1, axis=0)
    indx = find(Xmiss, lambda x: x >= (X_.shape[0]-3))
    
    if len(indx) > 0:
        for i in indx:
            if not shush:
                print('Removing variable ', varidX[i], ' due to 100% missing data')
        if isinstance(X, pd.DataFrame):
            for i in indx:
                cols_removed.append(varidX[i])
            X_pd = X.drop(X.columns[np.array(indx)+1], axis=1)
            X_ = np.array(X_pd.values[:,1:]).astype(float)
        else:
            for i in indx:
                cols_removed.append(varidX[i])
            X_ = np.delete(X_, indx, 1)
    else:
        X_pd = X.copy()
        
    new_cols = X_pd.columns[1:].tolist() 
    std_x = std(X_).flatten()
    
    indx = find(std_x, lambda x: x < min_var)
    if len(indx) > 0:
        for i in indx:
            if not shush:
                print('Removing variable ', new_cols[i], ' due to low variance')
        if isinstance(X_pd, pd.DataFrame):
            for i in indx:
                cols_removed.append(new_cols[i])
            X_ = X_pd.drop(X_pd.columns[np.array(indx)+1], axis=1)
        else:
            X_ = np.delete(X_, indx, 1)
            for j in indx:
                cols_removed.append(varidX[j])
        return X_, cols_removed    
    else:
        return X_pd, cols_removed

# =============================================================================
# Spectra preprocessing
# =============================================================================

def spectra_snv(x):
    '''Row-wise SNV transform for spectroscopic data.'''
    if isinstance(x, pd.DataFrame):
        x_columns = x.columns
        x_values = x.values
        x_values[:,1:] = spectra_snv(x_values[:,1:].astype(float))
        return pd.DataFrame(x_values, columns=x_columns)
    else:
        if x.ndim == 2:
            mean_x = np.mean(x, axis=1, keepdims=1)
            x = x - mean_x
            std_x = np.sqrt(np.sum(x**2, axis=1) / (x.shape[1]-1)).reshape(-1, 1)
            x = x / std_x
            return x
        else:
            x = x - np.mean(x)
            stdx = np.sqrt(np.sum(x**2) / (len(x)-1))
            return x / stdx
    
def spectra_savgol(ws, od, op, Dm):
    '''Row-wise Savitzky-Golay filter for spectra.'''
    if isinstance(Dm, pd.DataFrame):
        x_columns = Dm.columns.tolist()
        FirstElement = [x_columns[0]]
        x_columns = x_columns[1:]
        FirstElement.extend(x_columns[ws:-ws])
        x_values = Dm.values
        Col1 = Dm.values[:,0].tolist()
        Col1 = np.reshape(Col1, (-1, 1))
        aux, M = spectra_savgol(ws, od, op, x_values[:,1:].astype(float))
        data_ = np.hstack((Col1, aux))
        return pd.DataFrame(data=data_, columns=FirstElement), M
    else:
        l = Dm.shape[0] if Dm.ndim == 1 else Dm.shape[1]
        x_vec = np.arange(-ws, ws+1).reshape(-1, 1)
        X_sg = np.ones((2*ws+1, 1))
        for oo in np.arange(1, op+1):
            X_sg = np.hstack((X_sg, x_vec**oo))
        try:    
            XtXiXt = np.linalg.inv(X_sg.T @ X_sg) @ X_sg.T
        except:
            XtXiXt = np.linalg.pinv(X_sg.T @ X_sg) @ X_sg.T
        coeffs = (XtXiXt[od, :] * factorial(od)).reshape(1, -1)
        for i in np.arange(1, l-2*ws+1):
            if i == 1:
                M = np.hstack((coeffs, np.zeros((1, l-2*ws-1))))
            elif i < l-2*ws:
                m_ = np.hstack((np.zeros((1, i-1)), coeffs, np.zeros((1, l-2*ws-1-i+1))))
                M = np.vstack((M, m_))
            else:
                m_ = np.hstack((np.zeros((1, l-2*ws-1)), coeffs))
                M = np.vstack((M, m_))
        if Dm.ndim == 1: 
            Dm_sg = M @ Dm
        else:
            Dm_sg = Dm @ M.T
        return Dm_sg, M

def spectra_mean_center(Dm):
    '''Mean centering all spectra to have mean zero.'''
    if isinstance(Dm, pd.DataFrame):
        Dm_columns = Dm.columns
        Dm_values = Dm.values
        Dm_values[:,1:] = spectra_mean_center(Dm_values[:,1:].astype(float))
        return pd.DataFrame(Dm_values, columns=Dm_columns)
    else:
        if Dm.ndim == 2:
            return Dm - Dm.mean(axis=1)[:, None]
        else:
            return Dm - Dm.mean()

def spectra_autoscale(Dm):
    '''Autoscaling all spectra to have variance one.'''
    if isinstance(Dm, pd.DataFrame):
        Dm_columns = Dm.columns
        Dm_values = Dm.values
        Dm_values[:,1:] = spectra_autoscale(Dm_values[:,1:].astype(float))
        return pd.DataFrame(Dm_values, columns=Dm_columns)
    else:
        if Dm.ndim == 2:
            return Dm / Dm.std(axis=1, ddof=1)[:, None]
        else: 
            return Dm / Dm.std(ddof=1)

def spectra_baseline_correction(Dm):
    '''Shifting all spectra to have minimum zero.'''
    if isinstance(Dm, pd.DataFrame):
        Dm_columns = Dm.columns
        Dm_values = Dm.values
        Dm_values[:,1:] = spectra_baseline_correction(Dm_values[:,1:].astype(float))
        return pd.DataFrame(Dm_values, columns=Dm_columns)
    else:
        if Dm.ndim == 2: 
            return Dm - Dm.min(axis=1)[:, None]
        else: 
            return Dm - Dm.min()

def spectra_msc(Dm, reference_spectra=None):
    '''Multivariate Scatter Correction transform.'''
    if isinstance(Dm, pd.DataFrame):
        Dm_columns = Dm.columns
        Dm_values = Dm.values
        Dm_values[:,1:] = spectra_msc(Dm_values[:,1:].astype(float))
        return pd.DataFrame(Dm_values, columns=Dm_columns)
    else:
        if Dm.ndim == 2:
            if reference_spectra is None:
                reference_spectra = Dm.mean(axis=0)
            V = np.vstack([np.ones(reference_spectra.shape), reference_spectra])
            U = Dm @ V.T @ np.linalg.inv(V @ V.T)
            return (Dm - U[:,0, None]) / U[:,1, None]
        else:
            if reference_spectra is None:
                raise ValueError("msc needs a reference spectra or to be able to use the mean of many spectra")
            V = np.vstack([np.ones(reference_spectra.shape), reference_spectra])
            U = Dm @ V.T @ np.linalg.inv(V @ V.T)
            return (Dm - U[0]) / U[1]

# =============================================================================
# Bootstrap PLS
# =============================================================================
        
def bootstrap_pls(X, Y, num_latents, num_samples, **kwargs):
    if isinstance(X, pd.DataFrame):
        Dm_values = X.values[:,1:].astype(float)
    if isinstance(Y, pd.DataFrame):
        Y = Y.values[:,1:].astype(float)
    boot_pls_obj = []
    for _ in range(num_samples):
        sample_indexes = np.random.randint(Dm_values.shape[0], None, Dm_values.shape[0])
        boot_pls_obj.append(pls(Dm_values[sample_indexes], Y[sample_indexes], num_latents, shush=True, **kwargs))
    return boot_pls_obj

def bootstrap_pls_pred(X_new, bootstrap_pls_obj, quantiles=[0.025, 0.975]):
    for q in quantiles:
        if q >= 1 or q <= 0:
            raise ValueError("Quantiles must be between zero and one")
    means = []; sds = []
    for pls_obj in bootstrap_pls_obj:
        means.append(pls_pred(X_new, pls_obj)["Yhat"])
        sds.append(np.sqrt(pls_obj["speY"].mean()))
    means = np.array(means).squeeze()
    sds = np.array(sds)
    dist = norm(means, sds[:, None])
    ppf = []
    for q in quantiles:
        def cdf(x):
            return dist.cdf(x).mean(axis=0) - np.ones_like(x)*q
        ppf.append(fsolve(cdf, means.mean(axis=0)))
    return ppf

# =============================================================================
# Pyomo utilities
# =============================================================================

def np2D2pyomo(arr, *, varids=False):
    if not varids:
        return dict(((i+1, j+1), arr[i][j]) for i in range(arr.shape[0]) for j in range(arr.shape[1]))
    else:
        return dict(((varids[i], j+1), arr[i][j]) for i in range(arr.shape[0]) for j in range(arr.shape[1]))

def np1D2pyomo(arr, *, indexes=False):
    if arr.ndim == 2:
        arr = arr[0]
    if isinstance(indexes, bool):
        return dict(((j+1), arr[j]) for j in range(len(arr)))
    elif isinstance(indexes, list):
        return dict((indexes[j], arr[j]) for j in range(len(arr)))

def adapt_pls_4_pyomo(plsobj, *, use_var_ids=False):
    '''Create Pyomo-compatible parameters from PLS object.'''
    plsobj_ = plsobj.copy()
    A = plsobj['T'].shape[1]; N = plsobj['P'].shape[0]; M = plsobj['Q'].shape[0]
    pyo_A = list(np.arange(1, A+1))
    if not use_var_ids:
        pyo_N = list(np.arange(1, N+1)); pyo_M = list(np.arange(1, M+1))
        pyo_Ws = np2D2pyomo(plsobj['Ws']); pyo_Q = np2D2pyomo(plsobj['Q']); pyo_P = np2D2pyomo(plsobj['P'])
        var_t = np.var(plsobj['T'], axis=0)    
        pyo_var_t = np1D2pyomo(var_t)
        pyo_mx = np1D2pyomo(plsobj['mx']); pyo_sx = np1D2pyomo(plsobj['sx'])
        pyo_my = np1D2pyomo(plsobj['my']); pyo_sy = np1D2pyomo(plsobj['sy'])
    else:    
        pyo_N = plsobj['varidX']; pyo_M = plsobj['varidY']
        pyo_Ws = np2D2pyomo(plsobj['Ws'], varids=plsobj['varidX'])
        pyo_Q  = np2D2pyomo(plsobj['Q'], varids=plsobj['varidY'])
        pyo_P  = np2D2pyomo(plsobj['P'], varids=plsobj['varidX'])
        var_t = np.var(plsobj['T'], axis=0)    
        pyo_var_t = np1D2pyomo(var_t)
        pyo_mx = np1D2pyomo(plsobj['mx'], indexes=plsobj['varidX'])
        pyo_sx = np1D2pyomo(plsobj['sx'], indexes=plsobj['varidX'])
        pyo_my = np1D2pyomo(plsobj['my'], indexes=plsobj['varidY'])
        pyo_sy = np1D2pyomo(plsobj['sy'], indexes=plsobj['varidY'])
            
    plsobj_.update({'pyo_A':pyo_A, 'pyo_N':pyo_N, 'pyo_M':pyo_M, 'pyo_Ws':pyo_Ws, 'pyo_Q':pyo_Q,
                    'pyo_P':pyo_P, 'pyo_var_t':pyo_var_t, 'pyo_mx':pyo_mx, 'pyo_sx':pyo_sx,
                    'pyo_my':pyo_my, 'pyo_sy':pyo_sy, 'speX_lim95':plsobj['speX_lim95']})
    return plsobj_

def prep_pca_4_MDbyNLP(pcaobj, X):
    pcaobj_ = pcaobj.copy()
    X_nan_map = np.isnan(X)
    psi = (~X_nan_map) * 1
    X, dummy = n2z(X)
    A = pcaobj['T'].shape[1]; O = pcaobj['T'].shape[0]; N = pcaobj['P'].shape[0]
    pyo_A = list(np.arange(1, A+1)); pyo_N = list(np.arange(1, N+1)); pyo_O = list(np.arange(1, O+1))
    pcaobj_.update({'pyo_A':pyo_A, 'pyo_N':pyo_N, 'pyo_O':pyo_O,
                    'pyo_P_init':np2D2pyomo(pcaobj['P']), 'pyo_T_init':np2D2pyomo(pcaobj['T']),
                    'pyo_X':np2D2pyomo(X), 'pyo_psi':np2D2pyomo(psi)})
    return pcaobj_    

def prep_pls_4_MDbyNLP(plsobj, X, Y):
    plsobj_ = plsobj.copy()
    X_nan_map = np.isnan(X); psi = (~X_nan_map)*1; X, dummy = n2z(X)
    Y_nan_map = np.isnan(Y); theta = (~Y_nan_map)*1; Y, dummy = n2z(Y)
    A = plsobj['T'].shape[1]; O = plsobj['T'].shape[0]
    N = plsobj['P'].shape[0]; M = plsobj['Q'].shape[0]
    pyo_A = list(np.arange(1,A+1)); pyo_N = list(np.arange(1,N+1))
    pyo_O = list(np.arange(1,O+1)); pyo_M = list(np.arange(1,M+1))
    plsobj_.update({'pyo_A':pyo_A, 'pyo_N':pyo_N, 'pyo_O':pyo_O, 'pyo_M':pyo_M,
                    'pyo_P_init':np2D2pyomo(plsobj['P']), 'pyo_Ws_init':np2D2pyomo(plsobj['Ws']),
                    'pyo_T_init':np2D2pyomo(plsobj['T']), 'pyo_Q_init':np2D2pyomo(plsobj['Q']),
                    'pyo_X':np2D2pyomo(X), 'pyo_psi':np2D2pyomo(psi),
                    'pyo_Y':np2D2pyomo(Y), 'pyo_theta':np2D2pyomo(theta)})
    return plsobj_   

def conv_pls_2_eiot(plsobj, *, r_length=False):
    plsobj_ = plsobj.copy()
    A = plsobj['T'].shape[1]; N = plsobj['P'].shape[0]; M = plsobj['Q'].shape[0]
    pyo_A = list(np.arange(1,A+1)); pyo_N = list(np.arange(1,N+1)); pyo_M = list(np.arange(1,M+1))
    pyo_Ws = np2D2pyomo(plsobj['Ws']); pyo_Q = np2D2pyomo(plsobj['Q']); pyo_P = np2D2pyomo(plsobj['P'])
    var_t = np.var(plsobj['T'], axis=0)
    pyo_var_t = np1D2pyomo(var_t)
    pyo_mx = np1D2pyomo(plsobj['mx']); pyo_sx = np1D2pyomo(plsobj['sx'])
    pyo_my = np1D2pyomo(plsobj['my']); pyo_sy = np1D2pyomo(plsobj['sy'])
    
    if not isinstance(r_length, bool):
        if r_length < N:   
            indx_r = list(np.arange(1, r_length+1)); indx_rk_eq = list(np.arange(r_length+1, N+1))
        else:
            if r_length > N:
                print('r_length >> N !!'); print('Forcing r_length=N')
            indx_r = pyo_N; indx_rk_eq = 0
    else:
        indx_r = pyo_N; indx_rk_eq = 0
            
    plsobj_.update({'pyo_A':pyo_A, 'pyo_N':pyo_N, 'pyo_M':pyo_M, 'pyo_Ws':pyo_Ws, 'pyo_Q':pyo_Q,
                    'pyo_P':pyo_P, 'pyo_var_t':pyo_var_t, 'indx_r':indx_r, 'indx_rk_eq':indx_rk_eq,
                    'pyo_mx':pyo_mx, 'pyo_sx':pyo_sx, 'pyo_my':pyo_my, 'pyo_sy':pyo_sy,
                    'S_I':np.nan, 'pyo_S_I':np.nan, 'var_t':var_t})
    return plsobj_    

# =============================================================================
# Categorical, Multi-block PLS, Replicate data, gPROMS export
# =============================================================================

def cat_2_matrix(X):
    '''Convert categorical data into binary matrices for regression.'''
    FirstOne = True; Xmat = []; Xcat = []; XmatMB = []; blknames = []
    for x in X:
        if not FirstOne:
            blknames.append(x)
            categories = np.unique(X[x])
            Xmat_ = []
            for c in categories:
                Xcat.append(c)
                xmat_ = (X[x] == c) * 1
                Xmat.append(xmat_)
                Xmat_.append(xmat_)
            Xmat_ = pd.DataFrame(np.array(Xmat_).T, columns=categories) 
            Xmat_.insert(0, firstcol, X[firstcol]) 
            XmatMB.append(Xmat_)            
        else:
            firstcol = x; FirstOne = False
    Xmat = pd.DataFrame(np.array(Xmat).T, columns=Xcat) 
    Xmat.insert(0, firstcol, X[firstcol].values)      
    aux_dict = dict()
    for bname, bdata in zip(blknames, XmatMB):
        aux_dict[bname] = bdata
    return Xmat, aux_dict

def mbpls(XMB, YMB, A, *, mcsX=True, mcsY=True, md_algorithm_='nipals',
          force_nipals_=False, shush_=False, cross_val_=0, cross_val_X_=False, cca=False):
    '''Multi-Block PLS per Westerhuis, J. Chemometrics, 12, 301-321 (1998).'''
    x_means=[]; x_stds=[]; y_means=[]; y_stds=[]
    Xblk_scales=[]; Yblk_scales=[]; Xcols_per_block=[]; Ycols_per_block=[]
    X_var_names=[]; Y_var_names=[]; obsids=[]
    
    if isinstance(XMB, dict):
        data_ = []; names_ = []
        for k in XMB.keys():
            data_.append(XMB[k]); names_.append(k)
        XMB = {'data':data_, 'blknames':names_}
        x = XMB['data'][0]        
        columns = x.columns.tolist()
        obsid_column_name = columns[0]        
        obsids = x[obsid_column_name].tolist()
        c = 0
        for x in XMB['data']:        
            x_ = x.values[:,1:].astype(float)         
            columns = x.columns.tolist()
            for i, h in enumerate(columns):
                if i != 0:
                    X_var_names.append(XMB['blknames'][c]+' '+h)
            if isinstance(mcsX, bool):
                if mcsX:
                    x_, x_mean_, x_std_ = meancenterscale(x_)
                else:    
                    x_mean_ = np.zeros((1, x_.shape[1])); x_std_ = np.ones((1, x_.shape[1]))
            elif mcsX[c] == 'center':
                x_, x_mean_, x_std_ = meancenterscale(x_, mcs='center')
            elif mcsX[c] == 'autoscale':
                x_, x_mean_, x_std_ = meancenterscale(x_, mcs='autoscale')
            blck_scale = np.sqrt(np.sum(std(x_)**2))
            x_means.append(x_mean_); x_stds.append(x_std_)                   
            Xblk_scales.append(blck_scale); Xcols_per_block.append(x_.shape[1])
            x_ = x_ / blck_scale
            if c == 0: X_ = x_.copy() 
            else:      X_ = np.hstack((X_, x_))
            c += 1
    elif isinstance(XMB, pd.DataFrame):
        columns = XMB.columns.tolist()
        obsid_column_name = columns[0]; obsids = XMB[obsid_column_name].tolist()            
        for i, h in enumerate(columns):
            if i != 0: X_var_names.append(h)
        x_ = XMB.values[:,1:].astype(float)
        if isinstance(mcsX, bool):
            if mcsX: x_, x_mean_, x_std_ = meancenterscale(x_)
            else:    x_mean_ = np.zeros((1, x_.shape[1])); x_std_ = np.ones((1, x_.shape[1]))
        elif mcsX[0] == 'center':
            x_, x_mean_, x_std_ = meancenterscale(x_, mcs='center')
        elif mcsX[0] == 'autoscale':
            x_, x_mean_, x_std_ = meancenterscale(x_, mcs='autoscale')
        blck_scale = np.sqrt(np.sum(std(x_)**2))
        x_means.append(x_mean_); x_stds.append(x_std_)                   
        Xblk_scales.append(blck_scale); Xcols_per_block.append(x_.shape[1])
        X_ = x_ / blck_scale
        
    if isinstance(YMB, dict):
        data_ = []; names_ = []
        for k in YMB.keys():
            data_.append(YMB[k]); names_.append(k)
        YMB = {'data':data_, 'blknames':names_}
        c = 0
        for y in YMB['data']:        
            y_ = y.values[:,1:].astype(float)
            columns = y.columns.tolist()
            for i, h in enumerate(columns):
                if i != 0: Y_var_names.append(h)
            if isinstance(mcsY, bool):
                if mcsY: y_, y_mean_, y_std_ = meancenterscale(y_)
                else:    y_mean_ = np.zeros((1, y_.shape[1])); y_std_ = np.ones((1, y_.shape[1]))
            elif mcsY[c] == 'center':
                y_, y_mean_, y_std_ = meancenterscale(y_, mcs='center')
            elif mcsY[c] == 'autoscale':
                y_, y_mean_, y_std_ = meancenterscale(y_, mcs='autoscale')
            blck_scale = np.sqrt(np.sum(std(y_)**2))
            y_means.append(y_mean_); y_stds.append(y_std_)                   
            Yblk_scales.append(blck_scale); Ycols_per_block.append(y_.shape[1])
            y_ = y_ / blck_scale
            if c == 0: Y_ = y_.copy()
            else:      Y_ = np.hstack((Y_, y_))
            c += 1    
    elif isinstance(YMB, pd.DataFrame):
        y_ = YMB.values[:,1:].astype(float)
        columns = YMB.columns.tolist()
        for i, h in enumerate(columns):
            if i != 0: Y_var_names.append(h)
        if isinstance(mcsY, bool):
            if mcsY: y_, y_mean_, y_std_ = meancenterscale(y_)
            else:    y_mean_ = np.zeros((1, y_.shape[1])); y_std_ = np.ones((1, y_.shape[1]))
        elif mcsY[0] == 'center':
            y_, y_mean_, y_std_ = meancenterscale(y_, mcs='center')
        elif mcsY[0] == 'autoscale':
            y_, y_mean_, y_std_ = meancenterscale(y_, mcs='autoscale')
        blck_scale = np.sqrt(np.sum(std(y_)**2))
        y_means.append(y_mean_); y_stds.append(y_std_)                   
        Yblk_scales.append(blck_scale); Ycols_per_block.append(y_.shape[1])
        Y_ = y_ / blck_scale
        
    X_pd = pd.DataFrame(X_, columns=X_var_names)
    X_pd.insert(0, obsid_column_name, obsids)
    Y_pd = pd.DataFrame(Y_, columns=Y_var_names)
    Y_pd.insert(0, obsid_column_name, obsids)

    pls_obj_ = pls(X_pd, Y_pd, A, mcsX=False, mcsY=False, md_algorithm=md_algorithm_,
                 force_nipals=force_nipals_, shush=shush_, cross_val=cross_val_, cross_val_X=cross_val_X_, cca=cca)          
    pls_obj_['type'] = 'mbpls'
    
    Wsb=[]; Wb=[]; Tb=[]
    for i, c in enumerate(Xcols_per_block):
        start_index = 0 if i == 0 else np.sum(Xcols_per_block[0:i])
        end_index = start_index + c
        wsb_ = pls_obj_['Ws'][start_index:end_index, :].copy()
        for j in range(wsb_.shape[1]):
            wsb_[:, j] = wsb_[:, j] / np.linalg.norm(wsb_[:, j])
        Wsb.append(wsb_)
        wb_ = pls_obj_['W'][start_index:end_index, :].copy()
        for j in range(wb_.shape[1]):
            wb_[:, j] = wb_[:, j] / np.linalg.norm(wb_[:, j])
        Wb.append(wb_)
        
        Xb = X_[:, start_index:end_index]
        X_nan_map = np.isnan(Xb)
        not_Xmiss = (~X_nan_map) * 1
        Xb, dummy = n2z(Xb)
        TSS = np.sum(Xb**2)
        
        for a in range(A):            
            w_ = wb_[:, [a]]
            w_t = w_.T * not_Xmiss
            w_t = np.sum(w_t**2, axis=1, keepdims=True)           
            tb_ = (Xb @ w_) / w_t
            if a == 0: tb = tb_
            else:      tb = np.hstack((tb, tb_))
            tb_t = tb_.T * not_Xmiss.T
            tb_t = np.sum(tb_t**2, axis=1, keepdims=True)
            try:    pb_ = (Xb.T @ tb_) / tb_t
            except: pb_ = 0
            Xb = (Xb - tb_ @ pb_.T) * not_Xmiss
            r2pb_aux = 1 - (np.sum(Xb**2)/TSS)
            if a == 0: r2pb_ = r2pb_aux
            else:      r2pb_ = np.hstack((r2pb_, r2pb_aux))
        if i == 0: r2pbX = r2pb_
        else:      r2pbX = np.vstack((r2pbX, r2pb_))
        Tb.append(tb)
        
    for a in range(A):
        u = pls_obj_['U'][:, [a]].copy()
        for i, c in enumerate(Xcols_per_block):
            if i == 0: T_a = Tb[i][:, [a]]
            else:      T_a = np.hstack((T_a, Tb[i][:, [a]]))
        wt_ = (T_a.T @ u) / (u.T @ u)
        if a == 0: Wt = wt_
        else:      Wt = np.hstack((Wt, wt_))
    
    pls_obj_['x_means'] = x_means; pls_obj_['x_stds'] = x_stds
    pls_obj_['y_means'] = y_means; pls_obj_['y_stds'] = y_stds
    pls_obj_['Xblk_scales'] = Xblk_scales; pls_obj_['Yblk_scales'] = Yblk_scales
    pls_obj_['Wsb'] = Wsb; pls_obj_['Wt'] = Wt
    
    mx_ = [j for l in x_means for j in l[0]]
    pls_obj_['mx'] = np.array(mx_)    
    sx_ = [j * Xblk_scales[i] for i, l in enumerate(x_stds) for j in l[0]]
    pls_obj_['sx'] = np.array(sx_)
    my_ = [j for l in y_means for j in l[0]]
    pls_obj_['my'] = np.array(my_)    
    sy_ = [j * Yblk_scales[i] for i, l in enumerate(y_stds) for j in l[0]]
    pls_obj_['sy'] = np.array(sy_)
    
    if isinstance(XMB, dict):
        for a in range(A-1, 0, -1):
             r2pbX[:, [a]] = r2pbX[:, [a]] - r2pbX[:, [a-1]]
        r2pbXc = np.cumsum(r2pbX, axis=1)
    else:
        for a in range(A-1, 0, -1):
             r2pbX[a] = r2pbX[a] - r2pbX[a-1]
        r2pbXc = np.cumsum(r2pbX)

    pls_obj_['r2pbX'] = r2pbX; pls_obj_['r2pbXc'] = r2pbXc
    if isinstance(XMB, dict): pls_obj_['Xblocknames'] = XMB['blknames']
    if isinstance(YMB, dict): pls_obj_['Yblocknames'] = YMB['blknames']
    return pls_obj_

def replicate_data(mvm_obj, X, num_replicates, *, as_set=False, rep_Y=False, Y=False):
    if not rep_Y:
        if 'Q' in mvm_obj:
            preds = pls_pred(X, mvm_obj); xhat = preds['Xhat']; tnew = preds['Tnew']
        else:
            preds = pca_pred(X, mvm_obj); xhat = preds['Xhat']; tnew = preds['Tnew']
        xhat = tnew @ mvm_obj['P'].T
        xmcs = (X.values[:,1:] - mvm_obj['mx']) / mvm_obj['sx']
        data_residuals = xmcs - xhat
        if not as_set:
            if np.mod(num_replicates, X.shape[0]) == 0:
                new_set = np.tile(xhat, (int(num_replicates/X.shape[0]), 1))
            else:
                reps = int(np.floor(num_replicates/X.shape[0]))
                new_set = np.tile(xhat, (reps, 1))
                new_set = np.vstack((new_set, xhat[:np.mod(num_replicates, X.shape[0]), :]))                
            obsids = ['clone'+str(i) for i in np.arange(new_set.shape[0])+1]
        else:            
            new_set = np.tile(xhat, (num_replicates, 1))
            obsids = []
            obsid_o = X[X.columns[0]].values.astype(str).tolist()
            for i in np.arange(num_replicates)+1:
                obsids.extend([m+'_clone'+str(i) for m in obsid_o])
                
        uncertainty_matrix = np.empty((new_set.shape[0], 0))
        for i in range(data_residuals.shape[1]):
            ecdf = ECDF(data_residuals[:, i])
            new_residual = np.random.uniform(0, 1, new_set.shape[0])
            y = np.array(ecdf.y.tolist()); x = np.array(ecdf.x.tolist())
            new_residual = np.interp(new_residual, y[1:], x[1:])
            uncertainty_matrix = np.hstack((uncertainty_matrix, new_residual.reshape(-1, 1)))
        new_set = (new_set + uncertainty_matrix) * mvm_obj['sx'] + mvm_obj['mx']
        new_set_pd = pd.DataFrame(new_set, columns=X.columns[1:].tolist())
        new_set_pd.insert(0, X.columns[0], obsids)
        return new_set_pd
    elif 'Q' in mvm_obj:
        preds = pls_pred(X, mvm_obj); yhat = preds['Yhat']; tnew = preds['Tnew']
        yhat = tnew @ mvm_obj['Q'].T
        ymcs = (Y.values[:,1:] - mvm_obj['my']) / mvm_obj['sy']
        data_residuals = ymcs - yhat
        if not as_set:
            if np.mod(num_replicates, Y.shape[0]) == 0:
                new_set = np.tile(yhat, (int(num_replicates/Y.shape[0]), 1))
            else:
                reps = int(np.floor(num_replicates/Y.shape[0]))
                new_set = np.tile(yhat, (reps, 1))
                new_set = np.vstack((new_set, yhat[:np.mod(num_replicates, Y.shape[0]), :]))                
            obsids = ['clone'+str(i) for i in np.arange(new_set.shape[0])+1]
        else:            
            new_set = np.tile(yhat, (num_replicates, 1))
            obsids = []
            obsid_o = Y[Y.columns[0]].values.astype(str).tolist()
            for i in np.arange(num_replicates)+1:
                obsids.extend([m+'_clone'+str(i) for m in obsid_o])
                
        uncertainty_matrix = np.empty((new_set.shape[0], 0))
        for i in range(data_residuals.shape[1]):
            ecdf = ECDF(data_residuals[:, i])
            new_residual = np.random.uniform(0, 1, new_set.shape[0])
            y = np.array(ecdf.y.tolist()); x = np.array(ecdf.x.tolist())
            new_residual = np.interp(new_residual, y[1:], x[1:])
            uncertainty_matrix = np.hstack((uncertainty_matrix, new_residual.reshape(-1, 1)))
        new_set = (new_set + uncertainty_matrix) * mvm_obj['sy'] + mvm_obj['my']
        new_set_pd = pd.DataFrame(new_set, columns=Y.columns[1:].tolist())
        new_set_pd.insert(0, Y.columns[0], obsids)
        return new_set_pd
    else:
        print('You need a PLS model to replicate Y')

def export_2_gproms(mvmobj, *, fname='phi_export.txt'):
    '''Export PLS model to gPROMS syntax.'''
    top_lines = [     
    'PARAMETER', 'X_VARS AS ORDERED_SET', 'Y_VARS AS ORDERED_SET', 'A      AS INTEGER',
    'VARIABLE',
    'X_MEANS as ARRAY(X_VARS)    OF no_type', 'X_STD   AS ARRAY(X_VARS)    OF no_type',
    'Y_MEANS as ARRAY(Y_VARS)    OF no_type', 'Y_STD   AS ARRAY(Y_VARS)    OF no_type',
    'Ws      AS ARRAY(X_VARS,A)  OF no_type', 'Q       AS ARRAY(Y_VARS,A)  OF no_type',
    'P       AS ARRAY(X_VARS,A)  OF no_type', 'T       AS ARRAY(A)         OF no_type',
    'Tvar    AS ARRAY(A)         OF no_type',
    'X_HAT   AS ARRAY(X_VARS)    OF no_type # Mean-centered and scaled',
    'Y_HAT   AS ARRAY(Y_VARS)    OF no_type # Mean-centered and scaled',
    'X_PRED  AS ARRAY(X_VARS)    OF no_type # In original units',
    'Y_PRED  AS ARRAY(Y_VARS)    OF no_type # In original units',
    'X_NEW   AS ARRAY(X_VARS)    OF no_type # In original units',
    'X_MCS   AS ARRAY(X_VARS)    OF no_type # Mean-centered and scaled',
    'HT2                         AS no_type', 'SPEX                        AS no_type', 'SET']
    
    x_var_line = "X_VARS:=['" + "','".join(mvmobj['varidX']) + "'];"
    y_var_line = "Y_VARS:=['" + "','".join(mvmobj['varidY']) + "'];"
    top_lines.extend([x_var_line, y_var_line, 'A:='+str(mvmobj['T'].shape[1])+';'])
    
    mid_lines = [
    'EQUATION', 'X_MCS * X_STD = (X_NEW-X_MEANS);',
    'FOR j:=1 TO A DO', 'T(j) = SIGMA(X_MCS*Ws(,j));', 'END',
    'FOR i IN Y_VARS DO', 'Y_HAT(i) = SIGMA(T*Q(i,));', 'END',
    'FOR i IN X_VARS DO', 'X_HAT(i) = SIGMA(T*P(i,));', 'END',
    '(X_HAT * X_STD) + X_MEANS = X_PRED;', '(Y_HAT * Y_STD) + Y_MEANS = Y_PRED;',
    'HT2  = SIGMA ((T^2)/Tvar);', 'SPEX = SIGMA ((X_MCS - X_HAT)^2);']

    assign_lines = ['ASSIGN']
    for i, xvar in enumerate(mvmobj['varidX']):     
        assign_lines.append("X_MEANS('"+xvar+"') := "+str(mvmobj['mx'][0,i])+";" )
    for i, xvar in enumerate(mvmobj['varidX']):     
        assign_lines.append("X_STD('"+xvar+"') := "+str(mvmobj['sx'][0,i])+";" )
    for i, yvar in enumerate(mvmobj['varidY']):     
        assign_lines.append("Y_MEANS('"+yvar+"') := "+str(mvmobj['my'][0,i])+";" )
    for i, yvar in enumerate(mvmobj['varidY']):     
        assign_lines.append("Y_STD('"+yvar+"') := "+str(mvmobj['sy'][0,i])+";" )
    for i, xvar in enumerate(mvmobj['varidX']):     
        for j in np.arange(mvmobj['Ws'].shape[1]):
            assign_lines.append("Ws('"+xvar+"',"+str(j+1)+") := "+str(mvmobj['Ws'][i,j])+";")
    for i, xvar in enumerate(mvmobj['varidX']):     
        for j in np.arange(mvmobj['P'].shape[1]):
            assign_lines.append("P('"+xvar+"',"+str(j+1)+") := "+str(mvmobj['P'][i,j])+";")
    for i, yvar in enumerate(mvmobj['varidY']):     
        for j in np.arange(mvmobj['Q'].shape[1]):
            assign_lines.append("Q('"+yvar+"',"+str(j+1)+") := "+str(mvmobj['Q'][i,j])+";")
    tvar = np.std(mvmobj['T'], axis=0, ddof=1)
    for j in np.arange(mvmobj['T'].shape[1]):
        assign_lines.append("Tvar("+str(j+1)+") := "+str(tvar[j])+";" ) 

    lines = top_lines + mid_lines + assign_lines
    with open(fname, "w") as outfile:
        outfile.write("\n".join(lines))
    return

# =============================================================================
# Parsing utilities
# =============================================================================

def unique(df, colid):    
    '''Returns unique values in df column in order of occurrence.'''
    return df.drop_duplicates(subset=colid, keep='first')[colid].values.tolist()
    
def parse_materials(filename, sheetname):
    '''Build R matrices for JRPLS from linear table in Excel.'''
    materials = pd.read_excel(filename, sheet_name=sheetname)
    ok = True
    for lot in unique(materials, 'Finished Product Lot'):
        this_lot = materials[materials["Finished Product Lot"]==lot]
        for mt, m in zip(this_lot['Material'].values, this_lot['Material Lot'].values):
            try:
               if np.isnan(m):
                   print('Lot '+lot+' has no Material Lot for '+mt); ok = False; break
            except:
                pass
        if not ok: break
        print('Lot :'+lot+' ratio/qty adds to '+str(np.sum(this_lot['Ratio or Quantity'].values)))
    if ok:    
        JR = []
        materials_used = unique(materials, 'Material')
        fp_lots = unique(materials, 'Finished Product Lot')
        for m in materials_used:
            mat_lots = np.unique(materials['Material Lot'][materials['Material']==m]).tolist()
            r_mat = []
            for lot in fp_lots:
                rvec = np.zeros(len(mat_lots))
                this_lot_this_mat = materials[(materials["Finished Product Lot"]==lot) & (materials['Material']==m)]
                for l, r in zip(this_lot_this_mat['Material Lot'].values, this_lot_this_mat['Ratio or Quantity'].values):
                    rvec[mat_lots.index(l)] = r
                r_mat.append(rvec)    
            r_mat_pd = pd.DataFrame(np.array(r_mat), columns=mat_lots)
            r_mat_pd.insert(0, 'FPLot', fp_lots)    
            JR.append(r_mat_pd)
        return JR, materials_used
    else:
        print('Data needs revision'); return False, False
    
def isin_ordered_col0(df, alist):
    df_ = df[df[df.columns[0]].isin(alist)].set_index(df.columns[0]).reindex(alist).reset_index()
    return df_
    
def reconcile_rows(df_list):
    '''Reconcile observations across multiple dataframes.'''
    all_rows = []
    for df in df_list:
        all_rows.extend(df[df.columns[0]].values.tolist())
    all_rows = np.unique(all_rows)    
    for df in df_list:
        rows = df[df.columns[0]].values.tolist()
        all_rows = [r for r in all_rows if r in rows]
    return [isin_ordered_col0(df, all_rows) for df in df_list]
    
def reconcile_rows_to_columns(df_list_r, df_list_c): 
    '''Reconcile rows of df_list_r with columns of df_list_c.'''
    df_list_r_o = []; df_list_c_o = []
    for dfr, dfc in zip(df_list_r, df_list_c):
        all_ids = list(set(dfc.columns[1:].tolist()) & set(dfr[dfr.columns[0]].values.tolist()))
        # Preserve order from dfr
        rows = dfr[dfr.columns[0]].values.tolist()
        cols = dfc.columns[1:].tolist()
        all_ids = [i for i in rows if i in cols]
        dfr_ = isin_ordered_col0(dfr, all_ids)
        dfc_ = dfc[all_ids].copy()
        dfc_.insert(0, dfc.columns[0], dfc[dfc.columns[0]].values.tolist())
        df_list_r_o.append(dfr_); df_list_c_o.append(dfc_)
    return df_list_r_o, df_list_c_o

# =============================================================================
# LPLS, JRPLS, TPLS (use optimized _Ab_btbinv)
# =============================================================================

def lpls(X, R, Y, A, *, shush=False):
    '''LPLS Algorithm per Muteki et al. Chemom.Intell.Lab.Syst.85(2007) 186-194.'''
    # Validate R-Y (blend observations must match row-to-row)
    R, Y = _validate_inputs(R, Y, A=A)
    
    X_, obsidX, varidX = _extract_array(X)
    Y_, obsidY, varidY = _extract_array(Y)
    R_, obsidR, varidR = _extract_array(R)
        
    X_, x_mean, x_std = meancenterscale(X_)
    Y_, y_mean, y_std = meancenterscale(Y_)
    R_, r_mean, r_std = meancenterscale(R_)
    
    not_Xmiss = (~np.isnan(X_)) * 1
    not_Ymiss = (~np.isnan(Y_)) * 1
    not_Rmiss = (~np.isnan(R_)) * 1

    if not shush:
        print('phi.lpls using NIPALS executed on: ' + str(datetime.datetime.now()))
    X_, dummy = n2z(X_); Xhat = np.zeros(X_.shape)
    Y_, dummy = n2z(Y_); R_, dummy = n2z(R_)
    epsilon = 1E-9; maxit = 2000

    TSSX = np.sum(X_**2); TSSXpv = np.sum(X_**2, axis=0)
    TSSY = np.sum(Y_**2); TSSYpv = np.sum(Y_**2, axis=0)
    TSSR = np.sum(R_**2); TSSRpv = np.sum(R_**2, axis=0)
    
    r2X = None; r2Xpv = None; r2Y = None; r2Ypv = None; r2R = None; r2Rpv = None
    for a in range(A):
        ui = Y_[:, [np.argmax(std(Y_))]]
        Converged = False; num_it = 0
        while not Converged:
             hi = _Ab_btbinv(R_.T, ui, not_Rmiss.T)
             si = _Ab_btbinv(X_.T, hi, not_Xmiss.T)
             si = si / np.linalg.norm(si)
             ri = _Ab_btbinv(X_, si, not_Xmiss)
             ti = _Ab_btbinv(R_, ri, not_Rmiss)
             qi = _Ab_btbinv(Y_.T, ti, not_Ymiss.T)
             un = _Ab_btbinv(Y_, qi, not_Ymiss)
             
             if abs((np.linalg.norm(ui) - np.linalg.norm(un))) / np.linalg.norm(ui) < epsilon:
                 Converged = True
             if num_it > maxit:
                 Converged = True
             if Converged:
                 if not shush: print('# Iterations for LV #'+str(a+1)+': ', str(num_it))
                 pi = _Ab_btbinv(R_.T, ti, not_Rmiss.T)
                 vi = _Ab_btbinv(X_.T, ri, not_Xmiss.T)
                 R_ = (R_ - ti @ pi.T) * not_Rmiss
                 X_ = (X_ - ri @ vi.T) * not_Xmiss
                 Y_ = (Y_ - ti @ qi.T) * not_Ymiss
                 Xhat = Xhat + ri @ vi.T
                 if a == 0:
                     T=ti; P=pi; S=si; U=un; Q=qi; H=hi; V=vi; Rscores=ri
                 else:
                     T=np.hstack((T,ti)); U=np.hstack((U,un)); P=np.hstack((P,pi))
                     Q=np.hstack((Q,qi)); S=np.hstack((S,si)); H=np.hstack((H,hi))
                     V=np.hstack((V,vi)); Rscores=np.hstack((Rscores,ri))
                 r2X, r2Xpv = _calc_r2(X_, TSSX, TSSXpv, r2X, r2Xpv)
                 r2Y, r2Ypv = _calc_r2(Y_, TSSY, TSSYpv, r2Y, r2Ypv)
                 r2R, r2Rpv = _calc_r2(R_, TSSR, TSSRpv, r2R, r2Rpv)
             else:
                 num_it += 1; ui = un
        if a == 0: numIT = num_it
        else:      numIT = np.hstack((numIT, num_it))
    
    Xhat = Xhat * x_std + x_mean
    r2X, r2Xpv = _r2_cumulative_to_per_component(r2X, r2Xpv, A)
    r2Y, r2Ypv = _r2_cumulative_to_per_component(r2Y, r2Ypv, A)
    r2R, r2Rpv = _r2_cumulative_to_per_component(r2R, r2Rpv, A)
    
    eigs = np.var(T, axis=0)
    r2xc = np.cumsum(r2X); r2yc = np.cumsum(r2Y); r2rc = np.cumsum(r2R)
    if not shush:
        print('--------------------------------------------------------------')
        print('LV #     Eig       R2X       sum(R2X)   R2R       sum(R2R)   R2Y       sum(R2Y)')
        if A > 1:    
            for a in range(A):
                print("LV #"+str(a+1)+":   {:6.3f}    {:.3f}     {:.3f}      {:.3f}     {:.3f}      {:.3f}     {:.3f}".format(eigs[a], r2X[a], r2xc[a], r2R[a], r2rc[a], r2Y[a], r2yc[a]))
        else:
           print("LV #1:   {:6.3f}    {:.3f}     {:.3f}      {:.3f}     {:.3f}      {:.3f}     {:.3f}".format(eigs[0], r2X, r2xc[0], r2R, r2rc[0], r2Y, r2yc[0]))
        print('--------------------------------------------------------------')   
        
    var_t = (T.T @ T) / T.shape[0]
    lpls_obj = {'T':T, 'P':P, 'Q':Q, 'U':U, 'S':S, 'H':H, 'V':V, 'Rscores':Rscores,
              'r2x':r2X, 'r2xpv':r2Xpv, 'mx':x_mean, 'sx':x_std,
              'r2y':r2Y, 'r2ypv':r2Ypv, 'my':y_mean, 'sy':y_std,
              'r2r':r2R, 'r2rpv':r2Rpv, 'mr':r_mean, 'sr':r_std,
              'Xhat':Xhat, 'var_t':var_t}  
    if not isinstance(obsidX, bool): lpls_obj['obsidX'] = obsidX; lpls_obj['varidX'] = varidX
    if not isinstance(obsidY, bool): lpls_obj['obsidY'] = obsidY; lpls_obj['varidY'] = varidY
    if not isinstance(obsidR, bool): lpls_obj['obsidR'] = obsidR; lpls_obj['varidR'] = varidR
          
    T2 = hott2(lpls_obj, Tnew=T)
    n = T.shape[0]
    T2_lim99 = (((n-1)*(n+1)*A)/(n*(n-A))) * f99(A, (n-A))
    T2_lim95 = (((n-1)*(n+1)*A)/(n*(n-A))) * f95(A, (n-A))     
    speX = np.sum(X_**2, axis=1, keepdims=1); speX_lim95, speX_lim99 = spe_ci(speX)
    speY = np.sum(Y_**2, axis=1, keepdims=1); speY_lim95, speY_lim99 = spe_ci(speY)
    speR = np.sum(R_**2, axis=1, keepdims=1); speR_lim95, speR_lim99 = spe_ci(speR)
    
    lpls_obj.update({'T2':T2, 'T2_lim99':T2_lim99, 'T2_lim95':T2_lim95,
        'speX':speX, 'speX_lim99':speX_lim99, 'speX_lim95':speX_lim95,
        'speY':speY, 'speY_lim99':speY_lim99, 'speY_lim95':speY_lim95,
        'speR':speR, 'speR_lim99':speR_lim99, 'speR_lim95':speR_lim95})
    
    lpls_obj['Ss'] = S @ np.linalg.pinv(V.T @ S)
    lpls_obj['type'] = 'lpls'
    return lpls_obj       
    
def lpls_pred(rnew, lpls_obj):
    '''Prediction with an LPLS model.'''
    if isinstance(rnew, np.ndarray): rnew__ = [rnew.copy()]
    elif isinstance(rnew, list):     rnew__ = np.array(rnew)
    elif isinstance(rnew, pd.DataFrame): rnew__ = rnew.values[:,1:].astype(float)
    tnew = []; sper = []
    for rnew_ in rnew__:
        rnew_ = ((rnew_ - lpls_obj['mr']) / lpls_obj['sr']).reshape(-1, 1)
        ti = []
        for a in np.arange(lpls_obj['T'].shape[1]):
            ti_ = (rnew_.T @ lpls_obj['Rscores'][:,a] / (lpls_obj['Rscores'][:,a].T @ lpls_obj['Rscores'][:,a]))
            ti.append(ti_[0])
            rnew_ = rnew_ - (ti_ * lpls_obj['P'][:,a]).reshape(-1, 1)
        tnew.append(np.array(ti))
        sper.append(np.sum(rnew_**2))
    tnew = np.array(tnew); sper = np.array(sper)
    yhat = (tnew @ lpls_obj['Q'].T) * lpls_obj['sy'] + lpls_obj['my']
    return {'Tnew':tnew, 'Yhat':yhat, 'speR':sper}

def jrpls(Xi, Ri, Y, A, *, shush=False):
    '''JRPLS Algorithm per Garcia-Munoz Chemom.Intel.Lab.Syst., 133, pp.49-62.

    jrpls_obj = pyphi.jrpls(Xi, Ri, Y, A)
    Args:
         Xi: Phys. Prop. dictionary of DataFrames {material: df}
         Ri: Blending ratios dictionary of DataFrames {material: df}
         Y:  Product characteristics DataFrame [b x n]
         A:  Number of components
    Returns:
        jrpls_obj: Dictionary with JRPLS model parameters
    '''
    X = []; varidX = []; obsidX = []
    materials = list(Xi.keys())
    for k in Xi.keys():
        X_, obsidX_, varidX_ = _extract_array(Xi[k])
        X.append(X_); varidX.append(varidX_); obsidX.append(obsidX_)
        
    Y_, obsidY, varidY = _extract_array(Y)

    R = []; varidR = []; obsidR = []
    for k in materials:  
        R_, obsidR_, varidR_ = _extract_array(Ri[k])
        varidR.append(varidR_); obsidR.append(obsidR_); R.append(R_)
        
    x_mean = []; x_std = []; jr_scale = []; r_mean = []; r_std = []
    not_Xmiss = []; not_Rmiss = []; Xhat = []
    TSSX = []; TSSXpv = []; TSSR = []; TSSRpv = []
    X__ = []; R__ = []
    for X_i, R_i in zip(X, R):
        X_i, x_mean_, x_std_ = meancenterscale(X_i)
        R_i, r_mean_, r_std_ = meancenterscale(R_i)
        jr_scale_ = np.sqrt(X_i.shape[1])
        X_i = X_i / jr_scale_
        x_mean.append(x_mean_); x_std.append(x_std_); jr_scale.append(jr_scale_)
        r_mean.append(r_mean_); r_std.append(r_std_)
        not_Xmiss.append((~np.isnan(X_i)) * 1)
        not_Rmiss.append((~np.isnan(R_i)) * 1)
        X_i, dummy = n2z(X_i); R_i, dummy = n2z(R_i)
        X__.append(X_i); R__.append(R_i)
        Xhat.append(np.zeros(X_i.shape))
        TSSX.append(np.sum(X_i**2)); TSSXpv.append(np.sum(X_i**2, axis=0))
        TSSR.append(np.sum(R_i**2)); TSSRpv.append(np.sum(R_i**2, axis=0))
    X = X__; R = R__
    
    Y_, y_mean, y_std = meancenterscale(Y_)
    not_Ymiss = (~np.isnan(Y_)) * 1
    Y_, dummy = n2z(Y_)
    TSSY = np.sum(Y_**2); TSSYpv = np.sum(Y_**2, axis=0)
    
    if not shush:
        print('phi.jrpls using NIPALS executed on: ' + str(datetime.datetime.now()))
    epsilon = 1E-9; maxit = 2000    

    for a in range(A):
        ui = Y_[:, [np.argmax(std(Y_))]]
        Converged = False; num_it = 0
        while not Converged:
             hi = [_Ab_btbinv(R[i].T, ui, not_Rmiss[i].T) for i in range(len(R))]
             si = [_Ab_btbinv(X[i].T, hi[i], not_Xmiss[i].T) for i in range(len(X))]
             js = np.array([y for x in si for y in x])
             for i in range(len(si)):
                 si[i] = si[i] / np.linalg.norm(js)
             ri = [_Ab_btbinv(X[i], si[i], not_Xmiss[i]) for i in range(len(X))]
             jr = np.array([y for x in ri for y in x]).astype(float)
             R_ = np.hstack(R); not_Rmiss_ = np.hstack(not_Rmiss)
             ti = _Ab_btbinv(R_, jr, not_Rmiss_)
             qi = _Ab_btbinv(Y_.T, ti, not_Ymiss.T)
             un = _Ab_btbinv(Y_, qi, not_Ymiss)
             
             if abs((np.linalg.norm(ui) - np.linalg.norm(un))) / np.linalg.norm(ui) < epsilon:
                 Converged = True
             if num_it > maxit:
                 Converged = True
             if Converged:
                 if not shush:
                     print('# Iterations for LV #'+str(a+1)+': ', str(num_it))
                 pi = [_Ab_btbinv(R[i].T, ti, not_Rmiss[i].T) for i in range(len(R))]
                 vi = [_Ab_btbinv(X[i].T, ri[i], not_Xmiss[i].T) for i in range(len(X))]
                 for i in range(len(R)):    
                     R[i] = (R[i] - ti @ pi[i].T) * not_Rmiss[i]
                     X[i] = (X[i] - ri[i] @ vi[i].T) * not_Xmiss[i]
                     Xhat[i] = Xhat[i] + ri[i] @ vi[i].T
                 Y_ = (Y_ - ti @ qi.T) * not_Ymiss
                 
                 if a == 0:
                     T=ti; P=pi; S=si; U=un; Q=qi; H=hi; V=vi; Rscores=ri
                     r2X = []; r2Xpv = []; r2R = []; r2Rpv = []
                     for i in range(len(X)):
                         r2X.append(1 - np.sum(X[i]**2)/TSSX[i])
                         r2Xpv.append((1 - np.sum(X[i]**2, axis=0)/TSSXpv[i]).reshape(-1, 1))
                         r2R.append(1 - np.sum(R[i]**2)/TSSR[i])
                         r2Rpv.append((1 - np.sum(R[i]**2, axis=0)/TSSRpv[i]).reshape(-1, 1))
                     r2Y, r2Ypv = _calc_r2(Y_, TSSY, TSSYpv)
                 else:
                     T = np.hstack((T, ti)); U = np.hstack((U, un)); Q = np.hstack((Q, qi))
                     for i in range(len(P)):   P[i] = np.hstack((P[i], pi[i])) 
                     for i in range(len(S)):   S[i] = np.hstack((S[i], si[i]))
                     for i in range(len(H)):   H[i] = np.hstack((H[i], hi[i]))
                     for i in range(len(V)):   V[i] = np.hstack((V[i], vi[i]))
                     for i in range(len(Rscores)): Rscores[i] = np.hstack((Rscores[i], ri[i]))
                     for i in range(len(X)):   
                         r2X[i]    = np.hstack((r2X[i], 1 - np.sum(X[i]**2)/TSSX[i]))
                         r2Xpv[i]  = np.hstack((r2Xpv[i], (1 - np.sum(X[i]**2, axis=0)/TSSXpv[i]).reshape(-1, 1)))
                         r2R[i]    = np.hstack((r2R[i], 1 - np.sum(R[i]**2)/TSSR[i]))
                         r2Rpv[i]  = np.hstack((r2Rpv[i], (1 - np.sum(R[i]**2, axis=0)/TSSRpv[i]).reshape(-1, 1)))
                     r2Y, r2Ypv = _calc_r2(Y_, TSSY, TSSYpv, r2Y, r2Ypv)
             else:
                 num_it += 1; ui = un
        if a == 0: numIT = num_it
        else:      numIT = np.hstack((numIT, num_it))
    
    for i in range(len(Xhat)):        
        Xhat[i] = Xhat[i] * x_std[i] + x_mean[i]
    
    r2Y, r2Ypv = _r2_cumulative_to_per_component(r2Y, r2Ypv, A)
    r2xc = []; r2rc = []
    for i in range(len(X)):    
        for a in range(A-1, 0, -1):
            r2X[i][a]      = r2X[i][a] - r2X[i][a-1]
            r2Xpv[i][:, a] = r2Xpv[i][:, a] - r2Xpv[i][:, a-1]
            r2R[i][a]      = r2R[i][a] - r2R[i][a-1]
            r2Rpv[i][:, a] = r2Rpv[i][:, a] - r2Rpv[i][:, a-1]
    
    for i, r in enumerate(r2Xpv):
        if i == 0: r2xpv_all = r
        else:      r2xpv_all = np.vstack((r2xpv_all, r))
        r2xc.append(np.cumsum(r2X[i]))
        r2rc.append(np.cumsum(r2R[i]))
    
    eigs = np.var(T, axis=0)
    r2yc = np.cumsum(r2Y)
    r2rc = np.mean(np.array(r2rc), axis=0)
    r2xc = np.mean(np.array(r2xc), axis=0)
    r2x  = np.mean(np.array(r2X), axis=0)
    r2r  = np.mean(np.array(r2R), axis=0)
    
    if not shush:
        print('--------------------------------------------------------------')
        print('LV #     Eig       R2X       sum(R2X)   R2R       sum(R2R)   R2Y       sum(R2Y)')
        if A > 1:    
            for a in range(A):
                print("LV #"+str(a+1)+":   {:6.3f}    {:.3f}     {:.3f}      {:.3f}     {:.3f}      {:.3f}     {:.3f}".format(eigs[a], r2x[a], r2xc[a], r2r[a], r2rc[a], r2Y[a], r2yc[a]))
        else:
           print("LV #1:   {:6.3f}    {:.3f}     {:.3f}      {:.3f}     {:.3f}      {:.3f}     {:.3f}".format(eigs[0], r2x, r2xc[0], r2r, r2rc[0], r2Y, r2yc[0]))
        print('--------------------------------------------------------------')   
        
    var_t = (T.T @ T) / T.shape[0]
    jrpls_obj = {'T':T, 'P':P, 'Q':Q, 'U':U, 'S':S, 'H':H, 'V':V, 'Rscores':Rscores,
              'r2xi':r2X, 'r2xpvi':r2Xpv, 'r2xpv':r2xpv_all,
              'mx':x_mean, 'sx':x_std,
              'r2y':r2Y, 'r2ypv':r2Ypv, 'my':y_mean, 'sy':y_std,
              'r2ri':r2R, 'r2rpvi':r2Rpv, 'mr':r_mean, 'sr':r_std,
              'Xhat':Xhat, 'materials':materials, 'var_t':var_t}  
    if not isinstance(obsidX[0], bool):
        jrpls_obj['obsidXi'] = obsidX; jrpls_obj['varidXi'] = varidX
    varidXall = [materials[i]+':'+varidX[i][j] for i in range(len(materials)) for j in range(len(varidX[i]))]
    jrpls_obj['varidX'] = varidXall    
    if not isinstance(obsidR[0], bool):
       jrpls_obj['obsidRi'] = obsidR; jrpls_obj['varidRi'] = varidR  
    if not isinstance(obsidY, bool):
       jrpls_obj['obsidY'] = obsidY; jrpls_obj['varidY'] = varidY    
       
    T2 = hott2(jrpls_obj, Tnew=T)
    n = T.shape[0]
    T2_lim99 = (((n-1)*(n+1)*A)/(n*(n-A))) * f99(A, (n-A))
    T2_lim95 = (((n-1)*(n+1)*A)/(n*(n-A))) * f95(A, (n-A))    
    
    speX = []; speR = []; speX_lim95 = []; speX_lim99 = []; speR_lim95 = []; speR_lim99 = []
    for i in range(len(X)):
        speX.append(np.sum(X[i]**2, axis=1, keepdims=1))
        s95, s99 = spe_ci(np.sum(X[i]**2, axis=1, keepdims=1))
        speX_lim95.append(s95); speX_lim99.append(s99)
        speR.append(np.sum(R[i]**2, axis=1, keepdims=1))
        s95, s99 = spe_ci(np.sum(R[i]**2, axis=1, keepdims=1))
        speR_lim95.append(s95); speR_lim99.append(s99)
    speY = np.sum(Y_**2, axis=1, keepdims=1)
    speY_lim95, speY_lim99 = spe_ci(speY)
    
    jrpls_obj.update({'T2':T2, 'T2_lim99':T2_lim99, 'T2_lim95':T2_lim95,
        'speX':speX, 'speX_lim99':speX_lim99, 'speX_lim95':speX_lim95,
        'speY':speY, 'speY_lim99':speY_lim99, 'speY_lim95':speY_lim95,
        'speR':speR, 'speR_lim99':speR_lim99, 'speR_lim95':speR_lim95})
    
    Wsi = []; Ws = []
    for i in range(len(S)):
        Wsi.append(S[i] @ np.linalg.pinv(V[i].T @ S[i]))
        if i == 0: Ws = S[i] @ np.linalg.pinv(V[i].T @ S[i])
        else:      Ws = np.vstack((Ws, S[i] @ np.linalg.pinv(V[i].T @ S[i])))    
    jrpls_obj['Ssi'] = Wsi; jrpls_obj['Ss'] = Ws
    jrpls_obj['type'] = 'jrpls'
    return jrpls_obj       

def jrpls_pred(rnew, jrplsobj):
    '''Prediction with a JRPLS model.
    
    Args:
        rnew: dict like {'matid':[(lotid, rvalue)], ...} or list of arrays
        jrplsobj: JRPLS model from pyphi.jrpls
    Returns:
        preds: {'Tnew':tnew, 'Yhat':yhat, 'speR':sper}
    '''
    ok = True
    if isinstance(rnew, list):
        i = 0     
        for r, mr, sr in zip(rnew, jrplsobj['mr'], jrplsobj['sr']):
            if not(len(r) == len(mr[0])):
                ok = False
            if i == 0:
                rnew_ = r; mr_ = mr; sr_ = sr
                Rscores = jrplsobj['Rscores'][i]
                P = jrplsobj['P'][i]
            else:
                rnew_ = np.hstack((rnew_, r))    
                mr_ = np.hstack((mr_, mr)); sr_ = np.hstack((sr_, sr))
                Rscores = np.vstack((Rscores, jrplsobj['Rscores'][i]))
                P = np.vstack((P, jrplsobj['P'][i]))
            i += 1
    elif isinstance(rnew, dict):
        rnew_ = [['*']] * len(jrplsobj['materials'])
        for k in list(rnew.keys()):
            i = jrplsobj['materials'].index(k)
            ri = np.zeros((jrplsobj['mr'][i].shape[1]))
            for m, r in rnew[k]:
                e = jrplsobj['varidRi'][i].index(m)
                ri[e] = r
            rnew_[i] = ri
        return jrpls_pred(rnew_, jrplsobj)
         
    if ok:  
        bkzeros = 0; selmat = []
        for i, r in enumerate(jrplsobj['Rscores']):
            frontzeros = Rscores.shape[0] - bkzeros - r.shape[0]
            row = np.vstack((np.zeros((bkzeros, 1)), np.ones((r.shape[0], 1)), np.zeros((frontzeros, 1))))
            bkzeros += r.shape[0]; selmat.append(row)
            
        rnew_ = ((rnew_ - mr_) / sr_).reshape(-1, 1)
        ti = []
        for a in np.arange(jrplsobj['T'].shape[1]):
            ti_ = (rnew_.T @ Rscores[:, a] / (Rscores[:, a].T @ Rscores[:, a]))
            ti.append(ti_[0])
            rnew_ = rnew_ - (ti_ * P[:, a]).reshape(-1, 1)
        tnew = np.array(ti)
        sper = [np.sum(rnew_[row == 1]**2) for row in selmat]
        yhat = (tnew @ jrplsobj['Q'].T) * jrplsobj['sy'] + jrplsobj['my']
        return {'Tnew':tnew, 'Yhat':yhat, 'speR':sper}    
    else:
        return 'dimensions of rnew did not match model'

def tpls(Xi, Ri, Z, Y, A, *, shush=False):
    '''TPLS Algorithm per Garcia-Munoz Chemom.Intel.Lab.Syst., 133, pp.49-62.

    tpls_obj = pyphi.tpls(Xi, Ri, Z, Y, A)
    Args:
         Xi: Phys. Prop. dictionary of DataFrames {material: df}
         Ri: Blending ratios dictionary of DataFrames {material: df}
         Z:  Process conditions DataFrame [b x p]
         Y:  Product characteristics DataFrame [b x n]
         A:  Number of components
    Returns:
        tpls_obj: Dictionary with TPLS model parameters
    '''
    X = []; varidX = []; obsidX = []
    materials = list(Xi.keys())
    for k in Xi.keys():
        X_, obsidX_, varidX_ = _extract_array(Xi[k])
        X.append(X_); varidX.append(varidX_); obsidX.append(obsidX_)
        
    Y_, obsidY, varidY = _extract_array(Y)
    Z_, obsidZ, varidZ = _extract_array(Z)

    R_list = []; varidR = []; obsidR = []
    for k in materials:  
        R_, obsidR_, varidR_ = _extract_array(Ri[k])
        varidR.append(varidR_); obsidR.append(obsidR_); R_list.append(R_)
        
    x_mean = []; x_std = []; jr_scale = []; r_mean = []; r_std = []
    not_Xmiss = []; not_Rmiss = []; Xhat = []
    TSSX = []; TSSXpv = []; TSSR = []; TSSRpv = []
    X__ = []; R__ = []
    for X_i, R_i in zip(X, R_list):
        X_i, x_mean_, x_std_ = meancenterscale(X_i)
        R_i, r_mean_, r_std_ = meancenterscale(R_i)
        jr_scale_ = np.sqrt(X_i.shape[1])
        X_i = X_i / jr_scale_
        x_mean.append(x_mean_); x_std.append(x_std_); jr_scale.append(jr_scale_)
        r_mean.append(r_mean_); r_std.append(r_std_)
        not_Xmiss.append((~np.isnan(X_i)) * 1)
        not_Rmiss.append((~np.isnan(R_i)) * 1)
        X_i, dummy = n2z(X_i); R_i, dummy = n2z(R_i)
        X__.append(X_i); R__.append(R_i)
        Xhat.append(np.zeros(X_i.shape))
        TSSX.append(np.sum(X_i**2)); TSSXpv.append(np.sum(X_i**2, axis=0))
        TSSR.append(np.sum(R_i**2)); TSSRpv.append(np.sum(R_i**2, axis=0))
    X = X__; R = R__
    
    Y_, y_mean, y_std = meancenterscale(Y_)
    not_Ymiss = (~np.isnan(Y_)) * 1
    Y_, dummy = n2z(Y_)
    TSSY = np.sum(Y_**2); TSSYpv = np.sum(Y_**2, axis=0)
    
    Z_, z_mean, z_std = meancenterscale(Z_)
    not_Zmiss = (~np.isnan(Z_)) * 1
    Z_, dummy = n2z(Z_)
    TSSZ = np.sum(Z_**2); TSSZpv = np.sum(Z_**2, axis=0)
    
    if not shush:
        print('phi.tpls using NIPALS executed on: ' + str(datetime.datetime.now()))
    epsilon = 1E-9; maxit = 2000    

    for a in range(A):
        ui = Y_[:, [np.argmax(std(Y_))]]
        Converged = False; num_it = 0
        while not Converged:
             hi = [_Ab_btbinv(R[i].T, ui, not_Rmiss[i].T) for i in range(len(R))]
             si = [_Ab_btbinv(X[i].T, hi[i], not_Xmiss[i].T) for i in range(len(X))]
             js = np.array([y for x in si for y in x])
             for i in range(len(si)):
                 si[i] = si[i] / np.linalg.norm(js)
             ri = [_Ab_btbinv(X[i], si[i], not_Xmiss[i]) for i in range(len(X))]
             jr = np.array([y for x in ri for y in x]).astype(float)
             R_ = np.hstack(R); not_Rmiss_ = np.hstack(not_Rmiss)
             t_rx = _Ab_btbinv(R_, jr, not_Rmiss_)
             
             wi = _Ab_btbinv(Z_.T, ui, not_Zmiss.T)
             wi = wi / np.linalg.norm(wi)
             t_z = _Ab_btbinv(Z_, wi, not_Zmiss)
             
             Taux = np.hstack((t_rx, t_z))
             plsobj_ = pls(Taux, Y_, 1, mcsX=False, mcsY=False, shush=True, force_nipals=True)
             wt_i = plsobj_['W']; qi = plsobj_['Q']; un = plsobj_['U']; ti = plsobj_['T']
             
             if abs((np.linalg.norm(ui) - np.linalg.norm(un))) / np.linalg.norm(ui) < epsilon:
                 Converged = True
             if num_it > maxit:
                 Converged = True
             if Converged:
                 if not shush:
                     print('# Iterations for LV #'+str(a+1)+': ', str(num_it))
                 pi = [_Ab_btbinv(R[i].T, ti, not_Rmiss[i].T) for i in range(len(R))]
                 vi = [_Ab_btbinv(X[i].T, ri[i], not_Xmiss[i].T) for i in range(len(X))]
                 pzi = _Ab_btbinv(Z_.T, ti, not_Zmiss.T)
                 
                 for i in range(len(R)):    
                     R[i] = (R[i] - ti @ pi[i].T) * not_Rmiss[i]
                     X[i] = (X[i] - ri[i] @ vi[i].T) * not_Xmiss[i]
                     Xhat[i] = Xhat[i] + ri[i] @ vi[i].T
                 Y_ = (Y_ - ti @ qi.T) * not_Ymiss
                 Z_ = (Z_ - ti @ pzi.T) * not_Zmiss 
                 
                 if a == 0:
                     T=ti; P=pi; Pz=pzi; S=si; U=un; Q=qi; H=hi; V=vi
                     Rscores=ri; W=wi; Wt=wt_i
                     r2X = []; r2Xpv = []; r2R = []; r2Rpv = []
                     for i in range(len(X)):
                         r2X.append(1 - np.sum(X[i]**2)/TSSX[i])
                         r2Xpv.append((1 - np.sum(X[i]**2, axis=0)/TSSXpv[i]).reshape(-1, 1))
                         r2R.append(1 - np.sum(R[i]**2)/TSSR[i])
                         r2Rpv.append((1 - np.sum(R[i]**2, axis=0)/TSSRpv[i]).reshape(-1, 1))
                     r2Y, r2Ypv = _calc_r2(Y_, TSSY, TSSYpv)
                     r2Z, r2Zpv = _calc_r2(Z_, TSSZ, TSSZpv)
                 else:
                     T = np.hstack((T, ti)); U = np.hstack((U, un)); Q = np.hstack((Q, qi))
                     W = np.hstack((W, wi)); Wt = np.hstack((Wt, wt_i)); Pz = np.hstack((Pz, pzi))
                     for i in range(len(P)):   P[i] = np.hstack((P[i], pi[i])) 
                     for i in range(len(S)):   S[i] = np.hstack((S[i], si[i]))
                     for i in range(len(H)):   H[i] = np.hstack((H[i], hi[i]))
                     for i in range(len(V)):   V[i] = np.hstack((V[i], vi[i]))
                     for i in range(len(Rscores)): Rscores[i] = np.hstack((Rscores[i], ri[i]))
                     for i in range(len(X)):   
                         r2X[i]    = np.hstack((r2X[i], 1 - np.sum(X[i]**2)/TSSX[i]))
                         r2Xpv[i]  = np.hstack((r2Xpv[i], (1 - np.sum(X[i]**2, axis=0)/TSSXpv[i]).reshape(-1, 1)))
                         r2R[i]    = np.hstack((r2R[i], 1 - np.sum(R[i]**2)/TSSR[i]))
                         r2Rpv[i]  = np.hstack((r2Rpv[i], (1 - np.sum(R[i]**2, axis=0)/TSSRpv[i]).reshape(-1, 1)))
                     r2Y, r2Ypv = _calc_r2(Y_, TSSY, TSSYpv, r2Y, r2Ypv)
                     r2Z, r2Zpv = _calc_r2(Z_, TSSZ, TSSZpv, r2Z, r2Zpv)
             else:
                 num_it += 1; ui = un       
        if a == 0: numIT = num_it
        else:      numIT = np.hstack((numIT, num_it))
    
    for i in range(len(Xhat)):        
        Xhat[i] = Xhat[i] * x_std[i] + x_mean[i]
 
    r2Y, r2Ypv = _r2_cumulative_to_per_component(r2Y, r2Ypv, A)
    r2Z, r2Zpv = _r2_cumulative_to_per_component(r2Z, r2Zpv, A)

    r2xc = []; r2rc = []
    for i in range(len(X)):    
        for a in range(A-1, 0, -1):
            r2X[i][a]      = r2X[i][a] - r2X[i][a-1]
            r2Xpv[i][:, a] = r2Xpv[i][:, a] - r2Xpv[i][:, a-1]
            r2R[i][a]      = r2R[i][a] - r2R[i][a-1]
            r2Rpv[i][:, a] = r2Rpv[i][:, a] - r2Rpv[i][:, a-1]
    
    for i, r in enumerate(r2Xpv):
        if i == 0: r2xpv_all = r
        else:      r2xpv_all = np.vstack((r2xpv_all, r))
        r2xc.append(np.cumsum(r2X[i]))
        r2rc.append(np.cumsum(r2R[i]))

    r2yc = np.cumsum(r2Y); r2zc = np.cumsum(r2Z)
    r2rc = np.mean(np.array(r2rc), axis=0)
    r2xc = np.mean(np.array(r2xc), axis=0)
    r2x  = np.mean(np.array(r2X), axis=0)
    r2r  = np.mean(np.array(r2R), axis=0)
    
    if not shush:
        print('--------------------------------------------------------------')
        print('LV #     R2X       sum(R2X)   R2R       sum(R2R)   R2Z       sum(R2Z)   R2Y       sum(R2Y)')
        if A > 1:    
            for a in range(A):
                print("LV #"+str(a+1)+":   {:.3f}     {:.3f}      {:.3f}     {:.3f}      {:.3f}     {:.3f}      {:.3f}     {:.3f}".format(r2x[a], r2xc[a], r2r[a], r2rc[a], r2Z[a], r2zc[a], r2Y[a], r2yc[a]))
        else:
           print("LV #1:   {:.3f}     {:.3f}      {:.3f}     {:.3f}      {:.3f}     {:.3f}      {:.3f}     {:.3f}".format(r2x, r2xc[0], r2r, r2rc[0], r2Z, r2zc[0], r2Y, r2yc[0]))
        print('--------------------------------------------------------------')   
        
    var_t = (T.T @ T) / T.shape[0]
    tpls_obj = {'T':T, 'P':P, 'Q':Q, 'U':U, 'S':S, 'H':H, 'V':V, 'Rscores':Rscores,
              'r2xi':r2X, 'r2xpvi':r2Xpv, 'r2xpv':r2xpv_all,
              'mx':x_mean, 'sx':x_std,
              'r2y':r2Y, 'r2ypv':r2Ypv, 'my':y_mean, 'sy':y_std,
              'r2ri':r2R, 'r2rpvi':r2Rpv, 'mr':r_mean, 'sr':r_std,
              'r2z':r2Z, 'r2zpv':r2Zpv, 'mz':z_mean, 'sz':z_std,
              'Xhat':Xhat, 'materials':materials, 'Wt':Wt, 'W':W, 'Pz':Pz, 'var_t':var_t}  
    if not isinstance(obsidX[0], bool):
        tpls_obj['obsidXi'] = obsidX; tpls_obj['varidXi'] = varidX
    varidXall = [materials[i]+':'+varidX[i][j] for i in range(len(materials)) for j in range(len(varidX[i]))]
    tpls_obj['varidX'] = varidXall    
    if not isinstance(obsidR[0], bool):
       tpls_obj['obsidRi'] = obsidR; tpls_obj['varidRi'] = varidR  
    if not isinstance(obsidY, bool):
       tpls_obj['obsidY'] = obsidY; tpls_obj['varidY'] = varidY    
    if not isinstance(obsidZ, bool):
       tpls_obj['obsidZ'] = obsidZ; tpls_obj['varidZ'] = varidZ  
       
    T2 = hott2(tpls_obj, Tnew=T)
    n = T.shape[0]
    T2_lim99 = (((n-1)*(n+1)*A)/(n*(n-A))) * f99(A, (n-A))
    T2_lim95 = (((n-1)*(n+1)*A)/(n*(n-A))) * f95(A, (n-A))    
    
    speX = []; speR = []; speX_lim95 = []; speX_lim99 = []; speR_lim95 = []; speR_lim99 = []
    for i in range(len(X)):
        speX.append(np.sum(X[i]**2, axis=1, keepdims=1))
        s95, s99 = spe_ci(np.sum(X[i]**2, axis=1, keepdims=1))
        speX_lim95.append(s95); speX_lim99.append(s99)
        speR.append(np.sum(R[i]**2, axis=1, keepdims=1))
        s95, s99 = spe_ci(np.sum(R[i]**2, axis=1, keepdims=1))
        speR_lim95.append(s95); speR_lim99.append(s99)
    speY = np.sum(Y_**2, axis=1, keepdims=1); speY_lim95, speY_lim99 = spe_ci(speY)
    speZ = np.sum(Z_**2, axis=1, keepdims=1); speZ_lim95, speZ_lim99 = spe_ci(speZ)
    
    tpls_obj.update({'T2':T2, 'T2_lim99':T2_lim99, 'T2_lim95':T2_lim95,
        'speX':speX, 'speX_lim99':speX_lim99, 'speX_lim95':speX_lim95,
        'speY':speY, 'speY_lim99':speY_lim99, 'speY_lim95':speY_lim95,
        'speR':speR, 'speR_lim99':speR_lim99, 'speR_lim95':speR_lim95,
        'speZ':speZ, 'speZ_lim99':speZ_lim99, 'speZ_lim95':speZ_lim95})
    
    Wsi = []; Ws = []
    for i in range(len(S)):
        Wsi.append(S[i] @ np.linalg.pinv(V[i].T @ S[i]))
        if i == 0: Ws = S[i] @ np.linalg.pinv(V[i].T @ S[i])
        else:      Ws = np.vstack((Ws, S[i] @ np.linalg.pinv(V[i].T @ S[i])))    
    tpls_obj['Ssi'] = Wsi; tpls_obj['Ss'] = Ws
    tpls_obj['type'] = 'tpls'
    tpls_obj['Ws'] = W @ np.linalg.pinv(Pz.T @ W)
    return tpls_obj       

def jypls(Xi, Yi, A, *, shush=False):
    '''Joint-Y PLS (JYPLS) model.
    
    Per Garcia-Munoz, MacGregor, Kourti, Chemom. Intell. Lab. Syst. 79 (2005) 101-114.
    
    Fits a PLS model across multiple campaigns where each campaign has its own
    X block (different variables per campaign) but all campaigns share the same
    Y column space. The Q loadings are jointly estimated from all campaigns.
    
    jypls_obj = pyphi.jypls(Xi, Yi, A)
    
    Args:
        Xi: dict of DataFrames {'campaign_name': df_X}
            Each X can have different columns. First column is obs ID.
        Yi: dict of DataFrames {'campaign_name': df_Y}
            All Y blocks MUST have the same columns (joint Y space).
            Keys must match Xi. First column is obs ID.
        A:  Number of latent variables
    
    Returns:
        jypls_obj: Dictionary with model parameters including per-campaign
                   W, P, T, Ws and shared Q loadings.
    '''
    campaigns = list(Xi.keys())
    
    # --- Validate keys match ---
    if set(Xi.keys()) != set(Yi.keys()):
        raise ValueError(
            f"Xi and Yi must have the same campaign keys. "
            f"Xi keys: {list(Xi.keys())}, Yi keys: {list(Yi.keys())}")
    
    # --- Validate and reconcile per campaign, check Y columns ---
    y_col_ref = None
    y_col_ref_name = None
    for k in campaigns:
        Xi[k], Yi[k] = _validate_inputs(Xi[k], Yi[k])
        if isinstance(Yi[k], pd.DataFrame):
            ycols = Yi[k].columns[1:].tolist()
            if y_col_ref is None:
                y_col_ref = ycols
                y_col_ref_name = k
            elif ycols != y_col_ref:
                raise ValueError(
                    f"All Y blocks must have the same columns (Joint-Y space). "
                    f"Campaign '{k}' has columns {ycols[:5]}... "
                    f"but campaign '{y_col_ref_name}' has {y_col_ref[:5]}...")
    
    # --- Extract arrays ---
    X = []; Y = []; varidX = []; obsidX = []; varidY = False; obsidY = []
    for k in campaigns:
        x_, ox, vx = _extract_array(Xi[k])
        y_, oy, vy = _extract_array(Yi[k])
        X.append(x_); Y.append(y_)
        varidX.append(vx); obsidX.append(ox)
        obsidY.append(oy)
        if vy is not False:
            varidY = vy  # same for all campaigns
    
    # --- Compute joint Y mean and std from stacked Y ---
    Y_stacked = np.vstack(Y)
    y_mean = mean(Y_stacked)
    y_std  = std(Y_stacked)
    
    # --- Scale each block ---
    x_mean = []; x_std = []; blk_scale = []
    not_Xmiss = []; not_Ymiss = []
    TSSX = []; TSSXpv = []
    TSSY_i = []; TSSYpv_i = []
    
    for i in range(len(campaigns)):
        # X_i: mean center, autoscale, block scale
        X[i], xm, xs = meancenterscale(X[i])
        bs = np.sqrt(X[i].shape[1])
        X[i] = X[i] / bs
        x_mean.append(xm); x_std.append(xs); blk_scale.append(bs)
        not_Xmiss.append((~np.isnan(X[i])) * 1)
        X[i], _ = n2z(X[i])
        TSSX.append(np.sum(X[i]**2))
        TSSXpv.append(np.sum(X[i]**2, axis=0))
        
        # Y_i: scale with joint mean/std
        Y[i] = (Y[i] - y_mean) / y_std
        not_Ymiss.append((~np.isnan(Y[i])) * 1)
        Y[i], _ = n2z(Y[i])
        TSSY_i.append(np.sum(Y[i]**2))
        TSSYpv_i.append(np.sum(Y[i]**2, axis=0))
    
    # Joint Y TSS from scaled stacked Y
    Y_stacked_mcs = np.vstack(Y)
    TSSY   = np.sum(Y_stacked_mcs**2)
    TSSYpv = np.sum(Y_stacked_mcs**2, axis=0)
    
    if not shush:
        print('phi.jypls using NIPALS executed on: ' + str(datetime.datetime.now()))
    
    # --- NIPALS ---
    epsilon = 1E-9; maxit = 2000
    
    for a in range(A):
        # Initialize u_i from column with max variance in stacked Y
        Y_stack_curr = np.vstack(Y)
        max_col = np.argmax(std(Y_stack_curr))
        ui = [Y[i][:, [max_col]] for i in range(len(campaigns))]
        
        Converged = False; num_it = 0
        while not Converged:
            # Step 1: w_i = X_i' u_i / (u_i' u_i)
            wi = []
            for i in range(len(campaigns)):
                w_ = _Ab_btbinv(X[i].T, ui[i], not_Xmiss[i].T)
                w_ = w_ / np.linalg.norm(w_)
                wi.append(w_)
            
            # Step 2: t_i = X_i w_i / (w_i' w_i)
            ti = []
            for i in range(len(campaigns)):
                t_ = _Ab_btbinv(X[i], wi[i], not_Xmiss[i])
                ti.append(t_)
            
            # Step 3: Joint q from stacked t and Y
            t_stacked = np.vstack(ti)
            Y_stack_curr = np.vstack(Y)
            not_Ymiss_stacked = np.vstack(not_Ymiss)
            qi = _Ab_btbinv(Y_stack_curr.T, t_stacked, not_Ymiss_stacked.T)
            
            # Step 4: u_i = Y_i q / (q' q)
            un = []
            for i in range(len(campaigns)):
                u_ = _Ab_btbinv(Y[i], qi, not_Ymiss[i])
                un.append(u_)
            
            # Check convergence across all campaigns
            conv = all(
                abs(np.linalg.norm(ui[i]) - np.linalg.norm(un[i])) / max(np.linalg.norm(ui[i]), 1e-16) < epsilon
                for i in range(len(campaigns)))
            if conv or num_it > maxit:
                Converged = True
            
            if Converged:
                if not shush:
                    print('# Iterations for LV #'+str(a+1)+': ', str(num_it))
                
                # Sign convention: ensure var(t>0) > var(t<0) on stacked scores
                t_stacked = np.vstack(ti)
                if (len(t_stacked[t_stacked < 0]) > 0) and (len(t_stacked[t_stacked > 0]) > 0):
                    if np.var(t_stacked[t_stacked < 0]) > np.var(t_stacked[t_stacked >= 0]):
                        ti = [-t for t in ti]
                        wi = [-w for w in wi]
                        un = [-u for u in un]
                        qi = -qi
                
                # Compute p_i for deflation: p_i = X_i' t_i / (t_i' t_i)
                pi = []
                for i in range(len(campaigns)):
                    p_ = _Ab_btbinv(X[i].T, ti[i], not_Xmiss[i].T)
                    pi.append(p_)
                
                # Deflate all blocks
                for i in range(len(campaigns)):
                    X[i] = (X[i] - ti[i] @ pi[i].T) * not_Xmiss[i]
                    Y[i] = (Y[i] - ti[i] @ qi.T)    * not_Ymiss[i]
                
                # Store
                if a == 0:
                    T_blk = [[t] for t in ti]
                    P_blk = [[p] for p in pi]
                    W_blk = [[w] for w in wi]
                    U_blk = [[u] for u in un]
                    Q = qi
                    r2X = []; r2Xpv = []; r2Yi = []
                    for i in range(len(campaigns)):
                        r2X.append(1 - np.sum(X[i]**2) / TSSX[i])
                        r2Xpv.append((1 - np.sum(X[i]**2, axis=0) / TSSXpv[i]).reshape(-1, 1))
                        r2Yi.append(1 - np.sum(Y[i]**2) / TSSY_i[i])
                    Y_stack_res = np.vstack(Y)
                    r2Y, r2Ypv = _calc_r2(Y_stack_res, TSSY, TSSYpv)
                else:
                    for i in range(len(campaigns)):
                        T_blk[i].append(ti[i])
                        P_blk[i].append(pi[i])
                        W_blk[i].append(wi[i])
                        U_blk[i].append(un[i])
                    Q = np.hstack((Q, qi))
                    for i in range(len(campaigns)):
                        r2X[i]    = np.hstack((r2X[i], 1 - np.sum(X[i]**2) / TSSX[i]))
                        r2Xpv[i]  = np.hstack((r2Xpv[i], (1 - np.sum(X[i]**2, axis=0) / TSSXpv[i]).reshape(-1, 1)))
                        r2Yi[i]   = np.hstack((r2Yi[i], 1 - np.sum(Y[i]**2) / TSSY_i[i]))
                    Y_stack_res = np.vstack(Y)
                    r2Y, r2Ypv = _calc_r2(Y_stack_res, TSSY, TSSYpv, r2Y, r2Ypv)
            else:
                num_it += 1
                ui = un
    
    # --- Assemble per-campaign matrices ---
    T = [np.hstack(t_list) for t_list in T_blk]
    P = [np.hstack(p_list) for p_list in P_blk]
    W = [np.hstack(w_list) for w_list in W_blk]
    U = [np.hstack(u_list) for u_list in U_blk]
    
    # Ws per campaign
    Ws = []
    for i in range(len(campaigns)):
        Ws_i = W[i] @ np.linalg.pinv(P[i].T @ W[i])
        Ws_i[:, 0] = W[i][:, 0]
        Ws.append(Ws_i)
    
    # --- R2 per component ---
    r2Y, r2Ypv = _r2_cumulative_to_per_component(r2Y, r2Ypv, A)
    for i in range(len(campaigns)):
        for aa in range(A-1, 0, -1):
            r2X[i][aa]       = r2X[i][aa] - r2X[i][aa-1]
            r2Xpv[i][:, aa]  = r2Xpv[i][:, aa] - r2Xpv[i][:, aa-1]
            r2Yi[i][aa]      = r2Yi[i][aa] - r2Yi[i][aa-1]
    
    # Stacked T for diagnostics
    T_stacked = np.vstack(T)
    var_t = (T_stacked.T @ T_stacked) / T_stacked.shape[0]
    eigs = np.var(T_stacked, axis=0)
    
    # Aggregate R2X across campaigns
    r2x_avg = np.mean(np.array(r2X), axis=0)
    r2xc    = np.cumsum(r2x_avg)
    r2yc    = np.cumsum(r2Y)
    
    if not shush:
        print('--------------------------------------------------------------')
        print('LV #     Eig       R2X(avg)  sum(R2X)   R2Y       sum(R2Y)')
        if A > 1:    
            for aa in range(A):
                print("LV #"+str(aa+1)+":   {:6.3f}    {:.3f}     {:.3f}      {:.3f}     {:.3f}".format(
                    eigs[aa], r2x_avg[aa], r2xc[aa], r2Y[aa], r2yc[aa]))
        else:
           print("LV #1:   {:6.3f}    {:.3f}     {:.3f}      {:.3f}     {:.3f}".format(
               eigs[0], r2x_avg, r2xc[0], r2Y, r2yc[0]))
        print('')
        print('R2Y and R2X per campaign:')
        for i, k in enumerate(campaigns):
            r2xi_c = np.cumsum(r2X[i]) if A > 1 else r2X[i]
            r2yi_c = np.cumsum(r2Yi[i]) if A > 1 else r2Yi[i]
            r2xi_t = r2xi_c[-1] if A > 1 else r2xi_c
            r2yi_t = r2yi_c[-1] if A > 1 else r2yi_c
            print("  {}: R2X={:.3f}  R2Y={:.3f}".format(k, r2xi_t, r2yi_t))
        print('--------------------------------------------------------------')   
    
    # --- Diagnostics ---
    T2 = hott2({'T': T_stacked, 'var_t': var_t}, Tnew=T_stacked)
    n = T_stacked.shape[0]
    T2_lim99 = (((n-1)*(n+1)*A) / (n*(n-A))) * f99(A, (n-A))
    T2_lim95 = (((n-1)*(n+1)*A) / (n*(n-A))) * f95(A, (n-A))
    
    speX = []; speX_lim95 = []; speX_lim99 = []
    speY = []; speY_lim95 = []; speY_lim99 = []
    for i in range(len(campaigns)):
        sx_ = np.sum(X[i]**2, axis=1, keepdims=1)
        s95, s99 = spe_ci(sx_)
        speX.append(sx_); speX_lim95.append(s95); speX_lim99.append(s99)
        sy_ = np.sum(Y[i]**2, axis=1, keepdims=1)
        s95, s99 = spe_ci(sy_)
        speY.append(sy_); speY_lim95.append(s95); speY_lim99.append(s99)
    
    # Build varidX aggregated list
    varidXall = []
    for i, k in enumerate(campaigns):
        if varidX[i] is not False:
            varidXall.append([k + ':' + v for v in varidX[i]])
        else:
            varidXall.append([k + ':Var' + str(j) for j in range(X[i].shape[1] if isinstance(X[i], np.ndarray) else 0)])
    
    jypls_obj = {
        'T': T, 'P': P, 'Q': Q, 'W': W, 'Ws': Ws, 'U': U,
        'r2xi': r2X, 'r2xpvi': r2Xpv,
        'r2y': r2Y, 'r2ypv': r2Ypv, 'r2yi': r2Yi,
        'mx': x_mean, 'sx': x_std, 'blk_scale': blk_scale,
        'my': y_mean, 'sy': y_std,
        'campaigns': campaigns, 'var_t': var_t,
        'varidXi': varidX, 'varidXall': varidXall,
        'obsidXi': obsidX, 'obsidYi': obsidY,
        'T2': T2, 'T2_lim99': T2_lim99, 'T2_lim95': T2_lim95,
        'speX': speX, 'speX_lim99': speX_lim99, 'speX_lim95': speX_lim95,
        'speY': speY, 'speY_lim99': speY_lim99, 'speY_lim95': speY_lim95,
        'type': 'jypls'
    }
    if varidY is not False:
        jypls_obj['varidY'] = varidY
    
    return jypls_obj

def jypls_pred(xnew, campaign, jypls_obj):
    '''Prediction with a JYPLS model for a new observation from a specific campaign.
    
    pred = pyphi.jypls_pred(xnew, campaign, jypls_obj)
    
    Args:
        xnew:     DataFrame, ndarray, or 1D array with new X observation(s).
                  Variables must match those of the specified campaign.
        campaign: string, name of the campaign this observation belongs to
                  (must match a key used when building the model).
        jypls_obj: JYPLS model object built with pyphi.jypls.
    
    Returns:
        pred: dict {'Tnew', 'Yhat', 'speX', 'T2'}
    '''
    if campaign not in jypls_obj['campaigns']:
        raise ValueError(
            f"Campaign '{campaign}' not found in model. "
            f"Available campaigns: {jypls_obj['campaigns']}")
    
    idx = jypls_obj['campaigns'].index(campaign)
    
    # Extract array
    if isinstance(xnew, pd.DataFrame):
        X_ = np.array(xnew.values[:, 1:]).astype(float)
    elif isinstance(xnew, np.ndarray):
        X_ = xnew.copy()
        if X_.ndim == 1:
            X_ = X_.reshape(1, -1)
    
    # Validate dimensions
    expected_cols = jypls_obj['mx'][idx].shape[1] if jypls_obj['mx'][idx].ndim == 2 else len(jypls_obj['mx'][idx])
    if X_.shape[1] != expected_cols:
        raise ValueError(
            f"xnew has {X_.shape[1]} variables but campaign '{campaign}' "
            f"expects {expected_cols} variables.")
    
    # Scale: mean center, autoscale, block scale (same as training)
    X_mcs = (X_ - jypls_obj['mx'][idx]) / jypls_obj['sx'][idx]
    X_mcs = X_mcs / jypls_obj['blk_scale'][idx]
    
    # Handle missing data
    X_nan_map = np.isnan(X_mcs)
    
    if not X_nan_map.any():
        # Complete data: use Ws for direct projection
        tnew = X_mcs @ jypls_obj['Ws'][idx]
        xhat = tnew @ jypls_obj['P'][idx].T
        speX = np.sum((X_mcs - xhat)**2, axis=1, keepdims=True)
    else:
        # Missing data: deflation-based projection using W
        not_Xmiss = (~X_nan_map) * 1
        X_mcs, _ = n2z(X_mcs)
        for i in range(X_mcs.shape[0]):
            row_map = not_Xmiss[[i], :]
            tempW = jypls_obj['W'][idx] * row_map.T
            for aa in range(jypls_obj['Q'].shape[1]):
                WTW = tempW[:, [aa]].T @ tempW[:, [aa]]
                tnew_aux, _, _, _ = np.linalg.lstsq(WTW, tempW[:, [aa]].T @ X_mcs[[i], :].T, rcond=None)
                X_mcs[[i], :] = (X_mcs[[i], :] - tnew_aux @ jypls_obj['P'][idx][:, [aa]].T) * row_map
                if aa == 0: tnew_ = tnew_aux
                else:       tnew_ = np.vstack((tnew_, tnew_aux))
            if i == 0: tnew = tnew_.T
            else:      tnew = np.vstack((tnew, tnew_.T))
        xhat = tnew @ jypls_obj['P'][idx].T
        speX = np.sum((X_mcs - xhat)**2 * not_Xmiss, axis=1, keepdims=True)
    
    # Predict Y using shared Q, unscale
    yhat = tnew @ jypls_obj['Q'].T * jypls_obj['sy'] + jypls_obj['my']
    
    # T2 using pooled var_t
    var_t_inv = np.linalg.inv(jypls_obj['var_t'])
    T2 = np.sum((tnew @ var_t_inv) * tnew, axis=1)
    
    return {'Tnew': tnew, 'Yhat': yhat, 'speX': speX, 'T2': T2}

def tpls_pred(rnew, znew, tplsobj):
    '''Prediction with a TPLS model.
    
    Args:
        rnew: dict like {'matid':[(lotid, rvalue)], ...} or list of arrays
        znew: DataFrame, list, or ndarray with process conditions
        tplsobj: TPLS model from pyphi.tpls
    Returns:
        preds: {'Tnew':tnew, 'Yhat':yhat, 'speR':sper, 'speZ':spez}
    '''
    ok = True
    if isinstance(rnew, list):
        i = 0     
        for r, mr, sr in zip(rnew, tplsobj['mr'], tplsobj['sr']):
            if not(len(r) == len(mr[0])):
                ok = False
            if i == 0:
                rnew_ = r; mr_ = mr; sr_ = sr
                Rscores = tplsobj['Rscores'][i]
                P = tplsobj['P'][i]
            else:
                rnew_ = np.hstack((rnew_, r))    
                mr_ = np.hstack((mr_, mr)); sr_ = np.hstack((sr_, sr))
                Rscores = np.vstack((Rscores, tplsobj['Rscores'][i]))
                P = np.vstack((P, tplsobj['P'][i]))
            i += 1
    elif isinstance(rnew, dict):
        rnew_ = [['*']] * len(tplsobj['materials'])
        for k in list(rnew.keys()):
            i = tplsobj['materials'].index(k)
            ri = np.zeros((tplsobj['mr'][i].shape[1]))
            for m, r in rnew[k]:
                e = tplsobj['varidRi'][i].index(m)
                ri[e] = r
            rnew_[i] = ri
        return tpls_pred(rnew_, znew, tplsobj)
    
    if isinstance(znew, pd.DataFrame):
        znew_ = znew.values.reshape(-1)[1:].astype(float)
    elif isinstance(znew, list):
        znew_ = np.array(znew)
    elif isinstance(znew, np.ndarray):
        znew_ = znew.copy()
        
    if not(len(znew_) == tplsobj['mz'].shape[1]):
        ok = False
   
    if ok:  
        bkzeros = 0; selmat = []
        for i, r in enumerate(tplsobj['Rscores']):
            frontzeros = Rscores.shape[0] - bkzeros - r.shape[0]
            row = np.vstack((np.zeros((bkzeros, 1)), np.ones((r.shape[0], 1)), np.zeros((frontzeros, 1))))
            bkzeros += r.shape[0]; selmat.append(row)
       
        rnew_ = ((rnew_ - mr_) / sr_).reshape(-1, 1)
        znew_ = ((znew_ - tplsobj['mz']) / tplsobj['sz']).reshape(-1, 1)
        
        tnew = []
        for a in np.arange(tplsobj['T'].shape[1]):
            ti_rx_ = (rnew_.T @ Rscores[:, a] / (Rscores[:, a].T @ Rscores[:, a]))
            ti_z_ = znew_.T @ tplsobj['W'][:, a]
            ti_ = np.array([ti_rx_, ti_z_]).reshape(1, -1) @ tplsobj['Wt'][:, a]
            tnew.append(ti_[0])
            rnew_ = rnew_ - (ti_ * P[:, a]).reshape(-1, 1)
            znew_ = znew_ - (ti_ * tplsobj['Pz'][:, a]).reshape(-1, 1)
        
        sper = [np.sum(rnew_[row == 1]**2) for row in selmat]
        spez = np.sum(znew_**2)
        tnew = np.array(tnew)
        yhat = (tnew @ tplsobj['Q'].T) * tplsobj['sy'] + tplsobj['my']
        return {'Tnew':tnew, 'Yhat':yhat, 'speR':sper, 'speZ':spez}    
    else:
        return 'dimensions of rnew or znew did not match model'

# =============================================================================
# Varimax rotation
# =============================================================================

def varimax_(X, gamma=1.0, q=20, tol=1e-6):
    p, k = X.shape
    R = eye(k); d = 0
    for i in range(q):
        d_ = d
        Lambda = dot(X, R)
        u, s, vh = svd(dot(X.T, asarray(Lambda)**3 - (gamma/p) * dot(Lambda, diag(diag(dot(Lambda.T, Lambda))))))
        R = dot(u, vh)
        d = sum(s)
        if d_ != 0 and d/d_ < 1 + tol: break
    return dot(X, R)

def varimax_rotation(mvm_obj, X, *, Y=False):
    '''VariMax rotation on a PCA or PLS model.'''
    mvmobj = mvm_obj.copy()
    if isinstance(X, pd.DataFrame): X_ = X.values[:,1:].astype(float)
    else: X_ = X.copy()
    if isinstance(Y, pd.DataFrame): Y_ = Y.values[:,1:].astype(float)
    elif isinstance(Y, np.ndarray): Y_ = Y.copy()
        
    X_ = (X_ - mvmobj['mx']) / mvmobj['sx']
    not_Xmiss = (~np.isnan(X_)) * 1
    X_, Xmap = n2z(X_)
    TSSX = np.sum(X_**2); TSSXpv = np.sum(X_**2, axis=0)
    if not isinstance(Y, bool):
        Y_ = (Y_ - mvmobj['my']) / mvmobj['sy']
        not_Ymiss = (~np.isnan(Y_)) * 1
        Y_, Ymap = n2z(Y_)
        TSSY = np.sum(Y_**2); TSSYpv = np.sum(Y_**2, axis=0)
        
    A = mvmobj['T'].shape[1]
    if 'Q' in mvmobj:
        Wrot = varimax_(mvmobj['W'])
        Trot=[]; Prot=[]; Qrot=[]; Urot=[]
        r2X = None; r2Xpv = None; r2Y = None; r2Ypv = None
        for a in np.arange(A):
            ti = _Ab_btbinv(X_, Wrot[:,a], not_Xmiss)
            pi = _Ab_btbinv(X_.T, ti, not_Xmiss.T)
            qi = _Ab_btbinv(Y_.T, ti, not_Ymiss.T)
            ui = _Ab_btbinv(Y_, qi, not_Ymiss)
            X_ = (X_ - ti @ pi.T) * not_Xmiss
            Y_ = (Y_ - ti @ qi.T) * not_Ymiss
            
            r2Xpv_new = np.zeros(len(TSSXpv))
            r2Xpv_new[TSSXpv>0] = 1 - (np.sum(X_**2, axis=0)[TSSXpv>0] / TSSXpv[TSSXpv>0])
            r2X_new = 1 - np.sum(X_**2) / TSSX
            r2Ypv_new = np.zeros(len(TSSYpv))
            r2Ypv_new[TSSYpv>0] = 1 - (np.sum(Y_**2, axis=0)[TSSYpv>0] / TSSYpv[TSSYpv>0])
            r2Y_new = 1 - np.sum(Y_**2) / TSSY
            
            if r2X is None:
                r2X = r2X_new; r2Xpv = r2Xpv_new.reshape(-1,1)
                r2Y = r2Y_new; r2Ypv = r2Ypv_new.reshape(-1,1)
            else:
                r2X = np.hstack((r2X, r2X_new)); r2Xpv = np.hstack((r2Xpv, r2Xpv_new.reshape(-1,1)))
                r2Y = np.hstack((r2Y, r2Y_new)); r2Ypv = np.hstack((r2Ypv, r2Ypv_new.reshape(-1,1)))
            Trot.append(ti); Prot.append(pi); Qrot.append(qi); Urot.append(ui)
                            
        r2X, r2Xpv = _r2_cumulative_to_per_component(r2X, r2Xpv, A)
        r2Y, r2Ypv = _r2_cumulative_to_per_component(r2Y, r2Ypv, A)
        Trot = np.array(Trot).T[0]; Prot = np.array(Prot).T[0]
        Qrot = np.array(Qrot).T[0]; Urot = np.array(Urot).T[0]
        Wsrot = Wrot @ np.linalg.pinv(Prot.T @ Wrot)
        mvmobj.update({'W':Wrot, 'T':Trot, 'P':Prot, 'Q':Qrot, 'U':Urot, 'Ws':Wsrot,
                       'r2x':r2X, 'r2xpv':r2Xpv, 'r2y':r2Y, 'r2ypv':r2Ypv})
    else:
        Prot = varimax_(mvmobj['P'])
        Trot = []; r2X = None; r2Xpv = None
        for a in np.arange(A):
            ti = _Ab_btbinv(X_, Prot[:,a], not_Xmiss)
            Trot.append(ti)
            X_ = (X_ - ti @ Prot[:,[a]].T) * not_Xmiss
            r2Xpv_new = np.zeros(len(TSSXpv))
            r2Xpv_new[TSSXpv>0] = 1 - (np.sum(X_**2, axis=0)[TSSXpv>0] / TSSXpv[TSSXpv>0])
            r2X_new = 1 - np.sum(X_**2) / TSSX
            if r2X is None:
                r2X = r2X_new; r2Xpv = r2Xpv_new.reshape(-1,1)
            else:
                r2X = np.hstack((r2X, r2X_new)); r2Xpv = np.hstack((r2Xpv, r2Xpv_new.reshape(-1,1)))
        r2X, r2Xpv = _r2_cumulative_to_per_component(r2X, r2Xpv, A)
        mvmobj.update({'P':Prot, 'T':np.array(Trot).T[0], 'r2x':r2X, 'r2xpv':r2Xpv})
    return mvmobj

# =============================================================================
# Build polynomial
# =============================================================================

def findstr(string):
    return [i for i, s in enumerate(string) if s == '*' or s == '/']

def evalvar(data, vname):
    if vname.find('^') > 0:
        actual_vname = vname[:vname.find('^')].strip()
        if actual_vname in data.columns[1:]:
            power = float(vname[vname.find('^')+1:])
            return data[actual_vname].values.reshape(-1, 1)**power
        return False
    else:
        actual_vname = vname.strip()
        if actual_vname in data.columns[1:]:
            return data[actual_vname].values.reshape(-1, 1)
        return False

def writeeq(beta_, features_):
    eq_str = []
    for b, f, i in zip(beta_, features_, np.arange(len(features_))):
        if f == 'Bias':
            eq_str.append(str(b) if b < 0 else ' + '+str(b))
        else:
            eq_str.append((str(b) if b < 0 or i == 0 else ' + '+str(b)) + ' * '+f)
    return ''.join(eq_str)

def build_polynomial(data, factors, response, *, bias_term=True):
    '''Linear regression with variable selection assisted by PLS.'''
    for j, f in enumerate(factors):
        if f.find('*') > 0 or f.find('/') > 0:
            indx = findstr(f)
            for ii, i in enumerate(indx):
                if ii == 0: 
                    vname1 = f[0:i]
                    vname2 = f[i+1:indx[1]] if len(indx) > 1 else f[i+1:]
                    vals1 = evalvar(data, vname1); vals2 = evalvar(data, vname2)
                    xcol = vals1 * vals2 if f[i] == '*' else vals1 / vals2
                else:
                    vname = f[i+1:] if len(indx) == ii+1 else f[i+1:indx[ii+1]]
                    vals = evalvar(data, vname)
                    xcol = xcol * vals if f[i] == '*' else xcol / vals
            X = xcol if j == 0 else np.hstack((X, xcol))
        else:
            temp = evalvar(data, f)
            X = temp if j == 0 else np.hstack((X, temp))
    print('Built X from factors')            
    X_df = pd.DataFrame(X, columns=factors)
    X_df.insert(0, data.columns[0], data[data.columns[0]].values)
    Y_df = data[[data.columns[0], response]]
    Y_arr = data[response].values
    pls_obj = pls(X_df, Y_df, len(factors))
    Ypred = pls_pred(X_df, pls_obj)['Yhat']
    RMSE = [np.sqrt(np.mean((Y_df.values[:,1:].astype(float) - Ypred)**2))]
    
    vip = np.sum(np.abs(pls_obj['Ws'] * pls_obj['r2y']), axis=1).ravel()
    sort_indx = np.argsort(-vip)
    sort_asc_indx = np.argsort(vip)
    vip_sorted = vip[sort_indx]
    sorted_factors = [factors[i] for i in sort_indx]
    plt.figure()
    plt.bar(np.arange(len(sorted_factors)), vip_sorted)
    plt.xticks(np.arange(len(sorted_factors)), labels=sorted_factors, rotation=60)
    plt.ylabel('VIP'); plt.xlabel('Factors'); plt.tight_layout()
    
    sorted_asc_factors = [factors[i] for i in sort_asc_indx]
    X_df_m = X_df.copy()
    for f in sorted_asc_factors[:-1]:
        X_df_m.drop(f, axis=1, inplace=True)
        pls_obj_m = pls(X_df_m, Y_df, X_df_m.shape[1]-1, shush=True)
        Ypred = pls_pred(X_df_m, pls_obj_m)['Yhat']
        RMSE.append(np.sqrt(np.mean((Y_df.values[:,1:].astype(float) - Ypred)**2)))
   
    sorted_asc_labels = ['Full'] + [factors[i] for i in sort_asc_indx[:-1]]
    plt.figure()
    plt.bar(np.arange(len(factors)), RMSE)
    plt.xticks(np.arange(len(factors)), labels=sorted_asc_labels, rotation=60)
    plt.ylabel('RMSE ('+response+')'); plt.xlabel('Factors removed from model'); plt.tight_layout()
    Xaug = np.hstack((X, np.ones((X.shape[0], 1))))
    factors_out = factors.copy(); factors_out.append('Bias')
    betasOLSlssq, r1, r2, r3 = np.linalg.lstsq(Xaug, Y_arr, rcond=None)
    eqstr = writeeq(betasOLSlssq, factors_out)
    return betasOLSlssq, factors_out, Xaug, Y_arr, eqstr

# =============================================================================
# CCA
# =============================================================================

def cca(X, Y, tol=1e-6, max_iter=1000):
    """Canonical Correlation Analysis (CCA) on two datasets."""
    X = X - np.mean(X, axis=0); Y = Y - np.mean(Y, axis=0)
    Sigma_XX = X.T @ X; Sigma_YY = Y.T @ Y; Sigma_XY = X.T @ Y
    w_x = np.random.rand(X.shape[1]); w_y = np.random.rand(Y.shape[1])
    w_x /= np.linalg.norm(w_x); w_y /= np.linalg.norm(w_y)
    for iteration in range(max_iter):
        w_x_old = w_x.copy(); w_y_old = w_y.copy()
        w_x = np.linalg.solve(Sigma_XX, Sigma_XY @ w_y); w_x /= np.linalg.norm(w_x)
        w_y = np.linalg.solve(Sigma_YY, Sigma_XY.T @ w_x); w_y /= np.linalg.norm(w_y)
        correlation = w_x.T @ Sigma_XY @ w_y
        if np.linalg.norm(w_x - w_x_old) < tol and np.linalg.norm(w_y - w_y_old) < tol:
            break
    return correlation, w_x, w_y

def cca_multi(X, Y, num_components=1, tol=1e-6, max_iter=1000):
    """CCA with multiple canonical variates."""
    X = X - np.mean(X, axis=0); Y = Y - np.mean(Y, axis=0)
    correlations = []; W_X = []; W_Y = []
    for component in range(num_components):
        Sigma_XX = X.T @ X; Sigma_YY = Y.T @ Y; Sigma_XY = X.T @ Y
        w_x = np.random.rand(X.shape[1]); w_y = np.random.rand(Y.shape[1])
        w_x /= np.linalg.norm(w_x); w_y /= np.linalg.norm(w_y)
        for iteration in range(max_iter):
            w_x_old = w_x.copy(); w_y_old = w_y.copy()
            w_x = np.linalg.solve(Sigma_XX, Sigma_XY @ w_y); w_x /= np.linalg.norm(w_x)
            w_y = np.linalg.solve(Sigma_YY, Sigma_XY.T @ w_x); w_y /= np.linalg.norm(w_y)
            correlation = w_x.T @ Sigma_XY @ w_y
            if np.linalg.norm(w_x - w_x_old) < tol and np.linalg.norm(w_y - w_y_old) < tol:
                break
        correlations.append(correlation); W_X.append(w_x); W_Y.append(w_y)
        X -= (X @ w_x[:, np.newaxis]) @ w_x[np.newaxis, :]
        Y -= (Y @ w_y[:, np.newaxis]) @ w_y[np.newaxis, :]
    return {'correlations': np.array(correlations), 'W_X': np.array(W_X).T, 'W_Y': np.array(W_Y).T}