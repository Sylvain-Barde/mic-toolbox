# -*- coding: utf-8 -*-
"""Markov Information Criterion Toolbox

Contains the following tools:
    save_tree -- Save a context tree structure as a pickle file
    load_tree -- Load a pickled context tree structure
    bin_quant -- Perform the binary quantisation of a dataset
    corr_perm -- Correlation-based ordering of context bits
    naive_perm -- Naive ordering of context bits
    train -- Train a context tree on model simulations
    score -- Score empirical data using a trained tree
    particle run -- (experimental) simulate particle paths based on the context
        tree 

NOTE: Due to the use of cython the data types used for the function arguments 
must match the  declared types in the functions. See individual function help 
for more details

Created on Mon Jul  4 15:23:26 2016
This version 04/07/2019
@author: Sylvain Barde, University of Kent.
"""

# -- Imports
import numpy as np    # Objects require numpy arrays and functions 
cimport numpy as np   # Cython numpy import
import pickle         # Used for saving/loading trees
import zlib           # Used for compress/decompress saved trees
import time           # Benchmarking

import scipy.stats as scistats
import scipy.sparse as sparse

from libc.stdint cimport uint8_t, uint32_t    # Integer types
from classes cimport Tree, DataStruct, Tables # Classes
from functions_c cimport conv_d2a             # Custom Cython functions

#------------------------------------------------------------------------------
# -- Save tree
def save_tree(Tree T, str loc):
    """Save a context tree structure as a pickled file 
    
    Keyword arguments:
        T -- a context tree structure (see the Tree class in 'classes.pyx' for 
           more detail)
        loc -- string, path to the desired file
        
    Returns None
    """
                     
    # Type declarations for underlying tables
    cdef np.ndarray[np.uint8_t, ndim=2] node_id, node_org, node_dat
    cdef np.ndarray[np.uint16_t, ndim=2] a_count, b_count
    cdef np.ndarray[np.int32_t, ndim=2] log_beta, log_Q
    cdef np.ndarray[np.int32_t, ndim=1] perm
    cdef np.ndarray[np.double_t, ndim=2] bins
    cdef uint32_t node_fail, rescl, train_obs, mem
    cdef uint8_t d, lags, var
    cdef str tag
    cdef list r_vec
    cdef dict save_struct

    # Extract constants and tree information
    mem   = T.mem
    lags  = T.lags
    d     = T.d
    r_vec = T.r_vec
    var   = T.var
    perm  = np.asarray(T.perm)
    tag   = T.tag
    bins  = np.asarray(T.bins)
    
    # Extract tree diagnostics
    node_fail = T.node_fail
    rescl     = T.rescl
    train_obs = T.train_obs
    root      = T.root
        
    # Extract node data arrays
    node_id  = np.asarray(T.node_id)
    node_org = np.asarray(T.node_org)
    node_dat = np.asarray(T.node_dat)
    
    # Extract node counters and ratios
    a_count  = np.asarray(T.a_count)
    b_count  = np.asarray(T.b_count)
    log_beta = np.asarray(T.log_beta)
    log_Q    = np.asarray(T.log_Q)

    # Create temporary saving structure (list)
    save_struct = {'mem'       : mem,
                   'lags'      : lags,
                   'd'         : d,
                   'r_vec'     : r_vec,
                   'var'       : var,
                   'perm'      : perm,
                   'tag'       : tag,
                   'bins'      : bins,
                   'node_fail' : node_fail,
                   'rescl'     : rescl,
                   'train_obs' : train_obs,
                   'root'      : root,
                   'node_id'   : node_id,
                   'node_org'  : node_org,
                   'node_dat'  : node_dat,
                   'a_count'   : a_count,
                   'b_count'   : b_count,
                   'log_beta'  : log_beta,
                   'log_Q'     : log_Q}
    
    # Pickle/compress list to file (uses zlib directly - faster than gzip)
    fil = open(loc,'wb')
    fil.write(zlib.compress(pickle.dumps(save_struct, protocol=2)))
    fil.close()
    
    return

#------------------------------------------------------------------------------
# -- Load tree
def load_tree(str loc):
    """Load a pickled context tree structure 
    
    Keyword arguments:
        loc -- string, path to the desired file
            
    Returns:
        T -- a context tree structure (see the Tree class in 'classes.pyx' for 
           more detail)
    """

    cdef dict recov_struct

    # Decompress/unpickle list to file (uses zlib directly - faster than gzip)
    fil = open(loc,'rb')
    datas = zlib.decompress(fil.read(),0)
    fil.close()
    
    recov_struct = pickle.loads(datas)

    # Extract tree initialisation data
    tag   = recov_struct['tag']
    mem   = recov_struct['mem']
    lags  = recov_struct['lags']
    d     = np.double(recov_struct['d'])
    r_vec = recov_struct['r_vec']
    var   = recov_struct['var']
    perm  = recov_struct['perm']

    # Initialise an empty tree with correct fields
    T = Tree(tag,mem,lags,d,r_vec,var,perm)

    # Populate tree properties from recovered data

    # -> Support for the discretisation
    T.bins      = recov_struct['bins']

    # -> Tree diagnostics
    T.node_fail = recov_struct['node_fail']
    T.rescl     = recov_struct['rescl']
    T.train_obs = recov_struct['train_obs']
    T.root      = recov_struct['root']
    
    # -> Hashing/node ID data
    T.node_id   = recov_struct['node_id']
    T.node_org  = recov_struct['node_org']
    T.node_dat  = recov_struct['node_dat']
    
    # -> Internal counters / ratio attributes
    T.a_count  = recov_struct['a_count']
    T.b_count  = recov_struct['b_count']
    T.log_beta = recov_struct['log_beta']
    T.log_Q    = recov_struct['log_Q']

    return T

#------------------------------------------------------------------------------
# -- Binary quantisation
def bin_quant(np.ndarray[np.float64_t, ndim=2] data_vec, 
                 list lb_vec, list ub_vec, list r_vec, str flag):
    """Perform the binary quantisation of a dataset
    
    Keyword arguments:
        data_vec -- a N-by-K 2d numpy float 64 array containing the data, where
            N is the number of oservations and K is the number of variables.
        lb_vec -- a python list of K lower bounds.
        ub_vec -- a python list of K upper bounds.
        r_vec -- a python list of discretisation resolutions, in bits.
        flag -- a string controlling discretisation test options. The following
            options are available:
            --> 'notests': no discretisation tests are run
            --> 'nodislay': discretisations tests are run, but not displayed.
            --> Pass any other string to run & display tests
        
    Returns a dictionary with the following keys:
        'binary_data' -- a binary data structure (see the DataStruct class in 
                       'classes.pyx' for more detail)
        'diagnostics' -- a dictionary containing the diagnostic test results
            If the 'notests' option is selected, the dictionary is simply:
                'message' : 'No quantisation tests carried out'
            Otherwise, the diagnotics dictionary has the following keys:
                'out_of_bounds' : Number of observations out of bounds
                'min_vec' : Value of smallest observation
                'max_vec' : Value of largest observation
                'KS_stat' : Kolomogorov Smirnov test statistics 
                            H0: discretisation errors are uniformly distributed
                'KS_pval' : p-value for Kolomogorov Smirnov tests
                'LB_stat' : Ljung Box test statistics
                            H0: discretisation errors are uncorrelated            
                'LB_pval' : p-value for Ljung Box tests
                'SP_stat' : Spearman correlation statistic
                            H0: discretisation errors uncorrelated with levels  
                'SP_pval' : p-value for significance of spearman correlation
                'std_dat' : standard deviation of the data
                'd_units' : discretisation units
                'snr_emp' : empirical signal to noise ratio (SNR)
                'snr_the' : theoretical SNR (assuming gaussian data)
    """
    
    cdef np.ndarray[np.float64_t, ndim=2] err_vec
    cdef np.ndarray[np.float64_t, ndim=1] u_vec, err, sig, data_int, data
    cdef int i, num_vars, K, pad1, pad2, width
    cdef double n
    cdef list out_of_bounds, min_vec, max_vec, KS_stat, KS_pval, LB_stat, var_str 
    cdef list LB_pval, SP_stat, SP_pval, std_dat, d_units, snr_emp, snr_the
    cdef str divider, header, flt_str, int_str
    cdef dict diagnostics, output    

    # -- Descriptive statistics on min, max, etc if tests are required
    if not flag == 'notests':

        num_vars = len(r_vec)               # number of variables
        out_of_bounds = []
        min_vec = []
        max_vec = []
        
        for i in range(num_vars):    
            data = data_vec[:,i]
            min_vec.append(min(data))
            max_vec.append(max(data))
            ind_ub = (data > ub_vec[i]).nonzero()
            ind_lb = (data < lb_vec[i]).nonzero()
            np.put(data,ind_ub,ub_vec[i])
            np.put(data,ind_lb,lb_vec[i])
            out_of_bounds.append(len(ind_lb[0]) + len(ind_ub[0]))
     
    # -- Quantise data by creating an instance of DataStruct
    data_struct = DataStruct(data_vec, lb_vec, ub_vec, r_vec)
    
    # -- If no tests required, pass empty diagnostic, else, run tests
    if flag == 'notests':
        
        # - Prepare empty diagnostics message
        diagnostics = dict(message = 'No quantisation tests carried out')
        
    else:
    
        err_vec = np.asarray(data_struct.errors)
        
        n = err_vec.shape[0]                # number of observations
        u_vec = np.arange(0,1,1/n)          # Uniform vector for KStest2
        K = np.uint8(np.ceil(np.log(n)))    # degrees of freedom -> chi2
        k = np.arange(1,K+1)
    
        KS_stat = []
        KS_pval = []
        LB_stat = []
        LB_pval = []
        SP_stat = []
        SP_pval = []
        std_dat = []
        d_units = []
        snr_emp = []
        snr_the = []
        var_str = []
        for i in range(num_vars):
    
            data = data_vec[:,i]
            err = err_vec[:,i]
            x = err.flatten()
            x2 = (x - np.mean(x))/(np.var(x)**0.5)
            
            ks_test = scistats.ks_2samp(x,u_vec)
            KS_stat.append(ks_test[0])
            KS_pval.append(ks_test[1])
    
            rho = np.correlate(x2,x2,mode='full')[len(x)-1:]/n
            LB_stat.append(n*(n+2)*sum((rho[k]**2)/(n - k)))
            LB_pval.append(1 - float(scistats.chi2.cdf(LB_stat[i],K)))
    
            unit = (ub_vec[i]-lb_vec[i])/(2**r_vec[i])
            data_int = data - err*unit
            sp_corr = scistats.spearmanr(err.flatten(),data_int.flatten())
            SP_stat.append(sp_corr[0])
            SP_pval.append(sp_corr[1])
            
            sig = data/unit
            std_dat.append(np.std(data))
            snr_emp.append(10*np.log10(sum(sig**2)/sum(x**2)))
            snr_the.append(r_vec[i]*20*np.log10(2) + 10*np.log10(12)
                + 20*np.log10(np.std(data)/(ub_vec[i]-lb_vec[i])))
            
            var_str.append(i+1)
            d_units.append(unit)

        # - package diagnostic tests (dict)
        diagnostics = dict(out_of_bounds = out_of_bounds,
                           min_vec = min_vec,
                           max_vec = max_vec,
                           KS_stat = KS_stat,
                           KS_pval = KS_pval,
                           LB_stat = LB_stat,
                           LB_pval = LB_pval,
                           SP_stat = SP_stat,
                           SP_pval = SP_pval,
                           std_dat = std_dat,
                           d_units = d_units,
                           snr_emp = snr_emp,
                           snr_the = snr_the)
       
        # - Display diagnostic tests, if requested
        if not flag == 'nodisplay':
            
            # - Format header and strings
            width = max(51, 16 + num_vars*8)
            divider = '+' + '-'*(width-2) + '+'
            pad1 = int(np.floor((width - 26)/2))
            pad2 = width - 26 - pad1
            header = '|' + ' '*pad1 + 'Quantization diagnostics' \
                    + ' '*pad2 + '|'
            int_str = ' {:>7d}'*num_vars
            flt_str = ' {:7.3f}'*num_vars
            
            print(divider)
            print(header)
            print(divider)
            print(' N° of observations: {:>10d}'.format(int(n)))
        
            print('\n Var N°:      ' + int_str.format(*var_str))
            print(' Lower bound:   ' + flt_str.format(*lb_vec))
            print(' Upper bound:   ' + flt_str.format(*ub_vec))
            print(' Min obs.:      ' + flt_str.format(*min_vec))
            print(' Max obs.:      ' + flt_str.format(*max_vec))
            print(' Out of bounds: ' + int_str.format(*out_of_bounds))    
            print(' Resolution:    ' + int_str.format(*r_vec))
            print(' Standard dev.: ' + flt_str.format(*std_dat))
            print(' Discr. unit:   ' + flt_str.format(*d_units))    
            print('\n Signal-to-Noise ratio for quantisation (in dB):')
            print(' Var N°:      ' + int_str.format(*var_str))   
            print(' Theoretical:   ' + flt_str.format(*snr_the))
            print(' Effective:     ' + flt_str.format(*snr_emp))    
            print('\n' + divider)
            print(' Kolmogorov-Smirnov test for uniformity')
            print(' H0: Distribution of errors is uniform')
            print('\n Var N°:      ' + int_str.format(*var_str))
            print(' KS statistic:  ' + flt_str.format(*KS_stat))
            print(' P-value:       ' + flt_str.format(*KS_pval))    
            print('\n' + divider)
            print(' Ljung-Box tests for autocorrelation of errors')
            print(' H0: The errors are independently distributed')
            print('\n Critical value (Chi-sq, {:d} df):   {:10.3f}'.format(
                    K,scistats.chi2.ppf(0.95,K)))
            print('\n Var N°:      ' + int_str.format(*var_str))
            print(' LB statistic:  ' + flt_str.format(*LB_stat))
            print(' P-value:       ' + flt_str.format(*LB_pval))    
            print('\n' + divider)
            print(' Spearman correlation of error with digitised data')
            print(' H0: Quantisation errors not correlated with data')
            print('\n Var N°:      ' + int_str.format(*var_str))
            print(' Correlation:   ' + flt_str.format(*SP_stat))
            print(' P-value:       ' + flt_str.format(*SP_pval))    
            print('\n' + divider)
                
    # - Return outputs (dict)    
    output = {'binary_data' : data_struct,
              'diagnostics' : diagnostics}
    
    return output    

#------------------------------------------------------------------------------
# Correlation-based permutation
def corr_perm(
        np.ndarray[np.float64_t, ndim=2] data_vec, list r_vec, list hp_bit_vec,
        list var_vec, int lags, int d):
    """Correlation-based ordering of context bits
    
    Order the bits of the context string in order of decreasing correlation of
    the variables/lags with the variable of interest while prioritising the 
    more informative bits.
    
    Keyword arguments:
        data_vec -- a N-by-K 2d numpy float 64 array containing the data, where
            N is the number of oservations and K is the number of variables.
        r_vec -- a python list, number of high priority bits per variable
        hp_bit_vec -- a python list, number of high priority bits per variable
        var_vec -- a python list containing a conditioning order for the
            entropy decomposition. First entry in the list is the variable to 
            be predicted, Length can be less than K to reflect partial 
            conditioning. Examples (assume K=3):
            [3,1,2] -> predict variable 3 conditional on L lags of the K 
                   variables and contemporaneous realisations of variables 
                   1 and 2.
            [3,2] -> predict variable 3 conditional on L lags of the K 
                   variables and contemporaneous realisations of variable 2.
            [3] -> predict variable 3 conditional on L lags of the K variables
                   only.
        lags -- integer, maximumm number of lags in the markov process
        d -- integer, desired length of the context string in bits
        
    Returns a d-by-1 1D numpy array of int32 containing the permutation
    """
    
    cdef np.ndarray[np.float64_t, ndim=2] data_re, x, Ldata, corr_lag
    cdef np.ndarray[np.float64_t, ndim=1] y, corr_cur, v
    cdef np.ndarray[np.int32_t, ndim=1] r_end, r_start, var_lag, perm
    cdef np.ndarray[np.int32_t, ndim=1] lp_bits, hp_bits, perm_full
    cdef np.ndarray[np.uint8_t, ndim=1] var_vec_np, var_curr
    cdef int T, N, r_full, var, i, j, k, count, lag, pad, cut
    cdef float corr_curr_max, corr_lag_max, corr_lag_max_try
    cdef tuple corr
    
    T = data_vec.shape[0]  # Number of observations
    N = data_vec.shape[1]  # Number of variables
    
    r_end = np.int32(np.cumsum(r_vec)-1)    # End markers for variables
    r_start = r_end - np.int32(r_vec)+1     # Start markers for variables
    r_full = np.sum(r_vec)                  # Full resolution
    
    # -- Identify predicted and conditioning variables
    var_vec_np = np.asarray(var_vec, dtype = np.uint8) - 1
    var = var_vec_np[0]
    var_cur = np.setdiff1d(var_vec_np,var)
    var_lag = np.arange(N, dtype = np.int32)
    
    data_re = data_vec[:,var_vec_np]
    y = data_re[:,0]
    x = data_re[:,1:len(var_vec)]
    
    # -- Get absolute correlations with conditioning variables
    corr_cur = np.zeros(len(var_cur))
    for i in range(len(var_cur)):
        corr = scistats.pearsonr(y,x[:,i])
        corr_cur[i] = abs(corr[0])

    if len(var_cur) == 0:
        corr_cur = np.zeros(1)
    
    # -- create lag operator and get lagged correlations
    v = np.ones(T)
    L = sparse.dia_matrix((v, -1), shape=(T, T))
    Ldata = data_vec
    corr_lag= np.zeros([lags,N])
    for i in range(lags):
        Ldata = L*Ldata
        for j in range(N):
            corr = scistats.pearsonr(y,Ldata[:,j])
            corr_lag[i,j] = abs(corr[0])
    
    # Initialise display
    divider = '+' + '-'*54 + '+'
    print('\n' + divider)
    print(' Predicting variable: {:>3d}'.format(var_vec[0]))
    print(' Contemporaneous conditioning vars: ')
    print('\t' + str(var_cur+1))
    print(' Number of conditioning lags: {:>3d}'.format(lags))
    print('\n')
    
    lp_bits = np.asarray([], dtype = np.int32)
    hp_bits = np.asarray([], dtype = np.int32)
    
    count = 1
    while max(max(corr_cur),max(corr_lag.flatten())) > 0:
        
        corr_cur_max = max(corr_cur)
        k = np.argmax(corr_cur)
        
        corr_lag_max = 0
        i = 0
        for lag in range(lags):
            corr_lag_max_try = max(corr_lag[lag,:])
            if corr_lag_max_try > corr_lag_max:
                corr_lag_max = corr_lag_max_try
                i = lag
                j = np.argmax(corr_lag[lag,:])
    
        print(' --------- Iteration {:>3d} ---------'.format(count))
        if corr_cur_max > corr_lag_max:
            print(' Select contemporaneous var {:>3d}'.format(var_cur[k]+1))
            print(' Correlation: {:7.4f}'.format(corr_cur_max))

            pad = lags*r_full
            cut = r_end[var_cur[k]]- hp_bit_vec[var_cur[k]] + pad
            
            lp_bits = np.hstack((
                    np.arange(r_start[var_cur[k]]+pad, cut+1,
                              dtype = np.int32),
                    lp_bits
                    ))
            hp_bits = np.hstack((
                    np.arange(cut+1, r_end[var_cur[k]]+pad+1, 
                              dtype = np.int32),
                    hp_bits
                    ))
            
            corr_cur[k] = 0
            
        else:
            print(' Select lag {:>3d} of var {:>3d}'.format(i+1,var_lag[j]+1))
            print(' Correlation: {:7.4f}'.format(corr_lag_max))

            pad = i*r_full
            cut = r_end[var_lag[j]]- hp_bit_vec[var_lag[j]] + pad
            
            lp_bits = np.hstack((
                    np.arange(r_start[var_lag[j]]+pad, cut+1, 
                              dtype = np.int32),
                    lp_bits
                    ))
            hp_bits = np.hstack((
                    np.arange(cut+1, r_end[var_lag[j]]+pad+1, 
                              dtype = np.int32),
                    hp_bits
                    ))            

            corr_lag[i,j] = 0

        count += 1
        
    print(divider + '\n')
    perm_full = np.hstack((lp_bits, hp_bits))
    cut = len(perm_full) - d
    perm = perm_full[cut:len(perm_full)]

    return perm

#------------------------------------------------------------------------------
# Simple rearrangement-based permutation
def naive_perm(list r_vec, list hp_bit_vec, list var_vec, int lags, int d):
    """Naive ordering of context bits
    
    Order the bits of the context string using the initial order of variables 
    in the data while prioritising the more informative bits.
    
    Keyword arguments:
        r_vec -- a python list of discretisation resolutions, in bits.
        hp_bit_vec -- a python list, number of high priority bits per variable
        var_vec -- a python list containing a conditioning order for the
            entropy decomposition. First entry in the list is the variable to 
            be predicted, Length can be less than K to reflect partial 
            conditioning. Examples (assume K=3):
            [3,1,2] -> predict variable 3 conditional on L lags of the K 
                   variables and contemporaneous realisations of variables 
                   1 and 2.
            [3,2] -> predict variable 3 conditional on L lags of the K 
                   variables and contemporaneous realisations of variable 2.
            [3] -> predict variable 3 conditional on L lags of the K variables
                   only.
        lags -- integer, maximumm number of lags in the markov process
        d -- integer, desired length of the context string in bits
        
    Returns a d-by-1 1D numpy array of int32 containing the permutation
    """

    cdef np.ndarray[np.int32_t, ndim=2] context_base, perm_base_2d
    cdef np.ndarray[np.int32_t, ndim=2] perm_base_2d_l, perm_base_2d_h
    cdef np.ndarray[np.int32_t, ndim=1] r_end, r_start, perm, lp_bit_vec
    cdef np.ndarray[np.int32_t, ndim=1] perm_vec, perm_base
    cdef np.ndarray[np.int32_t, ndim=1] perm_base_l, perm_base_h
    cdef np.ndarray[np.uint8_t, ndim=1] var_vec_np, var_cond, var_perms
    cdef int num_vars, r_full, lp_bits, lp_count, hp_count, var_cond_bits, i
    cdef int d_full
    
    num_vars = len(r_vec)                   # Number of variables
    r_end = np.int32(np.cumsum(r_vec)-1)    # End markers for variables
    r_start = r_end - np.int32(r_vec)+1     # Start markers for variables
    r_full = np.sum(r_vec)                  # Full resolution
    
    # -- Identify predicted and conditioning variables
    var_vec_np = np.asarray(var_vec, dtype = np.uint8)
    var = var_vec_np[0]
    var_cond = np.setdiff1d(var_vec_np,var)
    
    # -- Calculate number of low/high priority bits (lp/hp)
    lp_bit_vec = np.int32(r_vec) - np.int32(hp_bit_vec)   # lp bit vector
    lp_bits = sum(lp_bit_vec)                             # Number of lp bits
    
    # -- Initialise full context  
    var_perms = np.zeros([num_vars],dtype=np.uint8) 
    var_perms[0:num_vars-1] = np.setdiff1d(np.arange(1,num_vars+1),var)
    var_perms[num_vars-1] = var
    context_base = np.reshape(
            np.arange(0,r_full*(lags+1), dtype = np.int32),
            [lags+1, r_full]
            ) + 1
    
    # -- Initialise empty permutations
    perm_base_2d = np.zeros([lags+1, r_full],dtype=np.int32)
    perm_vec = np.zeros([r_full],dtype=np.int32)
    
    # -- Iterate over variables
    lp_count = 0                       # Low priority start point
    hp_count = lp_bits                 # High priority start point
    var_cond_bits = 0                  # Conditioning bit count
    
    for i in range(num_vars):
    
        # Populate perm_base, allowing for contemporaneous conditioning
        if i in var_cond-1:       # If var. is conditioned on contemporaneously
            perm_base_2d[:,r_start[i]:r_end[i]+1] = \
                        context_base[:,r_start[i]:r_end[i]+1]
            
            var_cond_bits = var_cond_bits + r_vec[i]   # conditioning bit count
        
        else:
            perm_base_2d[1:lags+1,r_start[i]:r_end[i]+1] = \
                        context_base[0:lags,r_start[i]:r_end[i]+1] 

        # Populate column permutation vector
    
        # Place the variable's low priority bits
        perm_vec[lp_count:lp_count+lp_bit_vec[var_perms[i]-1]] = \
            np.arange(r_start[var_perms[i]-1],r_start[var_perms[i]-1] + \
            lp_bit_vec[var_perms[i]-1])
    
        # Place the variable's high priority bits
        perm_vec[hp_count:hp_count+hp_bit_vec[var_perms[i]-1]] = \
            np.arange(r_end[var_perms[i]-1] - hp_bit_vec[var_perms[i]-1]+1,\
            r_end[var_perms[i]-1]+1)
        
        lp_count = lp_count + lp_bit_vec[var_perms[i]-1]
        hp_count = hp_count + hp_bit_vec[var_perms[i]-1]
        
    # -- Rearrange low/high priority bits according to column permutation
    d_full = r_full*lags + var_cond_bits        # Tot. N° of conditioning bits
    perm_base_2d = perm_base_2d[:,perm_vec]     # Permute columns
    
    perm_base_2d_l = perm_base_2d[:,0:lp_bits]          # Low priority section
    perm_base_2d_h = perm_base_2d[:,lp_bits:r_full]     # High priority section

    # -- Flatten and concatenate low priority bits behind high priority
    perm_base_l = perm_base_2d_l.flatten()
    perm_base_h = perm_base_2d_h.flatten()
    
    perm_base_l = perm_base_l[np.nonzero(perm_base_l)]-1 # Remove 0s and adjust
    perm_base_h = perm_base_h[np.nonzero(perm_base_h)]-1 # Remove 0s and adjust

    perm_base = np.zeros([d_full], dtype = np.int32)
    perm_base[0:len(perm_base_l)] = perm_base_l
    perm_base[len(perm_base_l):d_full] = perm_base_h

    # -- Trim permutation vector down to selected tree depth and return
    perm = perm_base[d_full-d:d_full]

    return perm
    
#------------------------------------------------------------------------------
# -- Stage 1 training on simulated data
def train(object T, DataStruct data_struct, int mem, int lags, double d,  
          int var, str tag, np.ndarray[np.int32_t, ndim=1]  perm):
    """Train a context tree on model simulations
    
    First pass of the two-pass MIC protocol, learning model transition
    probabilities from the simulated data.
    
    Keyword arguments:
        T -- Two possible inputs are accepted:
            --> None -- This will initialise an empty tree
            --> an existing context tree structure (see the Tree class in  
              'classes.pyx' for more detail). This tree will be updated with 
              the training data. In this case, the last 6 argument values
              are ignored.
        data_struct -- a binary data structure (see the DataStruct class in 
                       'classes.pyx' for more detail)
        mem -- integer, maximum number of tree nodes (memory cap)
        lags -- integer, maximumm number of lags in the markov process
        d -- integer, desired length of the context string in bits
        var -- integer, indentifies which variable to predict
        tag -- string for tagging the tree (38 char max, displayed in logs)
        perm -- 1D numpy array of int32 containing a context permutation
        
    Returns a dictionary with the following keys:
        'T' -- A context tree structure, updated with the training data.
        'times' -- a dictionary containing diagnostic timers, with keys:
            'hash'    : Amount of time spent hashing contexts
            'update'  : Amount if time spent updating tree nodes
            'tot'     : Total time taken
            'percent' : Breakdown of elapsed time at each decile of training 
                        data
    """
    
    # Type declarations
    cdef np.ndarray[np.float64_t, ndim=1] time_percent, bins
    cdef np.ndarray[np.uint8_t, ndim=2] context_full, states
    cdef np.ndarray[np.uint8_t, ndim=1] context
    cdef double time_hash, time_tot, t_start, hash_start
    cdef int N, N_count, p_count, head_loc, N_crit
    cdef dict index_struct, times, output
    cdef str state
        
    # -- Initialise Display and tree
    divider = '+' + '-'*54 + '+'
    print('\n' + divider)
    print('|       Context Tree Weighting on training series      |')
    print(divider)
    # -- Initialise or load the tree structure
    if not isinstance(T, Tree):       # Empty object - initialise a full tree
    
        T = Tree(tag,mem,lags,d,data_struct.r,var,perm)
        print('           Empty tree structure initialised')
        
    else:               # Reuse existing tree
    
        pad = int(np.floor((38 - len(T.tag))/2))
        print(' '*pad + 'Using "' + T.tag + '" structure')

    print(divider)

    # -- Initialise lookup tables, parameters, mask and pointer variables. 
    tab = Tables(T.d)               # Tables and parameters required
    
    # -- Get data string and determine number of training observations
    data_string = data_struct.string
    N = data_string.shape[0]               # Number of observations
    N_crit = int(np.floor((N-T.lags)/10))  # Critical threshold (display)

    r_vec = data_struct.r           # Resolution vector of observations
    r_end = np.cumsum(r_vec)        # End markers for variables
    r_start = r_end - r_vec         # Start markers for variables
    r0 = r_start[var-1]             # Start location of variable
    r1 = r_end[var- 1]              # End location of variable

    # -- Initialise CTW algorithm
    N_count      = 0               # Initialise iterations counter (display)
    p_count      = 0               # Initialise percent counter    (display)
    time_hash    = 0               # Initialise hashing time 
    time_update  = 0               # Initialise update time 
    time_tot     = 0               # Initialise total time
    time_percent = np.zeros([10])  # Initialise percent completion time
    t_start      = time.clock()    # Start overall timer

    # -- Check if root index is unallocated (tree is untrained)
    if T.root == mem:
        
        # Allocate index of tree root
        root_struct = T.hash_mem(np.zeros([T.d],dtype=np.uint8),'sp',tab)
        root_vec    = root_struct['index']
        T.root      = root_vec[-1]
        
        # Enumerate possible states for the variable
        states = np.zeros([2**T.r,T.r],dtype = np.uint8)
        for i in range(2**T.r):
            state = np.binary_repr(i,width = T.r)
            states[i] = np.array(list(state),dtype=np.uint8)
        states = np.fliplr(states)

        # Generate variable support vector from states
        bins = conv_d2a(states, data_struct.lb[T.var - 1],
                        data_struct.ub[T.var - 1])
        T.bins = np.expand_dims(bins, axis=1)
        
    # -- Run CTW algorithm on available training observations
    for head_loc in range(T.lags,N):
    
        # Get context and observation from data string
        context_full = np.asarray(data_string[head_loc - T.lags:head_loc+1,:])
        obs = np.asarray(data_string[head_loc,r0:r1])
        
        # Process context bits according to permutation
        context = context_full.flatten()
        context = context[T.perm]
        
        # Hash the path and offshoot for the context
        hash_start = time.clock()
        index_struct = T.hash_mem(context,'sp',tab)
        time_hash += time.clock() - hash_start
        
        # Update tree with observations - Includes CTM updating on-the-fly
        update_start = time.clock()
        T.update(index_struct,obs,tab)
        time_update += time.clock() - update_start
        
        # Check if a % threshold is passed and display progess if TRUE
        N_count += 1                 # Update threshold count    
        if N_count > N_crit-1:       # Check against percentile value
            time_percent[p_count] = time.clock() - t_start;
            print('\t{:3d}% complete\t'.format(10*(p_count+1))  +
                  '{:10.4f} seconds'.format(time_percent[p_count]))
            N_count = 0
            p_count += 1

    # -- After execution, show run time diagnostics
    av_hash = time_hash/head_loc
    av_update = time_update/head_loc
    print(divider)
    print(' Total hashing time required      : {:10.4f}'.format(time_hash))
    print(' Average per context              : {:10.4e}'.format(av_hash))
    print(' Total CTW updating time required : {:10.4f}'.format(time_update))
    print(' Average per context              : {:10.4e}'.format(av_update))
    print(divider)
    time_tot = time.clock() - t_start   # Save total time
    
    # -- Package outputs as dicts
    times = {'hash'    : time_hash,
             'update'  : time_update,
             'tot'     : time_tot,
             'percent' : time_percent}
    
    output = {'T'     : T,
              'times' : times}
    
    return output
    
#------------------------------------------------------------------------------
# -- Stage 2 scoring on empirical data
def score(Tree T, DataStruct  data_struct):
    """Score empirical data using probabilities from a context tree
    
    Second pass of the two-pass MIC protocol, scoring the empricial transitions
    using model probabilities extracted from the tree.
    
    Keyword arguments:
        T -- an existing context tree structure (see the Tree class in  
              'classes.pyx' for more detail). This tree will be updated with 
              the training data.
        data_struct -- a binary data structure (see the DataStruct class in 
                       'classes.pyx' for more detail).
        
    Returns a dictionary with the following keys:
        'score': numpy array containing the bit length of N-lags observations,
        'bound_corr' : numpy array containing the corresponding expected bias
        'trans_tab' : provides node indices and (a,b) count data for all the 
                transitions in the empirical data (for debugging purposes)
    """
    
    # Type declarations
    cdef np.ndarray[np.uint8_t, ndim=2] context_full
    cdef np.ndarray[np.uint32_t, ndim=2] counts, trans_id
    cdef np.ndarray[np.uint32_t, ndim=3] trans_tab
    cdef np.ndarray[np.uint8_t, ndim=1] context
    cdef np.ndarray[np.double_t, ndim=1] obs_score, bound_corr, P, n, n_0
    cdef int N, head_loc, i
    cdef dict output
            

    # -- Initialise lookup tables, parameters, mask and pointer variables. 
    tab = Tables(T.d)               # Tables and parameters required
    
    # -- Get data string and determine number of training observations
    data_string = data_struct.string
    N = data_string.shape[0]        # Number of observations

    r_end = np.cumsum(T.r_vec)      # End markers for variables
    r_start = r_end - T.r_vec       # Start markers for variables
    r0 = r_start[T.var-1]           # Start location of variable
    r1 = r_end[T.var- 1]            # End location of variable

    # -- Initialise outputs 
    obs_score  = np.zeros([N-T.lags])       # bitrate vector
    bound_corr = np.zeros([N-T.lags])       # bound correction vec
    trans_tab  = np.zeros([N-T.lags,5,T.r], dtype = np.uint32) # transitions
    
    # -- Calculate scores and count transitions
    i = 0                               # Initialise symbol counter
    for head_loc in range(T.lags,N):    # -- Process observations
    
        # Get context and observation from data string
        context_full = np.asarray(data_string[head_loc - T.lags:head_loc+1,:])
        obs = np.asarray(data_string[head_loc,r0:r1])    # Observation bits
        
        # Process context bits according to permutation
        context = context_full.flatten()
        context = context[T.perm]
        
        # -- Hash the path and offshoot for the context
        index_struct = T.hash_mem(context,'g',tab)
        prob_struct  = T.cond_prob(index_struct, obs, tab)
        P        = prob_struct['P'] 
        counts   = prob_struct['counts']
        trans_id = prob_struct['trans_id']
        
        # -- Calculate marginal Rissanen bound on observation
        n = np.double(sum(counts))
        n_0 = np.ones([len(n)])
        n_0[n==0] = 0
        n[n==0] = 1
        bound_corr[i] = sum(0.5*n_0*(np.log2(n+1) - np.log2(n)))
        
        # -- Calculate the log score for the observation 
        for j in range(T.r-1,-1,-1):
            if obs[j] == 1:
                obs_score[i] += - np.log2(P[j])
            else:
                obs_score[i] += - np.log2(1-P[j])
                
            # Count transitions
            trans_tab[i,0,j] = trans_id[j,0]
            trans_tab[i,1,j] = trans_id[j,1]
            trans_tab[i,2,j] = counts[0,j]
            trans_tab[i,3,j] = counts[1,j]
            trans_tab[i,4,j] = obs[j]
            
        # Update counter before moving on
        i += 1
        
    # -- Package outputs (dict)
    output = {'score'      : obs_score,
              'bound_corr' : bound_corr,
              'trans_tab'  : trans_tab}
                   
    return output

#------------------------------------------------------------------------------
# -- Particle run from Context Tree probabilities
def particle_run(list tree_list, list lb, list ub, list r_vec, 
                 np.ndarray[np.float64_t, ndim=3] paths,
                 np.ndarray[np.float64_t, ndim=3] draw):
    """(experimental) simulate particle paths based on the context tree 
    
    Second pass of the two-pass MIC protocol, scoring the empricial transitions
    using model probabilities extracted from the tree.
    
    Keyword arguments:
        tree_list -- a python list of K context tree structure
        lb -- a python list of K lower bounds.
        lb -- a python list of K upper bounds.
        paths -- a T-by-N-by-K 3d numpy float 64 array containing the 
            intitialisation of the particle paths. T is the number of time 
            periods to simulate, N the number of particles and K the number of 
            variables. To initialise set lags (the first L time periods) of all
            N-by-K arrays equal to the desirded intitial condintions. The 
            remaining T-L time periods can be 0-valued, as they will be 
            overwritten.
        draw -- a T-by-N-by-K 3d numpy float 64 array containing draws from the
            [0,1] uniform distrubtion (provided externally for replication 
            purposes, can be done in-function in future using a provided RNG 
            seed)
            
        
    Returns a dictionary with the following keys:
        'paths': a T-by-N-by-K 3d numpy float 64 array containing the populated
            particle paths.
        't' : double, containing the time elapsed.
    """   
    
    # Type declarations
    cdef np.ndarray[np.float64_t, ndim=1] prob_vec, cdf_vec
    cdef np.ndarray[np.float64_t, ndim=2] bins
    cdef np.ndarray[np.uint8_t, ndim=2] context_base
    cdef np.ndarray[np.uint8_t, ndim=1] context
    cdef double tot_start, t_start, time_t, time_tot
    cdef int Nt, Np, lags, t, i, p, ind
    cdef object path, T, tab
    cdef dict index_struct, prob_struct, path_struct
    
    # -- Get some constants from the passed arrays
    Nt = draw.shape[0]
    Np = draw.shape[2]    
    lags = paths.shape[0] - Nt
    
    tot_start = time.clock()          # Start overall timer
    
    for t in range(Nt):
        t_start = time.clock()          # Start overall timer
    
        for (i, T) in enumerate(tree_list):
            
            # -- Initialise lookup tables and bins for the tree
            bins = np.asarray(T.bins)       # Extract bins    
            tab = Tables(T.d)               # Tables and parameters required
            
            if t == 0:  # On 1st step, all particles face the same probs.
            
                # -- Discretise path, get context
                path = DataStruct(paths[0:lags,:,0],lb,ub,r_vec)
                context_base = np.asarray(path.string)
                context = context_base.flatten()
                context = context[T.perm]
                
                # -- Hash the context path, get conditional probability vector
                index_struct = T.hash_mem(context,'g',tab)
                prob_struct  = T.prob_vec(index_struct,tab,'c')
                prob_vec     = prob_struct['P_vec']
                
                # -- Determine realisation of next outcome for all particles
                cdf_vec = np.insert(np.cumsum(prob_vec), 0, 0)
                for p in range(Np):
                    ind = np.where(draw[t,i,p] > cdf_vec)[0][-1]
                    paths[t+lags,i,p] = bins[ind] + 1e-6
                
            else:   # from then on, different probabilities
                    # Can probably'remember' some of the past ones, but not yet
                    
                for p in range(Np):
                    
                    # -- Discretise path, get context
                    path = DataStruct(paths[t:t+lags,:,p],lb,ub,r_vec)
                    context_base = np.asarray(path.string)
                    context = context_base.flatten()
                    context = context[T.perm]
                
                    # -- Hash the context path, get conditional prob. vector
                    index_struct = T.hash_mem(context,'g',tab)
                    prob_struct  = T.prob_vec(index_struct,tab,'c')
                    prob_vec     = prob_struct['P_vec']
                
                    # -- Get realisation of next outcome for all particles
                    cdf_vec = np.insert(np.cumsum(prob_vec), 0, 0)            
                    ind = np.where(draw[t,i,p] > cdf_vec)[0][-1]
                    paths[t+lags,i,p] = bins[ind] + 1e-6
    
    
        time_t = time.clock() - t_start   # Save iteration time
        print(' Iter: {:4d}\tTime required : {:10.4f}'.format(t,time_t))  
    
    
    time_tot = time.clock() - tot_start   # Save total time
    paths_struct = {'paths' : paths,
                    't' : time_tot}
    
    return paths_struct 

#------------------------------------------------------------------------------