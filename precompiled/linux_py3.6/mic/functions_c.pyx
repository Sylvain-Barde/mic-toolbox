# -*- coding: utf-8 -*-
"""Cythonised functions for the Markov Information Criterion Toolbox

Ancillary C functions for the MIC toolbox and associated classes. In most 
cases, these are implemented as faster replacements to native numpy functions

List of functions:
   - nonzero_sum:  Count number of non-empty rows in an array    
   - sum_overflow: Adds two integers with overflow limit
   - sign_2d:      Returns sign of a 2d int array as {-1,+1,0}
   - array_eq:     Array equality check, true if ALL elements match
   - int_min:      Returns smallest of two integers
   - conv_a2d:     Analog to digital converter (real-value data to binary)
   - conv_d2a:     Digital to analog converter (binary data to real-value)
   - hash_fast:    Fast pearson hash function (from EIDMA RS.97.01)
   - hash_obs:     Trivial hash function for storing observation bits

Created on Thu Sep 29 13:48:16 2016
This version 04/07/2019
@author: Sylvain Barde, University of Kent.
"""

# -- Imports
import numpy as np      # Objects require numpy arrays and functions 
cimport numpy as np     # Cython numpy import
from libc.stdint cimport int32_t # Import numeric types
from cpython cimport bool   # Import boolean

#------------------------------------------------------------------------------
# Fast count of non-empty rows
cdef np.ndarray[np.int32_t,ndim=1] nonzero_sum(np.ndarray[np.uint8_t,ndim=2] a):

    cdef np.ndarray[np.int32_t, ndim=1] non_zero
    cdef np.ndarray[np.int32_t, ndim=2] b
    cdef int R,C,r,c, count
    
    R = a.shape[0]  # Rows

    non_zero = np.zeros([R],dtype=np.int32)  # Initialise full array 
    b = a.astype(np.int32)
    count = 0

    for r in range(R):
        if not sum(b[r,:]) == 0:
            non_zero[count] = r
            count += 1

    non_zero = non_zero[0:count]            # Trim down to number of non-zeros
    
    return non_zero
#------------------------------------------------------------------------------
# Fast overflow protection
cdef int32_t sum_overflow(int32_t a, int32_t b, int32_t cl, int32_t cu):
    
    if a > 0 and b > cu - a:        # a+b will overflow in this case
        return cu
    elif a < 0 and b < cl - a:      # a+b will underflow in this case
        return cl
    else:                           # sum is OK, return it
        return a + b
    
#------------------------------------------------------------------------------
# 2D integer sign check

cdef np.ndarray[np.int32_t, ndim=2] sign_2d(np.ndarray[np.int32_t, ndim=2] a):

    cdef np.ndarray[np.int32_t, ndim=2] b
    cdef int r,c,R,C
    
    R = a.shape[0]  # Rows
    C = a.shape[1]  # cols
    
    b = np.zeros([R,C],dtype=np.int32)  # Initialise full array 

    for r in range(R):
        for c in range(C):
            if a[r,c] > 0:
                b[r,c] = 1
            elif a[r,c] < 0:
                b[r,c] = -1

    return b

#------------------------------------------------------------------------------
# -- Fast array equality check (speedup from np.array_equal)
cdef bool array_eq( uint8_t [:,:] a, uint8_t [:,:] b):

    cdef int L, i
    
    L = a.shape[1]               # ASSUMING a and b have same size!!
    
    for i in range(L):
        if not a[0,i] == b[0,i]: # If an entry doesn't match, different arrays
            return False
    
    return True                  # if out of loop, arrays are same

#------------------------------------------------------------------------------
# -- integer minimum (faster than np.minimum)
cdef int int_min(int a, int b):
    
    if a <= b :
        return a
    else:
        return b

#------------------------------------------------------------------------------
# -- Converters: analog to digital / digital to analog
cdef np.ndarray[np.uint8_t, ndim=2] conv_a2d(np.ndarray[np.float64_t, ndim=1] \
    data, int r, double lb, double ub):
    
    cdef int n, i
    cdef double unit
        
    n = len(data)                           # Number of observations
    code = np.zeros([n,r],dtype=np.uint8)   # Initialise coded output

    unit = (2**float(r))/(ub-lb)            # Coding unit - controls resolution
    data = np.floor((data - lb)*unit)       # Bin data into 2^r integer bins

    
    for i in range(r,0,-1):                 # For all powers of 2 in range
        bit = data >= 2**(i-1)              # Check if data > power of 2
        data = data - bit*2**(i-1)          # Update data if true
        code[:,i-1] = bit                   # Save bit value to code

    return code                             # Return code


cdef np.ndarray[np.float64_t, ndim=1] conv_d2a(np.ndarray[np.uint8_t, ndim=2] \
    code, double lb, double ub):
    
    cdef int n, r, i
    cdef double unit
    cdef np.ndarray[np.float64_t, ndim=1] data

    n = code.shape[0]               # Number of observations
    r = code.shape[1]               # Bits of resolution
    data = np.zeros([n])            # Initialise data vector

    unit = (ub-lb)/(2**float(r))    # coding unit - value of a binary level

    # Match binary string to digitised data levels
    for i in range(r):
        data = data + code[:,i]*2**i

    data = data*unit + lb           # Convert digitised data to correct units
    return data                     # Return data

#------------------------------------------------------------------------------
# -- Fast Pearson hash function (32-bit memory index, modulo memory size)
cdef np.ndarray[np.uint32_t, ndim=1] hash_fast(np.ndarray[np.uint8_t, ndim=2] \
    context_bytes, np.ndarray[np.int32_t, ndim=1] pearson_tab, int mem):

    cdef np.ndarray[np.uint32_t, ndim=2] index_base
    cdef np.ndarray[np.uint32_t, ndim=1] ind
    cdef int r,c,R,C
    R = context_bytes.shape[0]               # number of rows
    C = context_bytes.shape[1]               # number of columns
    
    index_base = np.empty([R,C],dtype = np.uint32) 
    for r in range(R):
        for c in range(C):
            index_base[r,c] = pearson_tab[context_bytes[r,c]]
            
    # uint32 index, modulo memory size        
    ind = np.sum(index_base,1,dtype = np.uint32) % mem    

    return ind
    
#------------------------------------------------------------------------------
# -- Trivial hash function for conditioning on observation bits
cdef np.ndarray[np.uint32_t, ndim=1] hash_obs(np.ndarray[np.uint8_t, ndim=1] \
    obs, int r):
     
    cdef int L
    
    L = len(obs)
    
    if L < r:
        obs = np.concatenate((np.zeros(r-L,dtype = np.int8),  obs))
        
    base = 2**(np.arange(1,r, dtype = np.uint32)*obs[1:r])
    vec = np.concatenate((np.zeros(1,dtype = np.uint32), np.flipud(base)))
    obs_path = np.flipud(np.cumsum(vec,dtype = np.uint32))
    
    return obs_path
    
#------------------------------------------------------------------------------