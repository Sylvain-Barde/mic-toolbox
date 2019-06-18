# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 11:10:48 2018

@author: sb636
"""


import numpy as np      # Objects require numpy arrays and functions 
cimport numpy as np     # Cython numpy import
from libc.stdint cimport uint8_t, uint16_t, uint32_t, int32_t # Integer types
from libc.math cimport floor
from cpython cimport bool # Boolean
cimport functions_c as f  # Custom Cython functions

#------------------------------------------------------------------------------
# Context tree object
cdef class Tree:
    
    # Atttributes:
    cdef public uint8_t[:,:]  node_id, node_org, node_dat
    cdef public uint16_t[:,:] a_count, b_count
    cdef public int32_t[:,:]  log_beta, log_Q
    cdef public int32_t[:]    perm
    cdef public uint32_t      node_fail, rescl, train_obs, mem, t, root
    cdef public uint8_t       d, lags, n_bytes, r, var
    cdef public double[:,:]   bins
    cdef public str           tag
    cdef public list          r_vec
    
    
    # Methods:
    cpdef void desc(self)

    cpdef dict hash_mem(self, np.ndarray[np.uint8_t, ndim=1] context, str mode, 
                   object tab)
                   
    cpdef np.ndarray[np.uint32_t, ndim=1] hash_path(self, 
                    np.ndarray[np.uint8_t, ndim=2] path, 
                    np.ndarray[np.uint8_t, ndim=2] origin, uint8_t d_base, 
                    np.ndarray[np.int32_t, ndim=1] pearson_tab, str mode)
                    
    cpdef double update(self, dict index_struct, 
                    np.ndarray[np.uint8_t, ndim=1] obs, object tab)
                    
    cpdef dict cond_prob(self, dict index_struct, 
                    np.ndarray[np.uint8_t, ndim=1] obs, object tab)
    
    cpdef dict uncond_prob(self, np.ndarray[np.uint8_t, ndim=1] obs,
                           object tab)

    cpdef dict prob_vec(self, dict index_struct, 
                    object tab, str mode)
#------------------------------------------------------------------------------
# Data structure object
cdef class DataStruct:
    
    # Attributes
    cdef public uint8_t[:,:] string
    cdef public double[:,:]  errors
    cdef public list         lb, ub, r
    
    # Methods
    cpdef recover(self)          # data recovery method
#------------------------------------------------------------------------------
# Lookup tables for integer arithmetic
cdef class Tables:
    
    # Attributes
    cdef public int32_t[:]   pearson
    cdef public int32_t[:]   log_tab
    cdef public int32_t[:]   jac_tab
    cdef public uint8_t[:,:] mask
    cdef public uint8_t[:,:] bit_pointer
    cdef public uint8_t[:,:] toggle
    cdef public uint32_t     A, cutoff
    cdef public int32_t      cl, cu
    cdef public uint16_t     reg, half_reg
    cdef public uint8_t      n_bytes
        
    # No methods
#------------------------------------------------------------------------------