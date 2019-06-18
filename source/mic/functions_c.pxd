# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 10:26:34 2016

@author: sb636
"""

cimport cython
import numpy as np      # Objects require numpy arrays and functions 
cimport numpy as np     # Cython numpy import
from libc.stdint cimport int32_t, uint8_t # Import numeric types
from cpython cimport bool   # Import boolean

# Declare C functions for use in the package classes
cdef np.ndarray[np.int32_t, ndim=1] nonzero_sum(np.ndarray[np.uint8_t, ndim=2] a)

cdef int32_t sum_overflow(int32_t a, int32_t b, int32_t cl, int32_t cu)

cdef np.ndarray[np.int32_t, ndim=2] sign_2d(np.ndarray[np.int32_t, ndim=2] a)

#cdef bool array_eq(np.ndarray[np.uint8_t, ndim=2] a, np.ndarray[np.uint8_t, ndim=2] b)
cdef bool array_eq( uint8_t [:,:] a, uint8_t [:,:] b)
    
cdef int int_min(int a, int b)

cdef np.ndarray[np.uint8_t, ndim=2] conv_a2d(np.ndarray[np.float64_t, ndim=1], int, double, double)

cdef np.ndarray[np.float64_t, ndim=1] conv_d2a(np.ndarray[np.uint8_t, ndim=2], double, double)

cdef np.ndarray[np.uint32_t, ndim=1] hash_fast(np.ndarray[np.uint8_t, ndim=2] context_bytes, np.ndarray[np.int32_t, ndim=1] pearson_tab, int mem)

cdef np.ndarray[np.uint32_t, ndim=1] hash_obs(np.ndarray[np.uint8_t, ndim=1] obs, int r)