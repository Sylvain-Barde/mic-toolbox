# -*- coding: utf-8 -*-
"""Classes for the Markov Information Criterion Toolbox

Contains the following classes:
    Tree -- Context tree object
    DataStruct -- Binary data structure
    Tables -- Lookup tables and constants for integer arithmetic/hashing

NOTE: While the classes can be imported and used directly, they are designed to
be used by the high-level functions in the toolbox. In particular, most of the
methods rely on properly formatted arguments.

Created on Tue Sep 27 08:41:20 2016
This version 04/07/2019
@author: Sylvain Barde, University of Kent.
"""

# -- Imports
import numpy as np      # Objects require numpy arrays and functions 
cimport numpy as np     # Cython numpy import
from libc.stdint cimport uint8_t, uint16_t, uint32_t, int32_t # Integer types
from libc.math cimport floor
from cpython cimport bool # Boolean
cimport functions_c as f  # Import custom Cython functions

#------------------------------------------------------------------------------
# Context tree object
cdef class Tree:
    """ Context tree class for the MIC model comparison protocol
    
    This is the main class for the algorithm, as it implements the context 
    trees required for context tree weighting/maximising. The implementation is
    an extension the integer arithmetic/peasron hashing implementation of:
        
    F.M.J. Willems and Tj. J. Tjalkens, Complexity Reduction of the 
    Context-Tree Weighting Algorithm: A Study for KPN Research, Technical 
    University of Eindhoven, EIDMA Report RS.97.01
    
    Extensions include:
    - 16 bit arithemtic (up from 8)
    - Pruning/rollout in the hashing function.
    - Full conditioning on observation bits (using an observation hash)
    
    Keyword arguments:
        tag -- string for tagging the tree (38 char max, displayed in logs)
        mem -- integer, maximum number of tree nodes (memory cap)
        lags -- integer, maximumm number of lags in the markov process
        d -- integer, desired length of the context string in bits
        r_vec -- a python list of discretisation resolutions, in bits.
        perm -- d-by-1 1D numpy array of int32 containing a permutation
        
    Attributes:
        # Constants, tree / discretisation structure
        self.mem -- Number of nodes allocated        
        self.lags -- Set observation lags
        self.d -- Depth of tree
        self.n_bytes -- Path length in Bytes
        self.r_vec -- Variable resolutions vector
        self.var -- Target variable
        self.r -- Resolution needed for target variable
        self.perm -- Permutation vector
        self.t -- Number of trees required for 
        self.bins -- Discretisation support for target variable
        
        # Tree diagnostics
        self.tag -- Tree description tag        
        self.node_fail -- Counter for node allocation failures
        self.rescl -- Counter for leaf rescalings
        self.train_obs -- Counter for training observations processed
        self.root -- Index of root node
        
        # Hashing/node ID data
        self.node_id -- ID of a node (allows hash collision checks)
        self.node_org -- ID of original leaf for path (allows pruning/rollout)
        self.node_dat -- Node hashing data, number of hash attempts and depth
        
        # Internal counters / ratio attributes
        self.a_count -- Number of '0' observations at node
        self.b_count -- Number of '1' observations at node
        self.log_beta -- Log of internal beta switch
        self.log_Q -- log of maximum a posteriori node probability
        
    Methods:
        desc -- Provides descriptive statistics for the tree 
        hash_mem -- Allocates and retrieves memory indexes for nodes
        hash_path -- Generates hash indices for the path from the context
        update -- Update node counters and ratios on path with an observation
        cond_prob -- Extract the conditional probability of a transition from
            the tree
        uncond_prob -- Extract the unconditional probability of a transition 
            from the tree
        prob_vec -- Extract the probability mass function conditional on a 
            context.
    """
    
    def __init__(self, str tag, int mem, int lags, double d, list r_vec, 
                 int var, np.ndarray[np.int32_t, ndim=1] perm):
                     
        # Type declarations for underlying tables
        cdef np.ndarray[np.uint8_t, ndim=2] node_id_base, node_org_base, 
        cdef np.ndarray[np.uint8_t, ndim=2] node_dat_base
        cdef np.ndarray[np.uint16_t, ndim=2] a_count_base, b_count_base
        cdef np.ndarray[np.int32_t, ndim=2] log_beta_base, log_Q_base
        
        # Constants - (tree / discretisation structure)
        self.mem       = mem                    # Number of nodes allocated        
        self.lags      = lags                   # Set observation lags
        self.d         = np.uint8(d)            # Depth of tree
        self.n_bytes   = np.uint8(np.ceil(d/8)) # Path length in Bytes
        self.r_vec     = r_vec                  # Variable resolutions vector
        self.var       = var                    # Target variable
        self.r         = np.uint8(self.r_vec[self.var - 1]) # Resolution needed
        self.perm      = perm                   # Permutation vector
        self.t         = 2**self.r - 1          # Number of bins required
        self.bins      = np.zeros([2**self.r,1],dtype=np.double) 
        
        # Tree diagnostics
        self.tag       = tag                    # Set tree description tag        
        self.node_fail = 0                      # Node allocation failures
        self.rescl     = 0                      # Counter for rescalings
        self.train_obs = 0                      # Training obs counter
        self.root      = mem                    # Index of root node
        
        # Hashing/node ID data
        node_id_base   = np.zeros([mem,self.n_bytes],dtype=np.uint8)   # ID
        node_org_base  = np.zeros([mem,self.n_bytes],dtype=np.uint8)   # Origin
        node_dat_base  = np.zeros([mem,2],dtype=np.uint8) # Node hashes+depth
        self.node_id   = node_id_base
        self.node_org  = node_org_base
        self.node_dat  = node_dat_base
        
        # Internal counters / ratio attributes
        a_count_base  = np.zeros([self.t,2*mem],dtype=np.uint16)
        b_count_base  = np.zeros([self.t,2*mem],dtype=np.uint16)
        log_beta_base = np.zeros([self.t,mem],dtype=np.int32)
        log_Q_base    = np.zeros([self.t,mem],dtype=np.int32)
        self.a_count  = a_count_base
        self.b_count  = b_count_base
        self.log_beta = log_beta_base
        self.log_Q    = log_Q_base

    #--------------------------------------------------------------------------
    # desc method
    cpdef void desc(self):
        """Provides descriptive statistics for the tree, no argument needed"""
    
        cdef uint32_t free_nodes,leaf_node, branch_nodes, node_mem
        cdef uint8_t depth, tries, max_tries_leaf, max_tries_branch
        cdef int pad, i
        cdef str divider
        cdef double av_tries_leaf, av_tries_branch, 
        cdef uint8_t[:,:] free = np.zeros([1,self.n_bytes],dtype = np.uint8)
    
        # -- Set counters
        free_nodes = 0
        leaf_nodes = 0
        branch_nodes = 0
        av_tries_leaf = 0
        av_tries_branch = 0
        max_tries_leaf = 1
        max_tries_branch = 1
        
        # -- Calculate memory cost per node
        node_mem = 16*self.t + 2*self.n_bytes + 2
                                        # 1st term corresponds to counts/ratios
                                        # 2nd is 'node_id' & 'node_org'
                                        # 3rd is 2 bytes for depth & tries

        # -- Perform memory diagnosis - Scan all memory locations
        for i in range(self.mem):
            check = self.node_id[i,None]

            if f.array_eq(check, free):               # Node is free
                free_nodes = free_nodes + 1 
            else:                                   # Node is used
                depth = self.node_dat[i,0]          # Stored depth
                tries = self.node_dat[i,1]          # Stored excess tries
                if depth == 1:                      # Node is a leaf node
                    leaf_nodes = leaf_nodes + 1;
                    av_tries_leaf = av_tries_leaf + tries + 1
                    max_tries_leaf = max(max_tries_leaf, tries + 1)     
                else:                               # Node is a branch node
                    branch_nodes = branch_nodes + 1;
                    av_tries_branch = av_tries_branch + tries + 1
                    max_tries_branch = max(max_tries_branch, tries + 1)

        # Calculate memory useage values for printing
        tot_mem = node_mem*self.mem
        eff_mem = node_mem*(self.mem - free_nodes)
        av_tries_leaf = av_tries_leaf/leaf_nodes
        av_tries_branch = av_tries_branch/branch_nodes
        
        # -- Print tree diagnosis
        divider = '+' + '-'*54 + '+'
        pad = int(np.floor((38 - len(self.tag))/2))
        if pad < 0:
            pad = 0   #% Don't pad if string is longer than space available

        print(' ')
        print(divider)
        print(' '*pad + 'Tree diagnostics: ' + self.tag)
        print(divider)
        print(' Variable ' + str(self.var) + ' of ' + str(len(self.r_vec)))        
        print(' Training observations:       {:>10d}'.format(self.train_obs))
        print(divider)
        print(' Memory locations allocated:  {:>10d} nodes'.format(self.mem))
        print(' Memory usage per node:       {:>10d} Bytes'.format(node_mem))
        print(' Memory space required:       {:>10d} Bytes'.format(tot_mem))
        print(' Effective memory useage:     {:>10d} Bytes'.format(eff_mem))
        print(divider)
        print(' Resolution:    {:>10d} bits'.format(self.r))
        print(' Tree depth:    {:>10d} bits'.format(self.d))
        print(' Leaf nodes:    {:>10d} nodes'.format(leaf_nodes))
        print(' Branch nodes:  {:>10d} nodes'.format(branch_nodes))
        print(' Free nodes:    {:>10d} nodes'.format(free_nodes))
        print(divider)
        print(' N° of failed allocations:     {:>7d}'.format(self.node_fail))
        print(' Maximum N° of leaf tries:     {:>7d}'.format(max_tries_leaf))
        print(' Average N° of leaf tries:     {:7.4f}'.format(av_tries_leaf))
        print(' Maximum N° of branch tries:   {:>7d}'.format(max_tries_branch))
        print(' Average N° of branch tries:   {:7.4f}'.format(av_tries_branch))
        print(divider)
        print(' Number of leaf counter rescalings: ' + str(self.rescl))
        print(divider)

    #--------------------------------------------------------------------------
    # hash_mem method
    cpdef dict hash_mem(self, np.ndarray[np.uint8_t, ndim=1] context, str mode, 
                   object tab):
        """Allocates and retrieves memory indexes for nodes
        
        Dual purpose hashing used to map contexts to the pool of nodes. Used to
        Set locations in the tree when training the tree on the training set 
        and to get memory locations in the tree when drawing probabilities in 
        the second stage. Relies on 'hash_path' to generate the hash indices.
                
        Keyword arguments:
            context -- 1D numpy array of uint8 containing the binary context.
            mode -- a string controlling the hashing options. The following
                options are available:
                --> 'sd': Set memory locations directly (faster allocation).
                --> 'sp': Set memory locations with pruning & rollout 
                (more memory efficient allocation).
                --> 'g' : Get memory locations (no allocation).
            tables -- an instance of the Table class.
            
        Returns a dictionary with the following keys:
            'index' : a 1d numpy array of uint32 containing the memory index of
                each node in the context path.
            'child' : a 1d numpy array of uint32 indicating which of the two 
                child nodes is on the path.
            'offshoot' : a 1d numpy array of uint32 containing the memory index
                of the child that is not on the path.
        """

        # Type declarations for method
        cdef np.ndarray[np.uint8_t, ndim=2] free, context_bytes, origin, id_vec
        cdef np.ndarray[np.uint8_t, ndim=2] context_path, offshoot_path, org
        cdef np.ndarray[np.uint8_t, ndim=2] org_bytes, org_path, check
        cdef np.ndarray[np.uint32_t, ndim=1] child, index, offshoot, org_child,
        cdef np.ndarray[np.uint32_t, ndim=1] root_index, rollout_index,
        cdef np.ndarray[np.uint32_t, ndim=1] path_start, rollout_child, id_child
        cdef np.ndarray[np.int32_t, ndim=1] d_crit_v
        cdef np.ndarray[np.uint8_t, ndim=1, cast = True] good
        cdef np.ndarray[np.uint8_t, ndim=1] org_bits 
        cdef dict index_struct
        cdef int i, rollout_length, d_roll, id_len, d_crit
        cdef uint32_t r_ind, id_crit, loc
        
        child = np.uint32(context)                        # Get child indicator
        free = np.zeros([1,self.n_bytes],dtype = np.uint8) # PUT 'free' in tab?
        
        pearson_tab = np.asarray(tab.pearson, dtype = np.int32)
        
        context_bytes = np.zeros([1,self.n_bytes],dtype = np.uint8)
        context_bytes[0,:] = np.packbits(context)
        context_path = np.tile(context_bytes,(self.d,1))
        origin = context_bytes     # Origin bytes (for rollout of pruned paths)

        # -- Use mask and pointer to generate leaf -> root and offshoot paths
        context_path  = context_path & tab.mask         # AND with mask
        offshoot_path = context_path ^ tab.toggle       # XOR -> offshoot path
        context_path  = context_path | tab.bit_pointer  # OR with pointer

        if mode == 'sd':     # Set memory locations directly (no pruning) 

            # - Set the hash index locations for the context path directly
            index = self.hash_path(context_path,origin,0,pearson_tab,'s')

        elif mode == 'sp':   # Set memory locations with pruning & rollout 

            # - Step 1: Get the hash index for the context path
            index = self.hash_path(context_path,origin,0,pearson_tab,'g')

            # - Step 2: Find branching depth of single path portion (from leaf)
            good = index < self.mem                # find good nodes (no error)
            id_vec = np.zeros([self.d,self.n_bytes],dtype = np.uint8)
            for i in range(self.d):             # copy good ids
                if good[i] == True:
                    id_vec[i,:] = self.node_id[index[i],:]
            d_crit_v = f.nonzero_sum(id_vec)
            
            if d_crit_v.size == 0:  # All nodes are free (no pre-existing tree)
                # - only on 1st iter! -> Assign a memory location to root node
                root_index = self.hash_path(context_path[[self.d-1]],
                                            origin,self.d-1,pearson_tab,'s')
                rollout_index = np.uint32([])       # No nodes to roll out
                path_start = np.uint32([])          # No pruning

            elif d_crit_v[0] == 0:    # All nodes already allocated (full path)
                
                root_index = index             # Index is the root path
                rollout_index = np.uint32([])  # No nodes to roll out
                path_start = np.uint32([])     # No pruning
                
            else:      # Path contains free & non-free nodes -> pruning/rollout
            
                d_crit = d_crit_v[0]             # Set crit depth
                root_index = index[d_crit:self.d]   # Already allocated path
                
                # - Step 3: recover original paths from branching node
                org = np.asarray(self.node_org[index[d_crit],None]) # Leaf                
                
                if f.array_eq(origin,org):  # Already a pruned single path
                    
                    rollout_index = np.uint32([])  # No nodes to roll out
                    path_start = np.uint32([])     # No pruning
                    
                else:

                    # -- Reconstruct original path data from leaf origin
                    org_bytes = np.tile(org,(self.d,1))     # Get original path
                    org_path  = org_bytes & tab.mask        # AND with mask
                    org_path  = org_path | tab.bit_pointer  # OR with pointer

                    org_bits = np.unpackbits(org)           # Original bits
                    org_bits = org_bits[0:self.d]           # Trim out padding
                    org_child = np.uint32(org_bits)         # Child index

                    # - Step 4: Find overlap between 2 paths (requires rollout)
                    #           Rollout is 1 node longer than overlap, except:
                    #           - if connecting to existing path (no rollout)            
                    #           - for leaf 'scale' case (hence 'min' condition)

                    common = np.equal(context_path[0:d_crit,:],
                                      org_path[0:d_crit,:])
                    num_com = np.sum(np.sum(common,1) == self.n_bytes)
                    rollout_length = f.int_min(num_com, d_crit - 1)
                    d_roll = d_crit - rollout_length - 1 # depth of rollout end
                    rollout_path = org_path[d_roll:d_crit]
                    rollout_child = org_child[d_roll:d_crit]
                    
                    # - Step 5: Assign a temporary index to rollout nodes 
                    rollout_index = self.hash_path(rollout_path,origin,0,
                                                   pearson_tab,'g')
               
                    r_ind = rollout_index[0]
                    if r_ind == self.mem:
                        check = free + 1      # in case node can't be allocated
                    else:
                        check = np.asarray(self.node_id[r_ind,None])  # Get 1st
                    
                    # -- Rollout modes: 
                    #       1 -> Connect to existing path (no rollout)
                    #       2 -> Rollout to leaf (no pruning)
                    #       3 -> Generic rollout & prune

                    if rollout_length == 0 and (rollout_index == self.mem or
                        not f.array_eq(check, free)):
                       
                        rollout_index = np.uint32([])  # No rollout (step 6)

                        # - Step 7: Assign memory location to first pruned node
                        path_start = self.hash_path(context_path[d_crit-1,None],
                                                    origin, d_roll,pearson_tab,
                                                    's')
                                                    
                    elif rollout_length == d_crit - 1:                # Mode 2

                        # - Step 5: Assign permanent index to rollout nodes
                        rollout_index = self.hash_path(rollout_path,org,
                                                       d_roll,pearson_tab,'s')
                        
                        # - Deal with memory allocation errors
                        err = np.where(rollout_index == self.mem)    # Identify
                        err = err[0]
                        rollout_index = np.delete(rollout_index,err)  # Remove
                        rollout_child = np.delete(rollout_child,err)  

                        path_start = np.uint32([])    #  No pruned node exists

                        # - Step 6: Copy branching node data into rollout nodes
                        #   !! copy a/b counts from child loc. in origin path
                        id_crit = index[d_crit] + \
                            org_child[d_crit]*np.uint32(self.mem)
                        id_child = rollout_index + \
                            rollout_child*np.uint32(self.mem)

                        id_len = len(id_child)
                        for i in range(id_len):
                            self.a_count[:,id_child[i]] = \
                                self.a_count[:,id_crit]
                            self.b_count[:,id_child[i]] = \
                                self.b_count[:,id_crit]
                            self.log_beta[:,rollout_index[i]] = \
                                self.log_beta[:,index[d_crit]]
                            self.log_Q[:,rollout_index[i]] = \
                                self.log_Q[:,index[d_crit]]

                        # Check rollout node is path start, fix if not 
                        if 0 not in err and not rollout_index[0] == index[0]:
                            rollout_index = np.delete(rollout_index,0)
                            path_start = self.hash_path(context_path[0,None],
                                            origin,0,pearson_tab,'s')
                        
                    else:                                             # Mode 3
                                                
                        # - Step 5: Assign permanent index to rollout nodes
                        rollout_index = self.hash_path(rollout_path,org,
                                                       d_roll,pearson_tab,'s')
                                                       
                        # - Deal with memory allocation errors
                        err = np.where(rollout_index == self.mem)  # Locate
                        err = err[0]
                        rollout_index = np.delete(rollout_index,err)  # Remove
                        rollout_child = np.delete(rollout_child,err)  
                        
                        # - Step 6: Copy branching node data into rollout nodes
                        #   !! copy a/b counts from child loc. in origin path
                        id_crit = index[d_crit] + \
                            org_child[d_crit]*np.uint32(self.mem)
                        id_child = rollout_index + \
                            rollout_child*np.uint32(self.mem)
                        
                        id_len = len(id_child)
                        for i in range(id_len):
                            self.a_count[:,id_child[i]] = \
                                self.a_count[:,id_crit]
                            self.b_count[:,id_child[i]] = \
                                self.b_count[:,id_crit]
                            self.log_beta[:,rollout_index[i]] = \
                                self.log_beta[:,index[d_crit]]
                            self.log_Q[:,rollout_index[i]] = \
                                self.log_Q[:,index[d_crit]]
                        
                        # - Trim 1st rollout node -> not needed in steps 7 & 8
                        if 0 not in err: # If first node not misallocated
                            rollout_index = np.delete(rollout_index,0)                        

                        # - Step 7: Assign memory location to first pruned node
                        d_start = d_crit - len(rollout_index) - 1
                        path_start = self.hash_path(context_path[d_start,None],
                                                    origin,d_roll,pearson_tab,
                                                    's')

            # - Step 8: Construct final index for pruned path
            index = np.concatenate((path_start, rollout_index, root_index))

            
        elif mode == 'g':   # - Get mode, just return indices (no allocation)
        
            index = self.hash_path(context_path, origin, 0, pearson_tab, mode)
            
            # -- Remove unallocated nodes from index & find critical depth   
            good = index < self.mem                # find good nodes (no error)
            id_vec = np.zeros([self.d,self.n_bytes],dtype = np.uint8)
            for i in range(self.d):             # copy good ids
                if good[i] == True:
                    id_vec[i,:] = self.node_id[index[i],:]
            d_crit_v = f.nonzero_sum(id_vec)
            
            index = index[d_crit_v[0]:self.d]
            
        cut = len(index)            # how long is the index path?
        
        # -- Get the offshoot hash index (needed for CTM)
        offshoot_path = offshoot_path | tab.bit_pointer         # OR on pointer
        offshoot_path = offshoot_path[(self.d - cut):self.d,:]          # Trim
        offshoot = self.hash_path(offshoot_path, origin, 0, pearson_tab,'g')
        
        
        # -- Package outputs (dict)
        child = child[(self.d - cut):self.d]
        index_struct ={'index' : index,
                       'child' : child,
                       'offshoot' : offshoot}

        return index_struct        

    #--------------------------------------------------------------------------
    cpdef np.ndarray[np.uint32_t, ndim=1] hash_path(self, 
                    np.ndarray[np.uint8_t, ndim=2] path, 
                    np.ndarray[np.uint8_t, ndim=2] origin, uint8_t d_base, 
                    np.ndarray[np.int32_t, ndim=1] pearson_tab, str mode):
        """Generates hash indices for the path from the context
        
        Hashing function which generates collision-free mapping from a binary 
        context string to a set of memory indices. Low level function used 
        exclusively by 'hash_mem', not designed to be used directly. Uses the
        Willems, Shtarkov & Tjalkens precomputed 32-bit hash table, 
        derived from the 8-bit table proposed in Pearson (1990), Computing 
        Practices.
                
        Keyword arguments:
            context -- 1D numpy array of uint8 containing the binary context.
            origin -- 2D numpy array of uint8 containing the context bytes of 
                the original leaf (used when the path has been pruned).
            d_base -- a uint8, depth of the first entry of the context 
                (used when the path has been pruned)
            peason_tab -- pearson hash table (provided by the 'Table' class).
            mode -- a string controlling the hashing options. The following
                options are available:
                --> 'sd': Set memory locations directly (faster allocation).
                --> 'sp': Set memory locations with pruning & rollout 
                (more memory efficient allocation).
                --> 'g' : Get memory locations (no allocation).            
            
        Returns:
            'index' : a 1d numpy array of uint32 containing the memory index of
                each node in the context path.
        """
        
        # Type declarations for method
        cdef np.ndarray[np.uint32_t, ndim=1] index, ind_table, offset
        cdef uint32_t index_temp, mem#, offset_i
        cdef uint8_t i, k, tries, d
        
        # memview buffers to speed up checking on np arrays
        cdef uint8_t[:,:] path_buf, org_buf, check
        cdef uint8_t[:,:] free = np.zeros([1,self.n_bytes],dtype = np.uint8)
        org_buf = origin            # populate buffers        
        
        # Constants     
        tries = 32
        mem = self.mem
        d = path.shape[0]
        
        # Initialise index at 'error' value (mem)
        index = np.ones([d],dtype = np.uint32)*mem

        # -- Generate proposed hash, with an offset in case of collision
        ind_table = f.hash_fast(path, pearson_tab, mem)
        offset = f.hash_fast(path[:,0][:,None] + 1, pearson_tab, mem) + 2

        for i in range(d):
            path_buf = path[i,None]     # populate path buffer
            for k in range(tries):
                
                # Build an index for the node using hashing and offsets
                if k == 0:      # On 1st try, index given by the context hash 
                    index_temp = ind_table[i]
                else:           # On subsequent tries, add existing offset
                    index_temp = (index_temp + offset[i]) % mem

                # -- Check for availability or collision in the index space
                check = self.node_id[index_temp,None]

                # If location empty & index free
                if f.array_eq(check, free) and index_temp not in index:

                    index[i] = index_temp    # Return location's 32-bit index

                    if mode[0] == 's':              # If in 'set' mode

                        self.node_id[index_temp,:] = path_buf   # Save ID
                        self.node_org[index_temp,:] = org_buf   # Save origin
                        self.node_dat[index_temp,0] = i+d_base  # Save depth
                        self.node_dat[index_temp,1] = k         # Save tries
               
                    break                     # i^th node is indexed, move on
                
                # else, prior data saved in index -> Check it    
                elif f.array_eq(check, path_buf):       
                    index[i] = index_temp  # Return 32-bit index
                    break                  # i^th node is indexed, move on                    

#        # If 31 extra tries don't find an index, then index = mem (error)
        return index
    
    #--------------------------------------------------------------------------
    cpdef double update(self, dict index_struct, 
                        np.ndarray[np.uint8_t, ndim=1] obs, object tab):
        """Update node counters and ratios on path with an observation
        
        Update the counts and ratios of the nodes on the leaf -> root path. 
        Uses 16 bit integer arithemtic speed and memory gains, following:
        F.M.J. Willems and Tj. J. Tjalkens, Complexity Reduction of the 
        Context-Tree Weighting Algorithm: A Study for KPN Research, Technical 
        University of Eindhoven, EIDMA Report RS.97.01
                
        Keyword arguments:
            index_struct -- a python dict containing the memory indices for the
                nodes on the leaf -> root path, generated by 'hash_mem'.
            obs -- 1D numpy array of uint8 containing the observation bits.
            tab -- an instance of the Table class.            
            
        Returns a double containing the log score of the MSB at the root 
        (only used for debugging)
        """
        
        cdef np.ndarray[np.uint32_t, ndim=1] id_vec, off, child, 
        cdef np.ndarray[np.uint32_t, ndim=1] k, id_child, id_not_child
        cdef np.ndarray[np.int32_t, ndim=1] log, jac        
        cdef np.ndarray[np.uint8_t, ndim=1, cast = True] good        
        cdef int32_t calc, Pe_0, Pe_1, Pw_0, Pw_1, log_beta, log_eta
        cdef int32_t log_eta_plus_one, log_beta_plus_one, diff_0, diff_1
        cdef int32_t log_num, log_denom, check, beta_ratio, a_count, b_count
        cdef double learn, eta, P_1
        cdef uint32_t fetch_id, fetch_off, fetch_id_c, fetch_id_o, k_m
        
        cdef int i ,j , r, d, n, m
        cdef int32_t half_reg, reg, cutoff, cl, cu, A
        cdef uint32_t id_n, mem, node_fail
        
        self.train_obs = self.train_obs + 1     # Increment training counter
        mem = self.mem          # Local memory variable, to speed up checks
        
        # -- Unpackage memory index structure
        id_vec = index_struct['index']
        off    = index_struct['offshoot']
        child  = index_struct['child']
        id_child = id_vec + child*mem
        id_not_child = id_vec + (1-child)*mem
        
        r = len(obs)            # bits of resolution    
        d = len(id_vec)         # Determine length of leaf -> root path
                                # Variable because of pruning

        # -- Hash the observation bits
        k = f.hash_obs(obs,self.r)
        
        # -- Find indices for good nodes (Protect against unallocated nodes)
        good = np.logical_and(id_vec < mem, off < mem)  
        
        # -- extract counts and ratios from the tree object   
        # part 1: Initialise matrices (using memoryviews)
        cdef int32_t[:,:] a_mat_c = np.empty([r,d], dtype = np.int32)
        cdef int32_t[:,:] a_mat_o = np.empty([r,d], dtype = np.int32)
        
        cdef int32_t[:,:] b_mat_c = np.empty([r,d], dtype = np.int32)
        cdef int32_t[:,:] b_mat_o = np.empty([r,d], dtype = np.int32)
        
        cdef int32_t[:,:] log_beta_mat = np.empty([r,d], dtype = np.int32)
        cdef int32_t[:,:] log_Qc_mat = np.empty([r,d], dtype = np.int32)
        cdef int32_t[:,:] log_Qo_mat = np.empty([r,d], dtype = np.int32)

        # part 2: Extract matrices as memoryviews -> Requires a loop...
        for m in range(r):
            k_m = k[m]            
            for n in range(d):
                if good[n] == True:          # extract good nodes
                    fetch_id = id_vec[n]
                    fetch_off = off[n] 
                    fetch_id_c = id_child[n]
                    fetch_id_o = id_not_child[n]                

                    a_mat_c[m,n] = self.a_count[k_m,fetch_id_c]
                    a_mat_o[m,n] = self.a_count[k_m,fetch_id_o]
                    
                    b_mat_c[m,n] = self.b_count[k_m,fetch_id_c]
                    b_mat_o[m,n] = self.b_count[k_m,fetch_id_o]
                    
                    log_beta_mat[m,n] = self.log_beta[k_m,fetch_id]
                    log_Qc_mat[m,n] = self.log_Q[k_m,fetch_id]
                    log_Qo_mat[m,n] = self.log_Q[k_m,fetch_off]

                else:                        # Set bad nodes to zero (np.empty)
                    a_mat_c[m,n] = 0
                    a_mat_o[m,n] = 0
                    
                    b_mat_c[m,n] = 0
                    b_mat_o[m,n] = 0
                    
                    log_beta_mat[m,n] = 0
                    log_Qc_mat[m,n] = 0
                    log_Qo_mat[m,n] = 0

        # Unpack constants from tab object (to local variables)
        log      = np.asarray(tab.log_tab, dtype = np.int32)
        jac      = np.asarray(tab.jac_tab, dtype = np.int32)
        A        = np.int32(tab.A)
        reg      = np.int32(tab.reg)
        half_reg = np.int32(tab.half_reg)
        cl       = np.int32(tab.cl)
        cu       = np.int32(tab.cu)
        cutoff   = np.int32(tab.cutoff)
        
        node_fail   = 0
        for j in range(r):     # --- Loop over 'r' observation bits
            
            # Step 0 - Create 'log_eta' from leaf node at start of path
            if id_vec[0] == mem:
                if j == 0:                                # First encounter ...
                    node_fail = node_fail + 1   # ... increment count
                log_eta = 0
            else:       
                log_eta = log[2*a_mat_c[j,0]] - log[2*b_mat_c[j,0]]         
    
                    
            for i in range(d):    # -- Loop over 'd' nodes in leaf -> root path
                id_n = id_vec[i]
                
                # Step 1 - Convert incoming eta to Pw's
                if log_eta >= 0:               # tab.jac has >0 indices
                    if log_eta < cutoff:   # Cutoff check for jacobian logs
                        log_eta_plus_one = log_eta + jac[log_eta]
                    else:
                        log_eta_plus_one = log_eta 
                else:
                    if -log_eta < cutoff:  # Cutoff check for jacobian logs
                        log_eta_plus_one = jac[-log_eta]
                    else:
                        log_eta_plus_one = 0
            
                Pw_0 = log_eta - log_eta_plus_one
                Pw_1 = - log_eta_plus_one
                
                # Steps 2 -> 5: depends if the node has allocated memory
                if id_n == mem:        # IF memory allocation failed
    
                    if j == 0:                                # First encounter
                        node_fail = node_fail + 1   # increment count
                        
                    # Step 2 - Calculate P_e's (no counts  -> equiprobable)
                    Pe_0 = - A
                    Pe_1 = Pe_0
                    
                    # Step 3 - Update outgoing eta using Pw's Pe's (No beta!)
                    diff_0 = Pe_0 - Pw_0            # log beta = 0 (beta = 1)
                    if diff_0 >= 0:                 # tab.jac has >0 indices
                        if diff_0 < cutoff:     # tab.jac cutoff check 
                            log_num = Pe_0 + jac[diff_0]
                        else:
                            log_num = Pe_0
                    else:
                        if -diff_0 < cutoff:
                            log_num = Pw_0 + jac[-diff_0]
                        else:
                            log_num = Pw_0
                            
                    diff_1 = Pe_1 - Pw_1            # log beta = 0 (beta = 1)
                    if diff_1 >= 0:                 # tab.jac has >0 indices
                        if diff_1 < cutoff:     # tab.jac cutoff check 
                            log_denom = Pe_1 + jac[diff_1]
                        else:
                            log_denom = Pe_1
                    else:
                        if -diff_1 < cutoff:
                            log_denom = Pw_1 + jac[-diff_1]
                        else:
                            log_denom = Pw_1
    
                    log_eta = log_num - log_denom  # Updated eta
                    
                    # Steps 4 & 5 (Update Beta + counts) can't be carried out
                
                else:                 # ELSE node has memory, carry on as usual
                    
                    # Step 2 - Calculate P_e's (w. counts built from children)
                    a_count = a_mat_c[j,i] + a_mat_o[j,i]
                    b_count = b_mat_c[j,i] + b_mat_o[j,i]
                
                    Pe_0 = log[2*a_count] - log[a_count + b_count] - A
                    Pe_1 = log[2*b_count] - log[a_count + b_count] - A
                    
                    log_beta = log_beta_mat[j,i]
                    
                    # Step 3 - Update outgoing eta using beta, Pw's Pe's
                    calc = f.sum_overflow(Pe_0,-Pw_0,cl,cu)      # Overflow sum
                    diff_0 = f.sum_overflow(log_beta,calc,cl,cu) # Overflow sum
                    if diff_0 >= 0:                    # tab.jac has >0 indices
                        if diff_0 < cutoff:            # tab.jac cutoff check 
                            log_num = Pe_0 + log_beta + jac[diff_0]
                        else:
                            log_num = Pe_0 + log_beta
                    else:
                        if -diff_0 < cutoff:
                            log_num = Pw_0 + jac[-diff_0]
                        else:
                            log_num = Pw_0
                     
                    calc = f.sum_overflow(Pe_1,-Pw_1,cl,cu)      # Overflow sum
                    diff_1 = f.sum_overflow(log_beta,calc,cl,cu) # Overflow sum
                    if diff_1 >= 0:                 # tab.jac has >0 indices
                        if diff_1 < cutoff:         # tab.jac cutoff check 
                            log_denom = Pe_1 + log_beta + jac[diff_1]
                        else:
                            log_denom = Pe_1 + log_beta
                    else:
                        if -diff_1 < cutoff:
                            log_denom = Pw_1 + jac[-diff_1]
                        else:
                            log_denom = Pw_1
    
                    log_eta = log_num - log_denom  # Updated eta
                                                        
                    # Step 4 & 5 - Update Beta and counts
                    if obs[j] == 0:
                        log_beta_mat[j,i]  = diff_0         # Update beta
                        log_beta = diff_0                   # Also local
                        if a_mat_c[j,i] == reg:             # a reg. full
                            a_mat_c[j,i] = half_reg         # halve it
                            b_mat_c[j,i] = b_mat_c[j,i]/2
                        else:
                            a_mat_c[j,i] = a_mat_c[j,i] + 1
                    else:
                        log_beta_mat[j,i]  = diff_1         # Update beta
                        log_beta = diff_1                   # Also local                        
                        if b_mat_c[j,i] == reg:    # b reg. full
                            b_mat_c[j,i] = half_reg  # halve it
                            a_mat_c[j,i] = b_mat_c[j,i] /2
                        else:
                            b_mat_c[j,i] = b_mat_c[j,i]  + 1                                    
                                    
                    # Step 6 - Perform CTM on node
                    # - calculate log(beta + 1) from log(beta) for node
                    if log_beta >= 0:                  # tab.jac has >0 indices
                        if log_beta < cutoff:      # tab.jac cutoff check
                            log_beta_plus_one = log_beta + jac[log_beta]
                        else:
                            log_beta_plus_one = log_beta
                    else:
                        if -log_beta < cutoff:     # tab.jac cutoff check
                            log_beta_plus_one = jac[-log_beta]
                        else:
                            log_beta_plus_one = 0
                    beta_ratio = log_beta - log_beta_plus_one
                    
                    # Process the Q variable from the child nodes using CTM
                    if i == 0:                    # at leaf nodes, log_Q = 0
                        check = 0
                    else:                         # Draw log_Q from child nodes
                        if not id_vec[i-1] == mem and not off[i-1] == mem:
                            check = log_Qc_mat[j,i-1] + log_Qo_mat[j,i-1]
                        else:
                            check = 0    # Protect against failed alloc.
                            
                    if check > log_beta:
                        log_Qc_mat[j,i] = check - log_beta_plus_one
                    else:
                        log_Qc_mat[j,i] = beta_ratio
    
                    # Finally, save root log score for MSB (for learning?)
                    if i == d-1 and j == r-1:
                        eta = 2**(np.double(log_eta)/np.double(A))
                        P_1 = 1/(eta+1)
                        learn = - np.double(obs[j])*np.log2(P_1) \
                                - np.double(1-obs[j])*np.log2(1-P_1)
        
        # Copy back loop output to object attributes and return          
        for m in range(r):
            k_m = k[m]
            for n in range(d):
                if good[n] == True:
                    fetch_id = id_vec[n]
                    fetch_id_c = id_child[n]
                    
                    self.a_count[k_m,fetch_id_c] = a_mat_c[m,n]
                    self.b_count[k_m,fetch_id_c] = b_mat_c[m,n]
                    
                    self.log_beta[k_m,fetch_id] = log_beta_mat[m,n]
                    self.log_Q[k_m,fetch_id] = log_Qc_mat[m,n]  
        
        self.node_fail = self.node_fail + node_fail
        
        return learn
    #--------------------------------------------------------------------------    
    cpdef dict cond_prob(self, dict index_struct, 
                    np.ndarray[np.uint8_t, ndim=1] obs, object tab):
        """Extract the conditional probability of a transition from the tree
        
        Uses Context Tree Maximisation to find the best location on the leaf ->
        root path from which to extract the conditional probability of the 
        transition, and provides the counts required for the calculation of 
        the bias.
        Uses 16 bit integer arithemtic speed and memory gains, following:
        F.M.J. Willems and Tj. J. Tjalkens, Complexity Reduction of the 
        Context-Tree Weighting Algorithm: A Study for KPN Research, Technical 
        University of Eindhoven, EIDMA Report RS.97.01
                
        Keyword arguments:
            index_struct -- a python dict containing the memory indices for the
                nodes on the leaf -> root path, generated by 'hash_mem'.
            obs -- 1D numpy array of uint8 containing the observation bits.
            tab -- an instance of the Table class.            
            
        Returns a dictionary with the following keys:
            'P' : a 1d nunmpy array of doubles containg the probabilities for 
                each bit of the observation
            'counts' : a 2d nunmpy array of unit32, contaning the a/b counts 
                for each bit of the observation 
            'trans_id' : a 2d nunmpy array of unit32, contaning the node ID and
                observation hash for each bit of the observation (enables 
                counting of distinct transitions, for debugging purposes)
        """
        
        cdef np.ndarray[np.uint32_t, ndim=1] id_vec, off, id_c
        cdef np.ndarray[np.uint32_t, ndim=1] k
        cdef np.ndarray[np.uint32_t, ndim=2] counts, trans_id
        cdef np.ndarray[np.double_t, ndim=1] P
        cdef np.ndarray[np.int32_t, ndim=2] Q_prod, log_beta, path_det, ctm
        cdef np.ndarray[np.int32_t, ndim=1] ind, ctm_max
        cdef np.ndarray[np.uint8_t, ndim=1, cast = True] good     
        cdef int d, r, i
        cdef uint32_t mem
        cdef uint32_t fetch_id, fetch_off, k_m
        cdef int32_t log_eta, log_eta_plus_one, a_count, b_count, cutoff
        cdef dict prob_struct

        # -- Unpackage memory index list
        id_vec = index_struct['index']
        off    = index_struct['offshoot']
        id_c   = index_struct['child']
        
        k = f.hash_obs(obs,self.r)  # Identify path in observation
        r = self.r
        d = len(id_vec)             # Determine length of leaf -> root path
                                    # Variable because of pruning
        mem = self.mem
        cutoff   = np.int32(tab.cutoff)

        
        # -- Find good nodes (Protect against unallocated nodes)
        good = np.logical_and(id_vec < mem, off < mem)  
        
        # -- Context maximisation via processing of beta and Q ratios
        Q_prod = np.zeros([self.r,d],dtype = np.int32)
        log_beta = np.zeros([self.r,d],dtype = np.int32)

        for m in range(r):
            k_m = k[m]            
            for n in range(d):
                if good[n] == True:
                    fetch_id = id_vec[n]
                    fetch_off = off[n]
                    Q_prod[m,n] = self.log_Q[k_m,fetch_id] + \
                                    self.log_Q[k_m,fetch_off] 
                    log_beta[m,n] = self.log_beta[k_m,fetch_id]

        lb0 = log_beta[:,0]
        
        path_det = log_beta[:,1:d] - Q_prod[:,0:d-1]
        ctm = np.fliplr(np.concatenate((lb0[:,None], path_det, \
            -np.ones([self.r,1],dtype = np.int32) ),axis =1))
        ctm_max = np.zeros([r],dtype = np.int32)    
        np.argmax(f.sign_2d(ctm), axis=1, out = ctm_max) # Leaf returned as '0'
        ind = d - ctm_max                                # Leaf returned as 'd'
                
        # -- Preallocate probability, count and transition ID vectors
        P        = np.zeros([self.r])
        counts   = np.zeros([2,self.r], dtype = np.uint32)
        trans_id = np.zeros([self.r,2], dtype = np.uint32)

        # -- Calculate probability vector from counts in the optimal nodes
        for i in range(self.r):
            
            if ind[i] == d:     # If leaf is selected, use child counts
                if id_vec[0] == mem:       # IF leaf is missing, use root
                    a_count = np.uint32(self.a_count[k[i],self.root]) + \
                        np.uint32(self.a_count[k[i],self.root +self.mem])
                    b_count = np.uint32(self.b_count[k[i],self.root]) + \
                        np.uint32(self.b_count[k[i],self.root + self.mem])
                    trans_id[i,0] = self.root
                else:                      # Else, use leaf as intended
                    a_count = self.a_count[k[i],id_vec[0] + 
                        id_c[0]*mem]
                    b_count = self.b_count[k[i],id_vec[0] + 
                        id_c[0]*mem]
                    trans_id[i,0] = id_vec[0]
            else:
                if id_vec[ind[i]] == mem:  # IF node is missing, use root
                    a_count = np.uint32(self.a_count[k[i],self.root]) + \
                        np.uint32(self.a_count[k[i],self.root +self.mem])
                    b_count = np.uint32(self.b_count[k[i],self.root]) + \
                        np.uint32(self.b_count[k[i],self.root + self.mem])
                    trans_id[i,0] = self.root
                else:                           # Else, use children counts
                    a_count = np.uint32(self.a_count[k[i],id_vec[ind[i]]]) + \
                        np.uint32(self.a_count[k[i],id_vec[ind[i]] + mem])
                    b_count = np.uint32(self.b_count[k[i],id_vec[ind[i]]]) + \
                        np.uint32(self.b_count[k[i],id_vec[ind[i]] + mem])
                    trans_id[i,0] = id_vec[ind[i]]
                    
            trans_id[i,1] = k[i]
        
            log_eta = tab.log_tab[2*a_count] - tab.log_tab[2*b_count]
            if log_eta >= 0:              # tab.jac has >0 indices
                if log_eta < cutoff:      # tab.jac cutoff check
                    log_eta_plus_one = log_eta + tab.jac_tab[log_eta]
                else:
                    log_eta_plus_one = log_eta
            else:
                if -log_eta < cutoff:     # tab.jac cutoff check
                    log_eta_plus_one = tab.jac_tab[-log_eta]
                else:
                    log_eta_plus_one = 0
            
            P[i] = 2**(np.double(-log_eta_plus_one)/np.double(tab.A))
            counts[0,i] = a_count
            counts[1,i] = b_count
        
        # -- Package outputs
        prob_struct = {'P' : P,
                       'counts' : counts,
                       'trans_id' : trans_id}

        return prob_struct
    #--------------------------------------------------------------------------
    cpdef dict uncond_prob(self, np.ndarray[np.uint8_t, ndim=1] obs, 
                           object tab):
        """Extract the unconditional probability of a transition from the tree
        
        Simply draws the probability of an observation from the root of the 
        tree
        Uses 16 bit integer arithemtic speed and memory gains, following:
        F.M.J. Willems and Tj. J. Tjalkens, Complexity Reduction of the 
        Context-Tree Weighting Algorithm: A Study for KPN Research, Technical 
        University of Eindhoven, EIDMA Report RS.97.01
                
        Keyword arguments:
            obs -- 1D numpy array of uint8 containing the observation bits.
            tab -- an instance of the Table class.            
            
        Returns a dictionary with the following keys:
            'P' : a 1d nunmpy array of doubles containg the probabilities for 
                each bit of the observation
            'counts' : a 2d nunmpy array of unit32, contaning the a/b counts 
                for each bit of the observation 
        """
        
        cdef np.ndarray[np.uint32_t, ndim=1] k
        cdef np.ndarray[np.uint32_t, ndim=2] counts
        cdef np.ndarray[np.double_t, ndim=1] P
        cdef uint16_t i
        cdef uint32_t a_count, b_count
        cdef int32_t log_eta, log_eta_plus_one, cutoff
        cdef dict prob_struct

        k = f.hash_obs(obs,self.r)  # Identify path in observation
        cutoff   = np.int32(tab.cutoff)

        counts = np.zeros([2,self.r],dtype = np.uint32) 
        P = np.zeros([self.r],dtype = np.double) 
        
        for i in range(self.r):
    
            a_count = np.uint32(self.a_count[k[i],self.root]) + \
                np.uint32(self.a_count[k[i],self.root +self.mem])
            b_count = np.uint32(self.b_count[k[i],self.root]) + \
                np.uint32(self.b_count[k[i],self.root + self.mem])
    
            log_eta = tab.log_tab[2*a_count] - tab.log_tab[2*b_count]
            
            if log_eta >= 0:              # tab.jac has >0 indices
                if log_eta < cutoff:      # tab.jac cutoff check
                    log_eta_plus_one = log_eta + tab.jac_tab[log_eta]
                else:
                    log_eta_plus_one = log_eta
            else:
                if -log_eta < cutoff:     # tab.jac cutoff check
                    log_eta_plus_one = tab.jac_tab[-log_eta]
                else:
                    log_eta_plus_one = 0
            
            P[i] = 2**(np.double(-log_eta_plus_one)/np.double(tab.A))
            counts[0,i] = a_count
            counts[1,i] = b_count
                
        # -- Package outputs
        prob_struct = {'P' : P,
                       'counts' : counts}
        
        return prob_struct
    #--------------------------------------------------------------------------
    cpdef dict prob_vec(self, dict index_struct, object tab, str mode):
        """Extract the probability mass function conditional on a 
            context.
        
        Used to convert the bit-level probabilities of observations into a 
        probability mass function on the discretised support of the variable. 
        This is useful for plotting the conditonal probability mass functions.
                
        Keyword arguments:
            index_struct -- a python dict containing the memory indices for the
                nodes on the leaf -> root path, generated by 'hash_mem'.
            tab -- an instance of the Table class.
            mode -- a string determining where to draw the probabilities from. 
                The following options are available:
                --> 'c': Conditional mode , will use 'cond_prob' to draw 
                    probabilities.
                --> 'u': Unconditional mode , will use 'uncond_prob' to draw 
                    probabilities from the root ('index_struct' is ignored).

        Returns a dictionary with the following keys:
            'P_vec': a 1d nunmpy array of doubles containg the probabilities on 
                the support of the variables (given by the 'bins' attribute of 
                the tree).
            'P' : a 2d nunmpy array of doubles containg the probabilities for 
                each bit of all possible observations.
        """
        
        cdef np.ndarray[np.uint8_t, ndim=2] states
        cdef np.ndarray[np.double_t, ndim=2] P, P_tab, P_prod
        cdef np.ndarray[np.double_t, ndim=1] P_vec
        cdef uint16_t i
        cdef str  state
        cdef dict prob_struct
        
        # Enumerate possible states for the variable
        states = np.zeros([2**self.r,self.r],dtype = np.uint8)
        for i in range(2**self.r):
            state = np.binary_repr(i,width = self.r)
            states[i] = np.array(list(state),dtype=np.uint8)
        
        states = np.fliplr(states)
        
        # -- Preallocate probability vector
        P = np.zeros([2**self.r,self.r], dtype = np.double)
        
        if mode == 'c':      # - Conditional mode
        
            # Get conditional probabilities for all states of the variable
            for i in range(2**self.r):
                prob_struct  = self.cond_prob(index_struct,
                                              np.asarray(states[i,:]), tab)
                P[i,:] = prob_struct['P'] 
        
        elif mode == 'u':   # - Unconditional mode

            # Get unconditional probabilities for all states of the variable
            for i in range(2**self.r):
                prob_struct = self.uncond_prob(np.asarray(states[i,:]), tab)
                P[i,:] = prob_struct['P'] 
        
        # Generate probability table for each bit in each state
        P_tab  = np.multiply(states,P) + np.multiply(1-states,1-P)
        P_prod = np.cumprod(P_tab,1)
        
        # Extract probability vector and normalise
        P_vec = P_prod[:,self.r-1]
        P_vec = P_vec/sum(P_vec)
        
        # -- Package outputs
        prob_struct = {'P_vec'  : P_vec,
                       'P'      : P}
        
        return prob_struct

#------------------------------------------------------------------------------
# Data structure object
cdef class DataStruct:
    """Binary data structure
        
    Used to perform the binary discretisation of the real-valued data and store
    both the binary data and the discretisation error, in order to be able to
    perform the discretisation diagnostics in the toolbox.
    
    Keyword arguments:
        data_vec -- a N-by-K 2d numpy float 64 array containing the data, where
            N is the number of oservations and K is the number of variables.
        lb_vec -- a python list of K lower bounds.
        lb_vec -- a python list of K upper bounds.
        r_vec -- a python list of discretisation resolutions, in bits.
        
    Attributes:
        self.string -- 2d numpy uint8 array containing the binary representation
            of the data
        self.errors -- a N-by-K 2d numpy float 64 array containing the 
            discretisation error
        self.lb -- list, containing the K lower bounds
        self.ub -- list, containing the K upper bounds
        self.r -- list of discretisation resolutions
        
    Methods:
        recover -- Recovers the original real-valued data
    """
   
    def __init__(self, np.ndarray[np.float64_t, ndim=2] data_vec, 
                 list lb_vec, list ub_vec, list r_vec):
        
        cdef int var, num_vars, n, cum_r, r_i
        cdef double lb_i, ub_i, unit
        cdef np.ndarray[np.uint8_t, ndim=2] code, data_bin
        cdef np.ndarray[np.float64_t, ndim=1] data, data_int, sig, err
        cdef np.ndarray[np.float64_t, ndim=2] err_vec

        n = data_vec.shape[0]               # number of observations
        num_vars = len(r_vec)               # number of variables
        
        code = np.zeros([n,sum(r_vec)],dtype=np.uint8)  # initialise output
        err_vec = np.zeros([n,num_vars])                # initialise errors
        cum_r = 0 
        
        for var in range(num_vars):
            # -- Extract variable-specific values from vectors
            data = data_vec[:,var]
            r_i  = r_vec[var]
            lb   = lb_vec[var]
            ub   = ub_vec[var]
        
            # -- Find and truncate out of bounds data
            ind_ub = (data > ub).nonzero()
            ind_lb = (data < lb).nonzero()
            np.put(data,ind_ub,ub)
            np.put(data,ind_lb,lb)
            
            # -- Quantise data with low-level function
            unit = (ub-lb)/(2**np.double(r_i))          # Discretisation unit
            data_bin = f.conv_a2d(data,r_i,lb,ub)
            data_int = f.conv_d2a(data_bin,lb,ub)
            sig = data/unit                             # Standardised data
            err = sig - data_int/unit                   # Quantization error
            
            # -- Package into outputs
            code[:,cum_r:cum_r + r_i] = data_bin
            err_vec[:,var] = err
            cum_r += r_i                    # Update the data count
        
        self.string = code
        self.errors = err_vec
        self.lb     = lb_vec
        self.ub     = ub_vec
        self.r      = r_vec

    #--------------------------------------------------------------------------
    cpdef recover(self):          # data recovery method
        """Recovers the original real-valued data
        
        Used to recover the original data, by first converting the binary 
        representation back to discrete levels on the original supprt, then 
        adding the discretisation error to recover the data.
        
        No keyword arguments

        Returns a N-by-K 2d numpy float 64 array containing the original data
        """
        
        cdef int n, var, num_vars, loc, r_i
        cdef double unit, lb, ub
        cdef list lb_vec, ub_vec, r_vec
        cdef np.ndarray[np.uint8_t, ndim=2] var_code, code
        cdef np.ndarray[np.float64_t, ndim=1] var_data
        cdef np.ndarray[np.float64_t, ndim=2] data, err

        code = np.asarray(self.string)      # Get encoded data
        err  = np.asarray(self.errors)      # Get upper bounds on range
        lb_vec   = self.lb                  # Get lower bounds on range
        ub_vec   = self.ub                  # Get upper bounds on range
        r_vec    = self.r                   # Get encoding resolutions

        n = code.shape[0]                   # Number of observations
        num_vars = len(r_vec)               # Number of variables
        data = np.zeros([n,num_vars])       # Initialise output

        loc = 0                             # Initialise location marker
        for var in range(num_vars):         # For each variable,
            
            r_i = r_vec[var]
            lb = lb_vec[var]
            ub = ub_vec[var]
            var_code = code[:,loc:loc+r_i]          # Get code for the variable
            var_data = f.conv_d2a(var_code, lb, ub)   # Convert it
            
            unit = (ub - lb)/(2**np.double(r_i))      # Calculate coding unit
            data[:,var] = var_data + err[:,var]*unit  # Add error terms

            loc += r_i        # Update location in code for next variable

        return data
#------------------------------------------------------------------------------
# Lookup tables for integer arithmetic
cdef class Tables:
    """Lookup tables and constants for integer arithmetic/hashing
    
    Keyword arguments:
        d -- The required depth of the tree.
        
    Attributes:
        # Constants
        self.A -- accuracy parameter
        self.reg -- uint16 register saturation
        self.half_reg -- unt16 half register
        self.cl -- int32 lower overflow bound
        self.cu -- int32 upper overflow bound
        self.n_bytes -- Number of context bytes needed given depth of tree
        
        # Integer arithemetic logarithmic look-up tables (no multiplication)
        self.log_tab -- logarithmic look-up table
        self.cutoff -- Cutoff for jacobian logarithm lookup
        self.jac_tab -- jacobian logartihm look-up table
        
        # Hashing tables
        self.pearson -- Pseudo-random permutation table for hash function
        self.mask -- Masking table for bitwise preparation of context path 
        self.bit_pointer -- Pointer for bitwise preparation of context path 
        self.toggle -- Toggle table for bitwise preparation of context path 
            offshoots
        
    No Methods
    """
        
    def __init__(self, double d):
        
        # Type declarations for underlying tables
        cdef uint8_t p
        cdef np.ndarray[np.double_t, ndim=1] ind_1
        cdef np.ndarray[np.double_t, ndim=1] ind_2
        cdef np.ndarray[np.int32_t, ndim=1] pearson_base
       
        cdef np.ndarray[np.uint8_t, ndim=2] mask_1, bit_pointer_1, full
        cdef np.ndarray[np.uint8_t, ndim=2] mask_base, bit_pointer_base, 
        cdef np.ndarray[np.uint8_t, ndim=2] toggle_base

        p = 16      # precision parameter (hardwired)

        # -- Set constants given precision parameter
        self.A        = np.uint32(2**(p+1))    # accuracy (p+1 : 2 child cntrs)
        self.reg      = np.uint16(2**p - 1)    # register saturation
        self.half_reg = np.uint16(2**(p-1))    # half register
        self.cl       = np.int32(-2**31 + 1)   # lower overflow bound
        self.cu       = np.int32(2**31 - 1)    # upper overflow bound
                        
        # -- Calculate logarithmic look-up table
        ind_1 = np.floor(self.A*np.log2(np.arange(1,2**18)) + 0.5)
        self.log_tab = np.int32(ind_1)        
        
        # -- Calculate Jacobian logarithmic look-up table
        self.cutoff = np.ceil(self.A*(np.log2(self.A)+1.5288)) + 1
        ind_2 = -np.arange(0,self.cutoff,1,dtype = np.double)/np.double(self.A)
        ind_2 = np.floor(self.A*np.log2(1 + 2**ind_2) + 0.5)
        self.jac_tab = np.int32(ind_2)  

        # -- Pseudo-random permutation table for pre-calculating hash function
        # Uses the Willems, Shtarkov & Tjalkens precomputed 32-bit hash table, 
        # derived from the 8-bit table proposed in Pearson (1990), Computing 
        # Practices.
        pearson_base = np.int32([
          175715, 11428377,  6429025,  1663333, 23160013, 23383373, 13454579,
          21820291, 15958541, 25300137,   829939, 11137997, 32754777, 30169415,
          5850653, 21372299,  1936299, 25930603, 28011331, 23806635, 21146549,
          11252897, 28614785, 10519007,  8511025, 31338949,  3261913, 29743389,
          31005773, 18632081,  5083357, 26271075, 14508753, 23253199, 13684507,
          13573115, 18611199, 33291877, 33449115,  6593227, 10144419, 13279781,
          10626139,  2382529,  5947455, 12599229,  4176947, 29110999,  3331965,
          14122125, 24939693,  9219547, 11394017, 31187013, 31474833,  4493797,
          9561129, 31730093,  2731497, 28174791, 32098091, 29830103, 19650243,
          30852053, 12833907, 30700077,  7482489,  2914805,  7992485, 32810335,
          10837921, 23044107, 27265791,   720783, 16748255, 26140285, 14581007,
          8196081, 17822045, 32595283, 22893479, 22259317, 27686021,  7636277,
          8729813, 20239751, 13993963, 25684823, 32200227, 22422391,  2324333,
          24604007, 23946753, 23462375,   124681, 31918193, 17330473,  7415959,
          19437313,  9896203, 16845629, 17513673, 20760837, 13174013, 17104055,
          16561691, 11934515,  1782765, 20180401, 32354743, 28423919, 28765833,
          15632831,  9027229, 29269159, 10266289, 10924435, 11637447, 26396405,
          13038615, 15996601,  1488961, 12075281,  4264165, 17884265, 14968853,
          6821141,  1381437, 18103393,  3957103,  6385465, 24066119, 20465275,
          4618805,  8008991,  3481237, 18781687,  9828029, 32947459, 12387141,
          16991359, 21266225,  8335701, 20009999, 22286055,   976719, 15159267,
          22012829, 31693831, 27002669,   470127, 19689079,  7239471,  7811001,
          19904693, 28882027, 11823663,  6958855,  3081979, 17234779, 16472607,
          22683613,  2088095, 31235775, 10403507, 12497441, 11673811,  2151187,
          13833155, 18072513, 29606323, 29471553, 28524619, 20990711,  4912877,
          16182419, 15503877,  9569595,   342621, 20602089,  6088723, 15209251,
          1254157, 19074505, 17680799, 29990825, 27240853, 27891119, 26586763,
          28216267,  9161271, 30029689,  3635335, 24676089,  8845649, 16339449,
          22149205, 33051657,  5507131,   539353,  3856427, 14167023,  2879015,
          32384923,  2595407, 26890135,  5216211, 26726993, 30560629,  5338407,
          24455053, 19369345, 26050871, 25245251, 20333385,  4409727, 21593797,
          25085337, 12949835, 26823529, 21719275, 23653017, 15374617, 10033225,
          18368933,  4826457, 27613267, 22565485,  5401919,  7159313, 20844915,
          1143761, 24367331, 30466953, 14911951, 25808479, 30301989,  6235377,
          19198055, 15754883,  6718009,  8534305,  3744253, 19004859, 33405627,
          29014907, 12286853, 24872215, 25499361, 18276439, 14702223,  5672667,
          9362289, 14381475, 24224259, 27394735])       
        self.pearson = pearson_base
        
        # Number of context bytes needed given depth of tree
        num_bytes = np.uint8(np.ceil(d/8))
        self.n_bytes = num_bytes

        # -- Basic blocks used for a single byte
        mask_1 = np.zeros([8,1],dtype=np.uint8)          # byte mask
        bit_pointer_1 = np.zeros([8,1],dtype=np.uint8)   # byte pointer
        full = np.uint8(255*np.ones([8,1]))                 # full byte
        
        mask_1[:,0] = [127, 63, 31, 15, 7, 3, 1, 0]          # fill mask
        bit_pointer_1[:,0] = [128, 64, 32, 16, 8, 4, 2, 1]   # fill pointer

        # -- Build the mask and pointer as a function of the byte requirement        
        mask_base = np.kron(np.uint8(np.eye(num_bytes)),mask_1) + \
              np.kron(np.uint8(np.triu(np.ones([num_bytes,num_bytes]),1)),full)
        bit_pointer_base = np.kron(np.uint8(np.eye(num_bytes)),bit_pointer_1)
        toggle_base = np.vstack((bit_pointer_base[1:int(d),:], \
                                 np.uint8(np.zeros([1,num_bytes]))))

        self.mask = mask_base[0:int(d),:]
        self.bit_pointer = bit_pointer_base[0:int(d),:]
        self.toggle = toggle_base
#------------------------------------------------------------------------------