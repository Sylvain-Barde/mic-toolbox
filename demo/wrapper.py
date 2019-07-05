import os
import sys
import time
import numpy as np
import mic.toolbox as mt
import multiprocessing as mp

def MIC_wrapper(inputs):
    """ wrapper function"""

    tic = time.time()

    # Unpack inputs and parameters
    params = inputs[0]
    model = inputs[1]
    var_vec = inputs[2]
    num = inputs[3]
    
    log_path = params['log_path']
    data_path = params['data_path']
    lb = params['lb']
    ub = params['ub']
    r_vec = params['r_vec']
    hp_bit_vec = params['hp_bit_vec']
    mem = params['mem']
    d = params['d']
    lags = params['lags']
    
    print (' Task number {:2d} initialised'.format(num))

    # Redirect output to file and print task/process information
    main_stdout = sys.stdout
    sys.stdout = open(log_path + '//log_' + str(num) + '.out', "w")
    print (' Task number :   {:3d}'.format(num))
    print (' Parent process: {:10d}'.format(os.getppid()))
    print (' Process id:     {:10d}'.format(os.getpid()))    
    
    # Load simulates/empirical data
    sim_data = np.loadtxt(model, delimiter="\t") 
    emp_data = np.loadtxt(data_path, delimiter="\t")
    
    # Pick a tag for the tree (useful for indentifying the tree later on)
    tag = model
    
    # -- Stage 1
    # -- Generate permutation from data
    perm = mt.corr_perm(emp_data[:,0:2], r_vec, hp_bit_vec, var_vec, lags, d)
    
    # Discretise the training data. 
    sim_dat1 = sim_data[:,0:2]
    data_struct = mt.bin_quant(sim_dat1,lb,ub,r_vec,'notests') # Note the 'notests' option
    data_bin = data_struct['binary_data']

    # Initialise a tree and train it, trying to predict the 1st variable
    var = var_vec[0]
    output = mt.train(None, data_bin, mem, lags, d, var, tag, perm)
    
    # Discretise the second run of training data
    sim_dat1 = sim_data[:,2:4]
    data_struct = mt.bin_quant(sim_dat1,lb,ub,r_vec,'notests') # Note, we are not running discretisation tests
    data_bin = data_struct['binary_data']

    # Extract the tree from the previous output and train it again. Only the 1st argument changes
    T = output['T']
    output = mt.train(T, data_bin, mem, lags, d, var, tag, perm)
    
    # Use the built-in descriptive statistic method to get some diagnostics
    T = output['T']
    T.desc()
    
    # -- Stage 2
    # Score the empirical data with the model probabilities
    scores = np.zeros([998,10])

    for j in range(10):
        loop_t = time.time()

        # Discretise the data
        k = 2*j
        dat = emp_data[:,k:k+2]
        data_struct_emp = mt.bin_quant(dat,lb,ub,T.r_vec,'notests')
        data_bin_emp = data_struct_emp['binary_data']

        # Score the data using the tree
        score_struct = mt.score(T, data_bin_emp)

        # Correct the measurement
        scores[:,j] = score_struct['score'] - score_struct['bound_corr']

        print('Replication {:2d}: {:10.4f} secs.'.format(j,time.time() - loop_t))
        
    # Redirect output to console and print completetion time
    sys.stdout = main_stdout
    toc = time.time() - tic
    print (' Task number {:3d} complete - {:10.4f} secs.'.format(int(num),toc))

    # Return scores
    return (scores)