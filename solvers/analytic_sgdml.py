
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 21:40:09 2021

@author: lihao
"""

from functools import partial

import numpy as np
import scipy as sp
import timeit
import time 

import sys
import logging
import warnings
import multiprocessing as mp
Pool = mp.get_context('fork').Pool

DONE = 1
NOT_DONE = 0


class Analytic(object):
    def __init__(self, gdml_train, desc, callback=None):

        self.log = logging.getLogger(__name__)

        self.gdml_train = gdml_train
        self.desc = desc

        self.callback = callback

    
    def solve_xyz(self, task,sig_candid, R_desc, R_d_desc, tril_perms_lin, tril_perms_lin_mirror, y):

        sig = sig_candid
        lam = task['lam']
        use_E_cstr = task['use_E_cstr']

        n_type=task['n_type']
        n_train, dim_d = R_d_desc.shape[:2]
        n_atoms = int((1 + np.sqrt(8 * dim_d + 1)) / 2)
        dim_i = 3 * n_atoms
        

        # Compress kernel based on symmetries
        col_idxs = np.s_[:]
        if 'cprsn_keep_atoms_idxs' in task:

            cprsn_keep_idxs = task['cprsn_keep_atoms_idxs']
            cprsn_keep_idxs_lin = (
                np.arange(dim_i).reshape(n_atoms, -1)[cprsn_keep_idxs, :].ravel()
            )

            col_idxs = (
                cprsn_keep_idxs_lin[:, None] + np.arange(n_train) * dim_i
            ).T.ravel()

        if self.callback is not None:
            self.callback = partial(
                self.callback,
                disp_str='Assembling kernel matrix',
            )
        
        alphas=np.empty((3*n_atoms*n_train,1))
        
        start_i = timeit.default_timer()
        K,dur_s = self.gdml_train._assemble_kernel_mat(
                n_type,
                R_desc,
                R_d_desc,
                tril_perms_lin,
                tril_perms_lin_mirror,
                sig,
                self.desc,
                use_E_cstr=use_E_cstr,
                col_idxs=col_idxs,
                callback=self.callback,
               
            )
        stop_i = timeit.default_timer()
        
        #start = time.time()
# your code here    
        #print()
        start = timeit.default_timer()
          # get the prediction for all C and all H atoms
            
         #index_type=int(ind_i/3)
         #index_atom=ind_i-3*index_type
         #index=np.repeat(np.arange(n_train)*dim_i,3)+np.tile(np.array([3*ind_i,3*ind_i+1,3*ind_i+2]),n_train)
        #K=K_all.copy()
         #K=K_all[index,index]
        # a=K_r_allr[index,:]
        # K=a[:,index]
         
     
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
    
            if K.shape[0] == K.shape[1]:
    
                K[np.diag_indices_from(K)] += lam  # regularize
    
                if self.callback is not None:
                    self.callback = partial(
                        self.callback,
                        disp_str='Solving linear system (Cholesky factorization)',
                    )
                    self.callback(NOT_DONE)
    
                try:
    
                    # Cholesky
                    L, lower = sp.linalg.cho_factor(
                        K, overwrite_a=True, check_finite=False
                    )
                    alphas  = sp.linalg.cho_solve(
                        (L, lower), y, overwrite_b=True, check_finite=False
                    )
                except np.linalg.LinAlgError:  # try a solver that makes less assumptions
    
                    if self.callback is not None:
                        self.callback = partial(
                            self.callback,
                            disp_str='Solving linear system (LU factorization)      ',  # Keep whitespaces!
                        )
                        self.callback(NOT_DONE)
    
                    try:
                        # LU
                        alphas  = sp.linalg.solve(
                            K,  y, overwrite_a=True, overwrite_b=True, check_finite=False
                        )
                    except MemoryError:
                        self.log.critical(
                            'Not enough memory to train this system using a closed form solver.\n'
                            + 'Please reduce the size of the training set or consider one of the approximate solver options.'
                        )
                        print()
                        sys.exit()
    
                except MemoryError:
                    self.log.critical(
                        'Not enough memory to train this system using a closed form solver.\n'
                        + 'Please reduce the size of the training set or consider one of the approximate solver options.'
                    )
                    print()
                    sys.exit()
            else:
    
                if self.callback is not None:
                    self.callback = partial(
                        self.callback,
                        disp_str='Solving overdetermined linear system (least squares approximation)',
                    )
                    self.callback(NOT_DONE)
    
                # least squares for non-square K
                alphas = np.linalg.lstsq(K, y, rcond=-1)[0]

        stop = timeit.default_timer()

        
        #start=time.time()
        inverse_time= stop-start
        #kernel_time=stop_i-start_i
        kernel_time=dur_s
            
        return alphas,inverse_time,kernel_time
    
   
        

     