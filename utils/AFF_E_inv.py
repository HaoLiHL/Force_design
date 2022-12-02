#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 13:17:20 2022

@author: lihao
"""

"""
Created on Wed Mar 10 15:52:35 2021

@author: lihao
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#import scipy as sp
import os
import timeit
import logging
import warnings

#os.chdir('/home/a510396/testl')
os.chdir('./')
#os.chdir('/Users/HL/Desktop/Study/SFM/AFF_code')
os.environ["MKL_NUM_THREADS"] = '4'
os.environ["NUMEXPR_NUM_THREADS"] = '4'
os.environ["OMP_NUM_THREADS"] = '4'

from utils import io
#import hashlib
import numpy as np
import scipy as sp
import multiprocessing as mp
Pool = mp.get_context('fork').Pool

from solvers.analytic_sgdml import Analytic
#from solvers.analytic_u_pp import Analytic
from utils import perm
from utils.desc import Desc
from functools import partial

# library for executing the external physics-based calculation 
import subprocess
import math
import pickle

#global glob
#glob = {}

def _share_array(arr_np, typecode_or_type):
    """
    Return a ctypes array allocated from shared memory with data from a
    NumPy array.
    Parameters
    ----------
        arr_np : :obj:`numpy.ndarray`
            NumPy array.
        typecode_or_type : char or :obj:`ctype`
            Either a ctypes type or a one character typecode of the
            kind used by the Python array module.
    Returns
    -------
        array of :obj:`ctype`
    """

    arr = mp.RawArray(typecode_or_type, arr_np.ravel())
    return arr, arr_np.shape

#draw_strat_sample(dataset2['E'],100)

def _assemble_kernel_mat_wkr(
    j, tril_perms_lin, tril_perms_lin_mirror, sig, n_type,use_E_cstr=False, exploit_sym=False, cols_m_limit=None
):
    r"""
    ----------
        j : int
            Index of training point.
        tril_perms_lin : :obj:`numpy.ndarray`
            1D array (int) containing all recovered permutations
            expanded as one large permutation to be applied to a tiled
            copy of the object to be permuted.
    Compute one row and column of the force field kernel matrix.
    The Hessian of the Matern kernel is used with n = 2 (twice
    differentiable). Each row and column consists of matrix-valued
    blocks, which encode the interaction of one training point with all
    others. The result is stored in shared memory (a global variable).
    """
    global glob

    R_desc_atom = np.frombuffer(glob['R_desc_atom']).reshape(glob['R_desc_shape_atom'])
    R_d_desc_atom = np.frombuffer(glob['R_d_desc_atom']).reshape(glob['R_d_desc_shape_atom'])
    K = np.frombuffer(glob['K']).reshape(glob['K_shape'])
    
    desc_func = glob['desc_func']
    
    n_train, dim_d = R_d_desc_atom.shape[:2]
    # 600; dim_d =66

    n_atoms = int((1 + np.sqrt(8 * dim_d + 1)) / 2)
    dim_i = 3 * n_atoms  # 36
    
    #n_perms =int(len(tril_perms_lin) / (dim_d * 6))
    n_perms = int(len(tril_perms_lin) / dim_d)  #12
    n_perm_atom=n_perms
    
    #blk_j = slice(j*3 , (j + 1) *3)
    blk_j = slice(j * dim_i, (j + 1) * dim_i)
    keep_idxs_3n = slice(None)  # same as [:]
    
    #tril_perms_lin_mirror=tril_perms_lin
    # Create permutated variants of 'rj_desc' and 'rj_d_desc'.
    #rj_desc_perms_mirror=np.reshape(
    #    np.tile(R_desc[j, :], n_perms/2)[tril_perms_lin_mirror], (int(n_perms/2), -1), order='F'
    #)
    
    rj_desc_perms = np.reshape(
        np.tile(R_desc_atom[j, :], n_perm_atom)[tril_perms_lin_mirror], (n_perm_atom, -1), order='F'
    )
    # rj_desc_perms = 2 * 66
    
    rj_d_desc = desc_func.d_desc_from_comp(R_d_desc_atom[j, :, :])[0][
        :, keep_idxs_3n
    ]  # convert descriptor back to full representation
    # rj_d_desc 66 * 36
    
    rj_d_desc_perms = np.reshape(
        np.tile(rj_d_desc.T, n_perm_atom)[:, tril_perms_lin_mirror], (-1, dim_d, n_perm_atom)
    )
    #  rj_d_desc_perms 36 * 66 * 2
    
    mat52_base_div = 3 * sig ** 4
    sqrt5 = np.sqrt(5.0)
    sig_pow2 = sig ** 2
    
    dim_i_keep = rj_d_desc.shape[1]  # 36
    diff_ab_outer_perms = np.empty((dim_d, dim_i_keep))   # 66 * 36
    #diff_ab_outer_perms_ij = np.empty((n_perm_atom,n_perm_atom,dim_d, dim_i_keep)) 
    diff_ab_perms = np.empty((n_perm_atom, dim_d))# 66 * 36
   # diff_ab_perms = np.empty((n_perm_atom, dim_d))  # 12 * 66
    #ri_d_desc = np.zeros((1, dim_d, dim_i)) # must be zeros! # 1* 66 * 36
    #k = np.empty((dim_i, dim_i_keep))   # 36 * 36
    #k=np.empty((1))
    ri_d_desc = np.zeros((1, dim_d, dim_i))
    k = np.empty((dim_i, dim_i_keep))
    #index_C=int(index_atom+n_atoms/2*3*index_type ) #2=n_type
    
    for i in range(0, j+1):
        blk_i = slice(i * dim_i, (i + 1) * dim_i)
        
        np.subtract(R_desc_atom[i, :], rj_desc_perms, out=diff_ab_perms)
        norm_ab_perms = sqrt5 * np.linalg.norm(diff_ab_perms, axis=1)
        mat52_base_perms = np.exp(-norm_ab_perms / sig) / mat52_base_div * 5
        
        np.einsum(
            'ki,kj->ij',
            diff_ab_perms * mat52_base_perms[:, None] * 5,
            np.einsum('ki,jik -> kj', diff_ab_perms, rj_d_desc_perms),
            out=diff_ab_outer_perms
        )
        
        diff_ab_outer_perms -= np.einsum(
            'ikj,j->ki',
            rj_d_desc_perms,
            (sig_pow2 + sig * norm_ab_perms) * mat52_base_perms,
        )
        
        desc_func.d_desc_from_comp(R_d_desc_atom[i, :, :], out=ri_d_desc)

        #K[blk_i, blk_j] = ri_d_desc[0].T.dot(diff_ab_outer_perms)
        np.dot(ri_d_desc[0].T, diff_ab_outer_perms, out=k)
        
        K[blk_i, blk_j] = -k
        # if (blk_i==blk_j) and (
        #     cols_m_limit is None or i < cols_m_limit
        # ):  # this will never be called with 'keep_idxs_3n' set to anything else than [:]
        #     K[blk_j, blk_i] = np.transpose(-k)
        K[blk_j, blk_i] =-k.T
        
    if use_E_cstr:

        E_off = K.shape[0] - n_train, K.shape[1] - n_train
        blk_j_full = slice(j * dim_i, (j + 1) * dim_i)
        for i in range(n_train):

            diff_ab_perms = R_desc_atom[i, :] - rj_desc_perms
            norm_ab_perms = sqrt5 * np.linalg.norm(diff_ab_perms, axis=1)

            K_fe = (
                5
                * diff_ab_perms
                / (3 * sig ** 3)
                * (norm_ab_perms[:, None] + sig)
                * np.exp(-norm_ab_perms / sig)[:, None]
            )
            K_fe = np.einsum('ik,jki -> j', K_fe, rj_d_desc_perms)
            K[blk_j_full, E_off[1] + i] = K_fe  # vertical
            K[E_off[0] + i, blk_j] = K_fe[keep_idxs_3n]  # lower horizontal

            K[E_off[0] + i, E_off[1] + j] = K[E_off[0] + j, E_off[1] + i] = (
                1 + (norm_ab_perms / sig) * (1 + norm_ab_perms / (3 * sig))
            ).dot(np.exp(-norm_ab_perms / sig))
         
    
    return blk_j.stop - blk_j.start



def _assemble_kernel_mat_wkr_test(
    j,tril_perms_lin, tril_perms_lin_mirror, sig,n_type, use_E_cstr=False, exploit_sym=False, cols_m_limit=None
):
    r"""
    ----------
        j : int
            Index of training point.
        tril_perms_lin : :obj:`numpy.ndarray`
            1D array (int) containing all recovered permutations
            expanded as one large permutation to be applied to a tiled
            copy of the object to be permuted.
    Compute one row and column of the force field kernel matrix.
    The Hessian of the Matern kernel is used with n = 2 (twice
    differentiable). Each row and column consists of matrix-valued
    blocks, which encode the interaction of one training point with all
    others. The result is stored in shared memory (a global variable).
    """
    global glob

    R_desc = np.frombuffer(glob['R_desc']).reshape(glob['R_desc_shape'])
    R_d_desc = np.frombuffer(glob['R_d_desc']).reshape(glob['R_d_desc_shape'])

    R_desc_val = np.frombuffer(glob['R_desc_val']).reshape(glob['R_desc_shape_val'])
    R_d_desc_val = np.frombuffer(glob['R_d_desc_val']).reshape(glob['R_d_desc_shape_val'])    
    
    #glob['R_desc_val'], glob['R_desc_shape_val'] = _share_array(R_desc_val_atom, 'd')
    #glob['R_d_desc_val'], glob['R_d_desc_shape_val'] = _share_array(R_d_desc_val_atom, 'd')    
 
    K = np.frombuffer(glob['K']).reshape(glob['K_shape'])
    
    desc_func = glob['desc_func']
    
    n_train, dim_d = R_d_desc.shape[:2]
    n_val, dim_d = R_d_desc_val.shape[:2]
    # dim_d =66

    n_atoms = int((1 + np.sqrt(8 * dim_d + 1)) / 2)
    dim_i = 3 * n_atoms  # 36

    n_perms = int(len(tril_perms_lin) / dim_d)
    n_perm_atom=n_perms
    
    blk_j = slice(j*dim_i , (j + 1)*dim_i )
    #blk_j = slice(j * dim_i, (j + 1) * dim_i)
    keep_idxs_3n = slice(None)  # same as [:]
  
    rj_desc_perms = np.reshape(
        np.tile(R_desc[j, :], n_perm_atom)[tril_perms_lin_mirror], (n_perm_atom, -1), order='F'
    )
    # rj_desc_perms = 12 * 66
    
    rj_d_desc = desc_func.d_desc_from_comp(R_d_desc[j, :, :])[0][
        :, keep_idxs_3n
    ]  # convert descriptor back to full representation
    # rj_d_desc 66 * 36
    

    
    rj_d_desc_perms = np.reshape(
        np.tile(rj_d_desc.T, n_perm_atom)[:, tril_perms_lin_mirror], (-1, dim_d, n_perm_atom)
    )
    #  rj_d_desc_perms 36 * 66 * 12
    
    mat52_base_div = 3 * sig ** 4
    sqrt5 = np.sqrt(5.0)
    sig_pow2 = sig ** 2
    
    dim_i_keep = rj_d_desc.shape[1]  # 36
    diff_ab_outer_perms = np.empty((dim_d, dim_i_keep))   # 66 * 36
    #diff_ab_outer_perms_ij = np.empty((n_perm_atom,n_perm_atom,dim_d, dim_i_keep)) 
    diff_ab_perms = np.empty((n_perm_atom, dim_d))# 66 * 36
   # diff_ab_perms = np.empty((n_perm_atom, dim_d))  # 12 * 66
    #ri_d_desc = np.zeros((1, dim_d, dim_i)) # must be zeros! # 1* 66 * 36
    #k = np.empty((dim_i, dim_i_keep))   # 36 * 36
    #k=np.empty((1))
    ri_d_desc = np.zeros((1, dim_d, dim_i))
    k = np.empty((dim_i, dim_i_keep))
    #index_C=int(index_atom+n_atoms/2*3*index_type ) #2=n_type
    
    for i in range(0, n_val):
        blk_i = slice(i * dim_i, (i + 1) * dim_i)
        #R_desc_val
        np.subtract(R_desc_val[i, :], rj_desc_perms, out=diff_ab_perms)
        norm_ab_perms = sqrt5 * np.linalg.norm(diff_ab_perms, axis=1)
        mat52_base_perms = np.exp(-norm_ab_perms / sig) / mat52_base_div * 5
        
        np.einsum(
            'ki,kj->ij',
            diff_ab_perms * mat52_base_perms[:, None] * 5,
            np.einsum('ki,jik -> kj', diff_ab_perms, rj_d_desc_perms),
            out=diff_ab_outer_perms
        )
        
        diff_ab_outer_perms -= np.einsum(
            'ikj,j->ki',
            rj_d_desc_perms,
            (sig_pow2 + sig * norm_ab_perms) * mat52_base_perms,
        )
        
        desc_func.d_desc_from_comp(R_d_desc_val[i, :, :], out=ri_d_desc)

        #K[blk_i, blk_j] = ri_d_desc[0].T.dot(diff_ab_outer_perms)
        np.dot(ri_d_desc[0].T, diff_ab_outer_perms, out=k)
        
        K[blk_i, blk_j] = -k
       # -----------
    
    if use_E_cstr:

        E_off = K.shape[0] - n_val, K.shape[1] - n_train
       
        blk_j_full = slice(j * dim_i, (j + 1) * dim_i)
        
        for i in range(n_val):
            blk_i = slice(i * dim_i, (i + 1) * dim_i)
            diff_ab_perms = R_desc_val[i, :] - rj_desc_perms
            norm_ab_perms = sqrt5 * np.linalg.norm(diff_ab_perms, axis=1)

            K_fe = (
                5
                * diff_ab_perms
                / (3 * sig ** 3)
                * (norm_ab_perms[:, None] + sig)
                * np.exp(-norm_ab_perms / sig)[:, None]
            )
            K_fe = np.einsum('ik,jki -> j', K_fe, rj_d_desc_perms)
            #print(dim_i)
            #blk_j_full = slice(j * dim_i, (j + 1) * dim_i)
           
            K[E_off[0] + i, blk_j] = K_fe[keep_idxs_3n]  # lower horizontal
            K[blk_i, E_off[1] + j] = K_fe  # vertical
            
            K[E_off[0] + i, E_off[1] + j] = (
                1 + (norm_ab_perms / sig) * (1 + norm_ab_perms / (3 * sig))
            ).dot(np.exp(-norm_ab_perms / sig))
            #K[E_off[0] + i, E_off[1] + j] = K[E_off[0] + j, E_off[1] + i] = (
            #    1 + (norm_ab_perms / sig) * (1 + norm_ab_perms / (3 * sig))
            #).dot(np.exp(-norm_ab_perms / sig))
    

        
    return blk_j.stop - blk_j.start

def _assemble_kernel_mat_wkr_test_E(
    j,tril_perms_lin, tril_perms_lin_mirror, sig,n_type, use_E_cstr=False, exploit_sym=False, cols_m_limit=None
):
    r"""
    ----------
        j : int
            Index of training point.
        tril_perms_lin : :obj:`numpy.ndarray`
            1D array (int) containing all recovered permutations
            expanded as one large permutation to be applied to a tiled
            copy of the object to be permuted.
    Compute one row and column of the force field kernel matrix.
    The Hessian of the Matern kernel is used with n = 2 (twice
    differentiable). Each row and column consists of matrix-valued
    blocks, which encode the interaction of one training point with all
    others. The result is stored in shared memory (a global variable).
    """
    global glob

    R_desc = np.frombuffer(glob['R_desc']).reshape(glob['R_desc_shape'])
    R_d_desc = np.frombuffer(glob['R_d_desc']).reshape(glob['R_d_desc_shape'])

    R_desc_val = np.frombuffer(glob['R_desc_val']).reshape(glob['R_desc_shape_val'])
    R_d_desc_val = np.frombuffer(glob['R_d_desc_val']).reshape(glob['R_d_desc_shape_val'])    
    
    #glob['R_desc_val'], glob['R_desc_shape_val'] = _share_array(R_desc_val_atom, 'd')
    #glob['R_d_desc_val'], glob['R_d_desc_shape_val'] = _share_array(R_d_desc_val_atom, 'd')    
 
    K = np.frombuffer(glob['K']).reshape(glob['K_shape'])
    
    desc_func = glob['desc_func']
    
    n_train, dim_d = R_d_desc.shape[:2]
    n_val, dim_d = R_d_desc_val.shape[:2]
    # dim_d =66

    n_atoms = int((1 + np.sqrt(8 * dim_d + 1)) / 2)
    dim_i = 3 * n_atoms  # 36

    n_perms = int(len(tril_perms_lin) / dim_d)
    n_perm_atom=n_perms
    
    blk_j = slice(j*dim_i , (j + 1)*dim_i )
    #blk_j = slice(j * dim_i, (j + 1) * dim_i)
    keep_idxs_3n = slice(None)  # same as [:]
  
    rj_desc_perms = np.reshape(
        np.tile(R_desc[j, :], n_perm_atom)[tril_perms_lin_mirror], (n_perm_atom, -1), order='F'
    )
    # rj_desc_perms = 12 * 66
    
    rj_d_desc = desc_func.d_desc_from_comp(R_d_desc[j, :, :])[0][
        :, keep_idxs_3n
    ]  # convert descriptor back to full representation
    # rj_d_desc 66 * 36
    
    rj_d_desc_perms = np.reshape(
        np.tile(rj_d_desc.T, n_perm_atom)[:, tril_perms_lin_mirror], (-1, dim_d, n_perm_atom)
    )
    #  rj_d_desc_perms 36 * 66 * 12
    

    sqrt5 = np.sqrt(5.0)

    #diff_ab_outer_perms_ij = np.empty((n_perm_atom,n_perm_atom,dim_d, dim_i_keep)) 
    diff_ab_perms = np.empty((n_perm_atom, dim_d))# 66 * 36
   
    for i in range(n_val):
       
        diff_ab_perms = R_desc_val[i, :] - rj_desc_perms
        norm_ab_perms = sqrt5 * np.linalg.norm(diff_ab_perms, axis=1)

        K_fe = (
            5
            * diff_ab_perms
            / (3 * sig ** 3)
            * (norm_ab_perms[:, None] + sig)
            * np.exp(-norm_ab_perms / sig)[:, None]
        )
        K_fe = np.einsum('ik,jki -> j', K_fe, rj_d_desc_perms)
        #print(dim_i)
        #blk_j_full = slice(j * dim_i, (j + 1) * dim_i)
       
        K[i, blk_j] = K_fe[keep_idxs_3n]  # lower horizontal
        
    return blk_j.stop - blk_j.start

def _assemble_kernel_E(
    j, tril_perms_lin, tril_perms_lin_mirror, sig, n_type,use_E_cstr=False, exploit_sym=False, cols_m_limit=None
):

    global glob

    R_desc_atom = np.frombuffer(glob['R_desc']).reshape(glob['R_desc_shape'])
    R_d_desc_atom = np.frombuffer(glob['R_d_desc']).reshape(glob['R_d_desc_shape'])
    K = np.frombuffer(glob['K']).reshape(glob['K_shape'])
    
    #desc_func = glob['desc_func']
    
    n_train, dim_d = R_d_desc_atom.shape[:2]
    # 600; dim_d =66

    n_atoms = int((1 + np.sqrt(8 * dim_d + 1)) / 2)
    dim_i = 3 * n_atoms  # 36
    
    #n_perms =int(len(tril_perms_lin) / (dim_d * 6))
    n_perms = int(len(tril_perms_lin) / dim_d)  #12
    n_perm_atom=n_perms
    #blk_j = slice(j*3 , (j + 1) *3)
    blk_j = slice(j * dim_i, (j + 1) * dim_i)

    rj_desc_perms = np.reshape(
        np.tile(R_desc_atom[j, :], n_perm_atom)[tril_perms_lin_mirror], (n_perm_atom, -1), order='F'
    )

    sqrt5 = np.sqrt(5.0)

    diff_ab_perms = np.empty((n_perm_atom, dim_d))# 66 * 36

    for i in range(j+1):

        diff_ab_perms = R_desc_atom[i, :] - rj_desc_perms
        norm_ab_perms = sqrt5 * np.linalg.norm(diff_ab_perms, axis=1)
        
        K[i, j] = K[j, i] = (
            1 + (norm_ab_perms / sig) * (1 + norm_ab_perms / (3 * sig))
        ).dot(np.exp(-norm_ab_perms / sig))
         
    
    return blk_j.stop - blk_j.start

def _assemble_kernel_E_test(
    j, tril_perms_lin, tril_perms_lin_mirror, sig, n_type,use_E_cstr=False, exploit_sym=False, cols_m_limit=None
):

    global glob

    R_desc_atom = np.frombuffer(glob['R_desc']).reshape(glob['R_desc_shape'])
    R_d_desc_atom = np.frombuffer(glob['R_d_desc']).reshape(glob['R_d_desc_shape'])
    
    R_desc_val = np.frombuffer(glob['R_desc_val']).reshape(glob['R_desc_shape_val'])
    R_d_desc_val = np.frombuffer(glob['R_d_desc_val']).reshape(glob['R_d_desc_shape_val'])  
    n_val, dim_d = R_d_desc_val.shape[:2]
    
    
    K = np.frombuffer(glob['K']).reshape(glob['K_shape'])
    
    #desc_func = glob['desc_func']
    
    n_train, dim_d = R_d_desc_atom.shape[:2]
    # 600; dim_d =66

    n_atoms = int((1 + np.sqrt(8 * dim_d + 1)) / 2)
    dim_i = 3 * n_atoms  # 36
    
    #n_perms =int(len(tril_perms_lin) / (dim_d * 6))
    n_perms = int(len(tril_perms_lin) / dim_d)  #12
    n_perm_atom=n_perms
    #blk_j = slice(j*3 , (j + 1) *3)
    blk_j = slice(j * dim_i, (j + 1) * dim_i)

    rj_desc_perms = np.reshape(
        np.tile(R_desc_atom[j, :], n_perm_atom)[tril_perms_lin_mirror], (n_perm_atom, -1), order='F'
    )

    sqrt5 = np.sqrt(5.0)

    diff_ab_perms = np.empty((n_perm_atom, dim_d))# 66 * 36

    for i in range(n_val):

        diff_ab_perms = R_desc_val[i, :] - rj_desc_perms
        norm_ab_perms = sqrt5 * np.linalg.norm(diff_ab_perms, axis=1)
        
        K[i, j] = (
            1 + (norm_ab_perms / sig) * (1 + norm_ab_perms / (3 * sig))
        ).dot(np.exp(-norm_ab_perms / sig))
         
    
    return blk_j.stop - blk_j.start


    

class AFFTrain(object):
    def __init__(self, max_processes=None, use_torch=False):
        global glob
        if 'glob' not in globals():  # Don't allow more than one instance of this class.
            glob = {}
        else:
            raise Exception(
                'You can not create multiple instances of this class. Please reuse your first one.'
            )

        self.log = logging.getLogger(__name__)

        self._max_processes = max_processes
        self._use_torch = use_torch

    def __del__(self):

        global glob

        if 'glob' in globals():
            del glob
            

    def draw_strat_sample(self,T,n, excl_idxs=None):
            """Draw sample from dataset that preserves its original distribution.
            The distribution is estimated from a histogram were the bin size is
            determined using the Freedman-Diaconis rule. This rule is designed to
            minimize the difference between the area under the empirical
            probability distribution and the area under the theoretical
            probability distribution. A reduced histogram is then constructed by
            sampling uniformly in each bin. It is intended to populate all bins
            with at least one sample in the reduced histogram, even for small
            training sizes.
            Parameters
            ----------
                T : :obj:`numpy.ndarray`
                    Dataset to sample from.
                n : int
                    Number of examples.
                excl_idxs : :obj:`numpy.ndarray`, optional
                    Array of indices to exclude from sample.
            Returns
            -------
                :obj:`numpy.ndarray`
                    Array of indices that form the sample.
            """
            if excl_idxs is None or len(excl_idxs) == 0:
                excl_idxs = None
    
            if n == 0:
                return np.array([], dtype=np.uint)
    
            if T.size == n:  # TODO: this only works if excl_idxs=None
                assert excl_idxs is None
                return np.arange(n)
    
            if n == 1:
                idxs_all_non_excl = np.setdiff1d(
                    np.arange(T.size), excl_idxs, assume_unique=True
                )
                return np.array([np.random.choice(idxs_all_non_excl)])
    
            # Freedman-Diaconis rule
            h = 2 * np.subtract(*np.percentile(T, [75, 25])) / np.cbrt(n)
            n_bins = int(np.ceil((np.max(T) - np.min(T)) / h)) if h > 0 else 1
            n_bins = min(
                n_bins, int(n / 2)
            )  # Limit number of bins to half of requested subset size.
    
            bins = np.linspace(np.min(T), np.max(T), n_bins, endpoint=False)
            idxs = np.digitize(T, bins)
    
            # Exclude restricted indices.
            if excl_idxs is not None and excl_idxs.size > 0:
                idxs[excl_idxs] = n_bins + 1  # Impossible bin.
    
            uniq_all, cnts_all = np.unique(idxs, return_counts=True)
    
            # Remove restricted bin.
            if excl_idxs is not None and excl_idxs.size > 0:
                excl_bin_idx = np.where(uniq_all == n_bins + 1)
                cnts_all = np.delete(cnts_all, excl_bin_idx)
                uniq_all = np.delete(uniq_all, excl_bin_idx)
    
            # Compute reduced bin counts.
            reduced_cnts = np.ceil(cnts_all / np.sum(cnts_all, dtype=float) * n).astype(int)
            reduced_cnts = np.minimum(
                reduced_cnts, cnts_all
            )  # limit reduced_cnts to what is available in cnts_all
    
            # Reduce/increase bin counts to desired total number of points.
            reduced_cnts_delta = n - np.sum(reduced_cnts)
    
            while np.abs(reduced_cnts_delta) > 0:
    
                # How many members can we remove from an arbitrary bucket, without any bucket with more than one member going to zero?
                max_bin_reduction = np.min(reduced_cnts[np.where(reduced_cnts > 1)]) - 1
    
                # Generate additional bin members to fill up/drain bucket counts of subset. This array contains (repeated) bucket IDs.
                outstanding = np.random.choice(
                    uniq_all,
                    min(max_bin_reduction, np.abs(reduced_cnts_delta)),
                    p=(reduced_cnts - 1) / np.sum(reduced_cnts - 1, dtype=float),
                    replace=True,
                )
                uniq_outstanding, cnts_outstanding = np.unique(
                    outstanding, return_counts=True
                )  # Aggregate bucket IDs.
    
                outstanding_bucket_idx = np.where(
                    np.in1d(uniq_all, uniq_outstanding, assume_unique=True)
                )[
                    0
                ]  # Bucket IDs to Idxs.
                reduced_cnts[outstanding_bucket_idx] += (
                    np.sign(reduced_cnts_delta) * cnts_outstanding
                )
                reduced_cnts_delta = n - np.sum(reduced_cnts)
    
            # Draw examples for each bin.
            idxs_train = np.empty((0,), dtype=int)
            for uniq_idx, bin_cnt in zip(uniq_all, reduced_cnts):
                idx_in_bin_all = np.where(idxs.ravel() == uniq_idx)[0]
                idxs_train = np.append(
                    idxs_train, np.random.choice(idx_in_bin_all, bin_cnt, replace=False)
                )
    
            return idxs_train

    
    def _assemble_kernel_mat(
            self,
            n_type,
            R_desc,
            R_d_desc,
            tril_perms_lin,
            tril_perms_lin_mirror,
            sig,
            desc,  # TODO: document me
            use_E_cstr=False,
            col_idxs=np.s_[:],  # TODO: document me
            callback=None,
        ):
        r"""
            Compute force field kernel matrix.
        """
        global glob
    
            # Note: This function does not support unsorted (ascending) index arrays.
            # if not isinstance(col_idxs, slice):
            #    assert np.array_equal(col_idxs, np.sort(col_idxs))
    
        n_train, dim_d = R_d_desc.shape[:2]  #R_d_desc.shape (n_train, 66, 3)
        #n_train , dim_d 66
        dim_i = 3 * int((1 + np.sqrt(8 * dim_d + 1)) / 2)  # dim = 3 * 12
    
        # Determine size of kernel matrix.
        #  **** need change
        #K_n_rows = n_train*3 #* 6   
        K_n_rows = n_train * dim_i  
        K_n_cols = n_train * dim_i   
        
        if use_E_cstr:
            K_n_rows += n_train
            K_n_cols += n_train
        #K_n_cols = n_train*3 # * 6  
        #K_n_cols = len(range(*col_idxs.indices(K_n_rows)))
        exploit_sym = False
        cols_m_limit = None
        is_M_subset = (
                isinstance(col_idxs, slice)
                and (col_idxs.start is None or col_idxs.start % dim_i == 0)
                and (col_idxs.stop is None or col_idxs.stop % dim_i == 0)
                and col_idxs.step is None
            )
        if is_M_subset:
            M_slice_start = (None if col_idxs.start is None else int(col_idxs.start / dim_i))
            M_slice_stop = None if col_idxs.stop is None else int(col_idxs.stop / dim_i)
            M_slice = slice(M_slice_start, M_slice_stop)
    
            J = range(*M_slice.indices(n_train))
    
            if M_slice_start is None:
                exploit_sym = True
                cols_m_limit = M_slice_stop
                
        #K = mp.RawArray('d', n_type * K_n_rows * K_n_cols)
        K = mp.RawArray('d',  K_n_rows * K_n_cols)
        glob['K'], glob['K_shape'] = K, (K_n_rows, K_n_cols)
        glob['R_desc_atom'], glob['R_desc_shape_atom'] = _share_array(R_desc, 'd')
        glob['R_d_desc_atom'], glob['R_d_desc_shape_atom'] = _share_array(R_d_desc, 'd')
    
        glob['desc_func'] = desc
        start = timeit.default_timer()
        pool = Pool(mp.cpu_count())
        #pool = Pool(self._max_processes)
        todo, done = K_n_cols, 0
        
        for done_wkr in pool.imap_unordered(
            partial(
                _assemble_kernel_mat_wkr,
                tril_perms_lin=tril_perms_lin,
                tril_perms_lin_mirror=tril_perms_lin_mirror,
                sig=sig,
                n_type=n_type,
                use_E_cstr=use_E_cstr,
                exploit_sym=exploit_sym,
                cols_m_limit=cols_m_limit,
            ),
            J,
        ):
            done += done_wkr
            
        pool.close()
        pool.join()  # Wait for the worker processes to terminate (to measure total runtime correctly).
        stop = timeit.default_timer()
        dur_s = (stop - start) 
        
        glob.pop('K', None)
        glob.pop('R_desc_atom', None)
        glob.pop('R_d_desc_atom', None)
    
        return np.frombuffer(K).reshape(glob['K_shape']),dur_s
    
        
    
    def create_task(self,
            train_dataset,
            n_train,
            valid_dataset,
            n_valid,
            n_test,
            sig,
            lam=1e-15,
            use_sym=True,
            use_E=True,
            use_E_cstr=False,
            use_cprsn=False,
            solver='analytic',  # TODO: document me
            solver_tol=1e-4,  # TODO: document me
            n_inducing_pts_init=25,  # TODO: document me
            interact_cut_off=None,  # TODO: document me
            callback=None,  # TODO: document me
        ):
        n_atoms = train_dataset['R'].shape[1]
        md5_train = io.dataset_md5(train_dataset)
        md5_valid = io.dataset_md5(valid_dataset)
        
        
        #if 'E' in train_dataset:
        idxs_train = self.draw_strat_sample(
                    train_dataset['E'], n_train
                )
        
        excl_idxs = (
                idxs_train if md5_train == md5_valid else np.array([], dtype=np.uint)
            )
        
        idxs_valid = self.draw_strat_sample(
                    valid_dataset['E'], n_valid, excl_idxs=excl_idxs,
                )
        
        excl_idxs1 = np.concatenate((idxs_train, idxs_valid))
        
        idxs_test = self.draw_strat_sample(
                    valid_dataset['E'], n_test, excl_idxs=excl_idxs1,
                )
        #else:
    # =============================================================================
    #             idxs_train = np.random.choice(
    #                 np.arange(train_dataset['F'].shape[0]),
    #                 n_train - m0_n_train,
    #                 replace=False,
    #             )
    # =============================================================================
        R_train = train_dataset['R'][idxs_train, :, :]
        R_val = train_dataset['R'][idxs_valid, :, :]
        R_test = train_dataset['R'][idxs_test, :, :]
        task = {
            'type': 't',
            #'code_version': __version__,
            'dataset_name': train_dataset['name'].astype(str),
            'dataset_theory': train_dataset['theory'].astype(str),
            'z': train_dataset['z'],
            'R_train': R_train,
            'R_val': R_val,
            'R_test':R_test,
            'F_train': train_dataset['F'][idxs_train, :, :],
            'F_val': train_dataset['F'][idxs_valid, :, :],
            'F_test': train_dataset['F'][idxs_test, :, :],
            'idxs_train': idxs_train,
            'md5_train': md5_train,
            'idxs_valid': idxs_valid,
            'idex_test':idxs_test,
            'md5_valid': md5_valid,
            'sig': sig,
            'lam': lam,
            'use_E': use_E,
            'use_E_cstr': use_E_cstr,
            'use_sym': use_sym,
            'use_cprsn': use_cprsn,
            'solver_name': solver,
            'solver_tol': solver_tol,
            'n_inducing_pts_init': n_inducing_pts_init,
            'interact_cut_off': interact_cut_off,
        }
        
        #if use_E:
        task['E_train'] = train_dataset['E'][idxs_train]
        task['E_val'] = train_dataset['E'][idxs_valid]
        task['E_test'] = train_dataset['E'][idxs_test]
        
        if use_sym:
                n_train = R_train.shape[0]
                R_train_sync_mat = R_train
                if n_train > 1000:
                    R_train_sync_mat = R_train[
                        np.random.choice(n_train, 1000, replace=False), :, :
                    ]
    # =============================================================================
    #                 self.log.info(
    #                     'Symmetry search has been restricted to a random subset of 1000/{:d} training points for faster convergence.'.format(
    #                         n_train
    #                     )
    # =============================================================================
                    
    # sort the permutation matrix by first column, then the mirror-wise symmetries are together
                nb = perm.find_perms(
                            R_train_sync_mat,
                            #train_dataset['R'][range(500),:,:],
                            train_dataset['z'],
                            lat_and_inv=None,
                            callback=callback,
                            max_processes=None,
                        )
                task['perms']=nb[nb[:,0].argsort()]
                #a[a[:,0].argsort()]
        else:
                task['perms'] = np.arange(train_dataset['R'].shape[1])[
                    None, :
                ]
        #task['F_train_atom']=train_dataset['F'][idxs_train, :, :]
        n_type,index_diff_atom=self.find_type(task['perms'])
        task['n_type']=n_type
        task['index_diff_atom']=index_diff_atom
        n_perms = task['perms'].shape[0]
        if use_cprsn and n_perms > 1:
    
                _, cprsn_keep_idxs = np.unique(
                    np.sort(task['perms'], axis=0), axis=1, return_index=True
                )
    
                task['cprsn_keep_atoms_idxs'] = cprsn_keep_idxs
    
        return task
    
    def find_type(self,task_perm):
        # to find out the number of different geometric atoms we have from this molecule
        # and the position of boundary
        n_perms=task_perm.shape[0]
        max_col = task_perm.max(axis=0)
        n_type= len(np.unique(max_col))
        index_diff_atom= np.empty((n_type))
        for i in range(n_type):
            index_diff_atom[i]=np.max(np.where(np.equal(max_col, np.unique(max_col)[i])))
        return n_type,index_diff_atom
    
    def create_model(
        self,
        task,
        solver,
        R_desc_atom, #R_desc_atom
        R_d_desc_atom, #R_desc_atom_atom
        tril_perms_lin, # tril_perms_lin_mirror
        std,
        alphas_F,
        alphas_E=None,
        solver_resid=None,
        solver_iters=None,
        norm_y_train=None,
        inducing_pts_idxs=None,  # NEW : which columns were used to construct nystrom preconditioner
    ):
        n_train, dim_d = R_desc_atom.shape[:2]
        n_atoms = int((1 + np.sqrt(8 * dim_d + 1)) / 2)
        
        i, j = np.tril_indices(n_atoms, k=-1)
        alphas_F_exp = alphas_F.reshape(-1, n_atoms, 3)
        
    
    def train(self, task,sig_candid_F,
            cprsn_callback=None,
            save_progr_callback=None,  # TODO: document me
            callback=None):
    
            #Train a model based on a training task.
    
        task = dict(task)
        solver = task['solver_name']
        n_train, n_atoms = task['R_train'].shape[:2]
        n_val, n_atoms = task['R_val'].shape[:2]
        desc = Desc(
                n_atoms,
                interact_cut_off=task['interact_cut_off'],
                max_processes=None,
            )
        n_perms = task['perms'].shape[0]  # 12 on benzene
        tril_perms = np.array([desc.perm(p) for p in task['perms']])

        tril_pos=task['perms']
        
        # tril_perms stores the 12 permutations on the 66 descriptor
        dim_i = 3 * n_atoms #36
        dim_d = desc.dim  #66 on benzene
        perm_offsets = np.arange(n_perms)[:, None] * dim_d
        # perm_offsets a [12,1] matrix stores the [0, 66, 66*2, ..., 12*66]

        tril_perms_lin = (tril_perms + perm_offsets).flatten('F')
        n_type=task['n_type']
        

        
        lat_and_inv = None
        R = task['R_train']  #.reshape(n_train, -1) 
        
        R_val=task['R_val'] #.reshape(n_val,-1)
        # R is a n_train * 36 matrix        
        tril_perms_lin_mirror = tril_perms_lin
        # tril_perms_lin stores a vectorized permuations of all 12 permuations' descriptor position
        
        R_atom=R
        R_val_atom=R_val
        #R_mirror=
        R_desc_atom, R_d_desc_atom = desc.from_R(R_atom,lat_and_inv=lat_and_inv,
                callback=None)
        
        R_desc_val_atom, R_d_desc_val_atom = desc.from_R(R_val_atom,lat_and_inv=lat_and_inv,
                callback=None)
        
        lam = task['lam']
       
        F_train_atom=task['F_train'].ravel().copy()
        F_val_atom=task['F_val'].ravel().copy()
        
        E_train_mean = None
        #y = task['F_train'].ravel().copy()
        y_val= F_val_atom.copy()
        
        E_val = task['E_val'].ravel().copy()
        E_val_mean = np.mean(E_val)
        #ye_val=-E_val + E_val_mean
        ye_val=E_val #- E_val_mean

        #y_std = np.std(y_atom)
        #y_atom /= y_std
        
        
        #y_c=y_atom[:,:,0]
        #y_h=y_atom[:,:,1]
        num_iters = None  # number of iterations performed (cg solver)
        resid = None  # residual of solution
        #843 line in train _py
        sig_candid=sig_candid_F
        #np.arange(10,100,10)
        #sig_candid=np.arange(10,200,30) #alkane
        num_i=sig_candid.shape[0]
        MSA=np.ones((num_i))*1e8
        #theta_hat=np.ones((num_i))
        kernel_time_all=np.zeros((num_i))
        for i in range(num_i):
            y_atom= F_train_atom.copy()
            E_train_ori = task['E_train'].ravel().copy()
            E_train_mean = np.mean(E_train_ori)
            E_train_std = np.std(E_train_ori)
            #ye_train=-E_train + E_train_mean
            ye_train=E_train_ori - E_train_mean
            
            E_train= ye_train/E_train_std
            print('This is '+repr(i)+'th task: sigma='+repr(sig_candid[i]))
            start_i = timeit.default_timer()
            
            alphas=np.empty((int(n_train*n_atoms*3),1))
            
            H=np.repeat([1],n_train)
            R=self._assemble_kernel_mat_E(
                    n_type,
                    R_desc_atom,
                    R_d_desc_atom,
                    R_desc_atom,
                    R_d_desc_atom,
                    tril_perms_lin,
                    tril_perms_lin_mirror,
                    sig_candid[i],
                    desc,
                    use_E_cstr=task['use_E_cstr'],
                    col_idxs= np.s_[:],
                    callback=None,
                )
            
            R[np.diag_indices_from(R)] += lam 
            
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                try:
                    L_R, lower_R = sp.linalg.cho_factor(
                    R, overwrite_a=True, check_finite=False
                )
                    R_inv_H=sp.linalg.cho_solve(   
                        (L_R, lower_R), H,overwrite_b=True, check_finite=False)
                    R_inv_Y=sp.linalg.cho_solve(   
                        (L_R, lower_R), E_train,overwrite_b=True, check_finite=False)
                    theta_hat=1/(H.T @ R_inv_H ) *(H.T @ R_inv_Y)
                    
                    
                    
                except np.linalg.LinAlgError: 
                    R_inv_H  = sp.linalg.solve(
                            R,  H, overwrite_a=True, overwrite_b=True, check_finite=False
                        )
                    R_inv_Y = sp.linalg.solve(
                            R,  E_train, overwrite_a=True, overwrite_b=True, check_finite=False
                        )
                    theta_hat=1/(H.T @ R_inv_H ) *(H.T @ R_inv_Y)
            
            S_square = (E_train - H * theta_hat) @ (R_inv_Y - R_inv_H * theta_hat)
            #print("S_square is ",S_square)
            r_T=self._assemble_kernel_mat_E_test(
                    n_type,
                    R_desc_atom,
                    R_d_desc_atom,
                    R_desc_val_atom,
                    R_d_desc_val_atom,
                    tril_perms_lin,
                    tril_perms_lin_mirror,
                    sig_candid[i],
                    desc,
                    use_E_cstr=task['use_E_cstr'],
                    col_idxs= np.s_[:],
                    callback=None,
                )
            alphas=R_inv_Y-R_inv_H *theta_hat
            E_pred= (np.repeat([1],n_val)*theta_hat + r_T @ (R_inv_Y-R_inv_H *theta_hat))*E_train_std  + E_train_mean
            
            
           # F_hat_val = F_hat_val1*y_std
            ae=np.mean(np.abs(E_val-E_pred))
            MSA[i]=ae
            #MSA_E=np.mean(np.abs(F_hat_val_E-ye_val))
            RMSE_E=np.sqrt(np.mean((E_val-E_pred)**2))/np.std(E_val)

            
            #print(' This is '+repr(i)+'th task: MAE of E='+repr(ae))  
            #print(' This is '+repr(i)+'th task: MAE of E='+repr(MSA_E)) 
            print(' This is '+repr(i)+'th task: RMSE of E='+repr(RMSE_E)) 
           
            if ae<=min(MSA):
                alphas_ret=alphas.copy()
                theta_opt=theta_hat
                # var = S^2/n
                var_square_opt = S_square/n_train
                #E_train_mean_opt
        kernel_time_ave=np.mean(kernel_time_all)
        print('***-----    Overall training  time: '+str(kernel_time_ave)+'seconds.  -----***')         
        alphas_opt=alphas_ret
        sig_optim=sig_candid[MSA==min(MSA)]
        #RMSE=np.sqrt(np.mean((F_hat_val-F_val_atom)**2))/np.std(F_val_atom)
        #print(' Optima task : sigma ='+repr(sig_optim)+':  MAE, RMSE='+repr(MSA[i])+' ; '+repr(RMSE))
        print(' Optima task : sigma ='+repr(sig_optim)+':  MAE='+repr(min(MSA)))
        
        trained_model = {'sig_optim':sig_optim[0], 'theta_opt':theta_opt,'alphas_opt':alphas_opt,'var_square_opt':var_square_opt,'E_train_mean':E_train_mean,'E_train_std':E_train_std}
        
        return trained_model
    
    def test(self, task,trained_model,
            cprsn_callback=None,
            save_progr_callback=None,  # TODO: document me
            callback=None):
        task = dict(task)
        sig_optim = trained_model['sig_optim']
        theta_hat = trained_model['theta_opt']
        alphas_opt = trained_model['alphas_opt']
        #solver = task['solver_name']
        n_train, n_atoms = task['R_train'].shape[:2]
        n_val, n_atoms = task['R_test'].shape[:2]
        desc = Desc(
                n_atoms,
                interact_cut_off=task['interact_cut_off'],
                max_processes=None,
            )
        n_perms = task['perms'].shape[0]  # 12 on benzene
        tril_perms = np.array([desc.perm(p) for p in task['perms']])

        #tril_pos=task['perms']
        
        # tril_perms stores the 12 permutations on the 66 descriptor
       #dim_i = 3 * n_atoms #36
        dim_d = desc.dim  #66 on benzene
        perm_offsets = np.arange(n_perms)[:, None] * dim_d
        # perm_offsets a [12,1] matrix stores the [0, 66, 66*2, ..., 12*66]
        tril_perms_lin = (tril_perms + perm_offsets).flatten('F')
        
          # tril_perms_lin stores a vectorized permuations of all 12 permuations' descriptor position
        n_type=task['n_type']
        
        lat_and_inv = None
        R = task['R_train']  #.reshape(n_train, -1) 
        
        R_val=task['R_test'] #.reshape(n_val,-1)
        # R is a n_train * 36 matrix 
        
         
     
        tril_perms_lin_mirror = tril_perms_lin
     
        R_atom=R
        R_val_atom=R_val
        #n_bl=int(n_perms/n_type)
          
        
        #R_mirror=
        R_desc_atom, R_d_desc_atom = desc.from_R(R_atom,lat_and_inv=lat_and_inv,
                callback=None)
        
        R_desc_val_atom, R_d_desc_val_atom = desc.from_R(R_val_atom,lat_and_inv=lat_and_inv,
                callback=None)
        
        #F_train_atom=task['F_train'].ravel().copy()
        #F_train_atom=task['F_train'].ravel().copy()
        #y_atom= F_train_atom
        E_val = task['E_test'].ravel().copy()
        #ye_val=E_val
        #E_val_mean = np.mean(E_val)
        #ye_val=-E_val + E_val_mean
        
        
        r_T=self._assemble_kernel_mat_E_test(
                    n_type,
                    R_desc_atom,
                    R_d_desc_atom,
                    R_desc_val_atom,
                    R_d_desc_val_atom,
                    tril_perms_lin,
                    tril_perms_lin_mirror,
                    sig_optim,
                    desc,
                    use_E_cstr=task['use_E_cstr'],
                    col_idxs= np.s_[:],
                    callback=None,
                )
            #alphas=R_inv_Y-R_inv_H *theta_hat
        #E_pred= np.repeat([1],n_val)*theta_hat + r_T @ alphas_opt
        #E_pred= theta_hat + r_T @ alphas_opt
        E_train_mean = trained_model['E_train_mean']
        E_train_std = trained_model['E_train_std']
        E_pred= (theta_hat + r_T @ alphas_opt)*E_train_std + E_train_mean
        #F_hat_val = F_hat_val1*y_std
            #np.mean(np.abs(a-F_val_atom[:,0,0]))
            
            #F_hat_val *=y_std
            
        MAE=np.mean(np.abs(E_val-E_pred))
        RMSE=np.sqrt(np.mean((E_val-E_pred)**2))/np.std(E_val)
        print(' The MAE of testing dataset : '+repr(MAE))
        print(' The RMSE/SD of testing dataset : '+repr(RMSE))
        return MAE
   
    def predict(self, task,R_predict,trained_model,
            cprsn_callback=None,
            save_progr_callback=None,  # TODO: document me
            callback=None):
        # R_predict: should be M * N * 3
        task = dict(task)
        sig_optim = trained_model['sig_optim']
        theta_hat = trained_model['theta_opt']
        alphas_opt = trained_model['alphas_opt']
        #solver = task['solver_name']
        n_train, n_atoms = task['R_train'].shape[:2]
        n_val, n_atoms = R_predict.shape[:2]
        desc = Desc(
                n_atoms,
                interact_cut_off=task['interact_cut_off'],
                max_processes=None,
            )
        n_perms = task['perms'].shape[0]  # 12 on benzene
        tril_perms = np.array([desc.perm(p) for p in task['perms']])

        #tril_pos=task['perms']
        
        # tril_perms stores the 12 permutations on the 66 descriptor
       #dim_i = 3 * n_atoms #36
        dim_d = desc.dim  #66 on benzene
        perm_offsets = np.arange(n_perms)[:, None] * dim_d
        # perm_offsets a [12,1] matrix stores the [0, 66, 66*2, ..., 12*66]
        tril_perms_lin = (tril_perms + perm_offsets).flatten('F')
        
          # tril_perms_lin stores a vectorized permuations of all 12 permuations' descriptor position
        n_type=task['n_type']
        
        lat_and_inv = None
        R = task['R_train']  #.reshape(n_train, -1) 
        
        R_val_atom=R_predict #.reshape(n_val,-1)
        # R is a n_train * 36 matrix 
        
         
     
        tril_perms_lin_mirror = tril_perms_lin
     
        R_atom=R
        #R_val_atom=R_val
        #n_bl=int(n_perms/n_type)
          
        
        #R_mirror=
        R_desc_atom, R_d_desc_atom = desc.from_R(R_atom,lat_and_inv=lat_and_inv,
                callback=None)
        
        R_desc_val_atom, R_d_desc_val_atom = desc.from_R(R_val_atom,lat_and_inv=lat_and_inv,
                callback=None)
        
        #F_train_atom=task['F_train'].ravel().copy()
        #F_train_atom=task['F_train'].ravel().copy()
        #y_atom= F_train_atom
        E_val = task['E_test'].ravel().copy()
        #ye_val=E_val
        #E_val_mean = np.mean(E_val)
        #ye_val=-E_val + E_val_mean
        
        
        r_T=self._assemble_kernel_mat_E_test(
                    n_type,
                    R_desc_atom,
                    R_d_desc_atom,
                    R_desc_val_atom[None],
                    R_d_desc_val_atom[None],
                    tril_perms_lin,
                    tril_perms_lin_mirror,
                    sig_optim,
                    desc,
                    use_E_cstr=task['use_E_cstr'],
                    col_idxs= np.s_[:],
                    callback=None,
                )
            #alphas=R_inv_Y-R_inv_H *theta_hat
        #E_pred= np.repeat([1],n_val)*theta_hat + r_T @ alphas_opt
        E_train_mean = trained_model['E_train_mean']
        E_train_std = trained_model['E_train_std']
        E_pred= (theta_hat + r_T @ alphas_opt)*E_train_std + E_train_mean
            
        #F_hat_val = F_hat_val1*y_std
            #np.mean(np.abs(a-F_val_atom[:,0,0]))
            
            #F_hat_val *=y_std
            
        MAE=np.mean(np.abs(E_val-E_pred))
        RMSE=np.sqrt(np.mean((E_val-E_pred)**2))/np.std(E_val)
        #print(' The MAE of testing dataset : '+repr(MAE))
        #print(' The RMSE/SD of testing dataset : '+repr(RMSE))
        return E_pred
    
    def predict_loss(self,task,trained_model,R_val_atom,c,E_target):
        task = dict(task)
        sig_optim = trained_model['sig_optim']
        theta_hat = trained_model['theta_opt']
        alphas_opt = trained_model['alphas_opt']
        var_square_opt = trained_model['var_square_opt']
        
        #solver = task['solver_name']
        n_train, n_atoms = task['R_train'].shape[:2]
        desc = Desc(
                n_atoms,
                interact_cut_off=task['interact_cut_off'],
                max_processes=None,
            )
        n_perms = task['perms'].shape[0]  # 12 on benzene
        tril_perms = np.array([desc.perm(p) for p in task['perms']])

        #tril_pos=task['perms']
        
        # tril_perms stores the 12 permutations on the 66 descriptor
       #dim_i = 3 * n_atoms #36
        dim_d = desc.dim  #66 on benzene
        perm_offsets = np.arange(n_perms)[:, None] * dim_d
        # perm_offsets a [12,1] matrix stores the [0, 66, 66*2, ..., 12*66]
        tril_perms_lin = (tril_perms + perm_offsets).flatten('F')
        tril_perms_lin_mirror = tril_perms_lin
        
        # tril_perms_lin stores a vectorized permuations of all 12 permuations' descriptor position
        n_type=task['n_type']
        
        lat_and_inv = None
        R_atom = task['R_train']  #.reshape(n_train, -1) 
        
        #R_val_atom=task['R_train'][ind_initial,None] + (np.random.normal(size = n_atoms*3)*1e-15).reshape(1,-1,3)
        
        
        R_desc_atom, R_d_desc_atom = desc.from_R(R_atom,lat_and_inv=lat_and_inv,
                callback=None)
        
        R_desc_val_atom, R_d_desc_val_atom = desc.from_R(R_val_atom,lat_and_inv=lat_and_inv,
                callback=None)
        
        R_desc_val_atom1=R_desc_val_atom[None]
        R_d_desc_val_atom1=R_d_desc_val_atom[None]

        #E_val = task['E_test'].ravel().copy()

        
        r_T=self._assemble_kernel_mat_E_test(
                    n_type,
                    R_desc_atom,
                    R_d_desc_atom,
                    R_desc_val_atom1,
                    R_d_desc_val_atom1,
                    tril_perms_lin,
                    tril_perms_lin_mirror,
                    sig_optim,
                    desc,
                    use_E_cstr=task['use_E_cstr'],
                    col_idxs= np.s_[:],
                    callback=None,
                )
       
           
        tem =( theta_hat + r_T @ alphas_opt)*trained_model['E_train_std'] + trained_model['E_train_mean']

        E_pred= tem

        delta=self._deltaE(R_desc_atom,
                           R_d_desc_atom,
                           R_desc_val_atom1, 
                           R_d_desc_val_atom1,
                           tril_perms_lin,
                           tril_perms_lin_mirror,
                           sig_optim,
                           n_type,
                           )
        
        R_cor=self._assemble_kernel_mat_E(
                n_type,
                R_desc_atom,
                R_d_desc_atom,
                R_desc_atom,
                R_d_desc_atom,
                tril_perms_lin,
                tril_perms_lin_mirror,
                sig_optim,
                desc,
                use_E_cstr=task['use_E_cstr'],
                col_idxs= np.s_[:],
                callback=None,
            )
        
       
        lam = task['lam']
        #lam = 1e-10
        R_cor[np.diag_indices_from(R_cor)] += lam 
        
        R_cor1 = R_cor.copy()
        L = sp.linalg.cho_factor(R_cor1, lower = True)
        
        R_inv_r = sp.linalg.cho_solve(L,r_T.T)
        correlation = 1 - r_T @ R_inv_r + lam
        predicted_var = var_square_opt * correlation
        gradient_var = -2 * c *  delta @ R_inv_r * var_square_opt
        loss = np.abs(E_pred-E_target)**2 + c * predicted_var
        gradient = 2*(E_pred-E_target)*(delta@alphas_opt).reshape(-1,1)  + gradient_var
        
        return loss,gradient,E_pred,predicted_var
    
    def inverseE_new(self, task,trained_model,E_target, ind_initial,tol_MAE, lr,c,
            num_step=10,
            cprsn_callback=None,
            save_progr_callback=None,  # TODO: document me
            callback=None):
        
        task = dict(task)
        sig_optim = trained_model['sig_optim']
        theta_hat = trained_model['theta_opt']
        alphas_opt = trained_model['alphas_opt']
        var_square_opt = trained_model['var_square_opt']
        E_train_mean= trained_model['E_train_mean']
        
        #solver = task['solver_name']
        n_train, n_atoms = task['R_train'].shape[:2]
        n_val= 1
        desc = Desc(
                n_atoms,
                interact_cut_off=task['interact_cut_off'],
                max_processes=None,
            )
        n_perms = task['perms'].shape[0]  # 12 on benzene
        tril_perms = np.array([desc.perm(p) for p in task['perms']])

        #tril_pos=task['perms']
        
        # tril_perms stores the 12 permutations on the 66 descriptor
       #dim_i = 3 * n_atoms #36
        dim_d = desc.dim  #66 on benzene
        perm_offsets = np.arange(n_perms)[:, None] * dim_d
        # perm_offsets a [12,1] matrix stores the [0, 66, 66*2, ..., 12*66]
        tril_perms_lin = (tril_perms + perm_offsets).flatten('F')
        tril_perms_lin_mirror = tril_perms_lin
        
        # tril_perms_lin stores a vectorized permuations of all 12 permuations' descriptor position
        n_type=task['n_type']
        
        lat_and_inv = None
        R = task['R_train']  #.reshape(n_train, -1) 
        
        R_val_atom=task['R_train'][ind_initial,None] + (np.random.normal(size = n_atoms*3)*1e-1).reshape(1,-1,3)
        
        R_desc_atom, R_d_desc_atom = desc.from_R(R,lat_and_inv=lat_and_inv,
                callback=None)
        
        
        # R is a n_train * 36 matrix 
        E_pred=0
        tem=0
        count = 0
        #for k in range(100):
        R_last = None
        import math
        E_pred = math.inf
        #n_step = 10
        
        E_predict_rec = []
        E_var_rec = []
        loss_rec = []
        
        #add th ebacktracking line search to find the best step size 
        t = 0.1 # adjusting parameter for ebacktracking line search:q
        
        beta = 0.1
        R_initial = task['R_train'][ind_initial,None] #+ (np.random.normal(size = n_atoms*3)*1e-15).reshape(1,-1,3)
        c_lr = lr
        prev_loss = math.inf
        R_design = []
        while count<num_step and np.abs(E_pred-E_target)>tol_MAE:
            
            current_loss, current_grad,temp1,temp2 = self.predict_loss(task, trained_model, R_initial, c, E_target)
            R_next = R_initial - current_grad.reshape(n_atoms,3) * lr
            next_loss,next_grad,temp1,temp2 = self.predict_loss(task, trained_model, R_next, c, E_target)
            #c_lr = lr
            while next_loss>current_loss + t*c_lr*(current_grad.T@ current_grad):
                print("the current learing rate is {}".format(c_lr))
                c_lr = c_lr*beta
                R_next = R_initial - current_grad.reshape(n_atoms,3) * c_lr
                next_loss,next_grad,temp1,temp2 = self.predict_loss(task, trained_model, R_next, c, E_target)
            
            current_loss, current_grad,cur_pred_mean,cur_pred_var = self.predict_loss(task, trained_model, R_initial, c, E_target)
            R_next = R_initial - current_grad.reshape(n_atoms,3) * c_lr
            
            
            #tem=tem+1
            # if tem>10:
            print('*** current_loss is',current_loss)
            print('*** predicted_mean is',cur_pred_mean)
            print('*** predicted_variance is',cur_pred_var[0][0])
            

            if not loss_rec or current_loss < min(loss_rec):
                R_best = R_initial.copy()
                loss_best = current_loss
                E_best = cur_pred_mean
            
            # stop once it converge or not decreasing any more 
            if current_loss>prev_loss or abs(current_loss-prev_loss)<tol_MAE:
                print('current loss greater than prev loss, this is the last evaluation !!!')
                break
            prev_loss = current_loss
                
            E_predict_rec.append(cur_pred_mean)
            loss_rec.append(current_loss)
            #gradient_var = -2 * c *  delta @ R_inv_r
            E_var_rec.append(cur_pred_var[0][0])
            #print('*** loss is',loss[0][0])
            #print('analystic gradient',(np.matmul(delta,alphas_opt).reshape(n_atoms,3)))
            R_initial = R_next
            
            R_design.append(R_initial)
            
            count +=1 
        return {'R_best':R_best,'E_var_rec':E_var_rec,'E_best':E_best,'E_predict_rec':E_predict_rec,'loss_rec':loss_rec,'loss_best':loss_best,'R_design':np.array(R_design)}
      
    
  
    def inverseE(self, task,trained_model,E_target, ind_initial,tol_MAE, lr,c,num_step,
            cprsn_callback=None,
            save_progr_callback=None,  # TODO: document me
            callback=None):
        
        task = dict(task)
        sig_optim = trained_model['sig_optim']
        theta_hat = trained_model['theta_opt']
        alphas_opt = trained_model['alphas_opt']
        var_square_opt = trained_model['var_square_opt']
        E_train_mean= trained_model['E_train_mean']
        
        #solver = task['solver_name']
        n_train, n_atoms = task['R_train'].shape[:2]
        n_val= 1
        desc = Desc(
                n_atoms,
                interact_cut_off=task['interact_cut_off'],
                max_processes=None,
            )
        n_perms = task['perms'].shape[0]  # 12 on benzene
        tril_perms = np.array([desc.perm(p) for p in task['perms']])

        #tril_pos=task['perms']
        
        # tril_perms stores the 12 permutations on the 66 descriptor
       #dim_i = 3 * n_atoms #36
        dim_d = desc.dim  #66 on benzene
        perm_offsets = np.arange(n_perms)[:, None] * dim_d
        # perm_offsets a [12,1] matrix stores the [0, 66, 66*2, ..., 12*66]
        tril_perms_lin = (tril_perms + perm_offsets).flatten('F')
        tril_perms_lin_mirror = tril_perms_lin
        
        # tril_perms_lin stores a vectorized permuations of all 12 permuations' descriptor position
        n_type=task['n_type']
        
        lat_and_inv = None
        R = task['R_train']  #.reshape(n_train, -1) 
        
        #R_val_atom=task['R_train'][ind_initial,None] 
        R_val_atom=task['R_train'][ind_initial,None] + (np.random.normal(size = n_atoms*3)*1e-4).reshape(1,-1,3)
        
        
        R_desc_atom, R_d_desc_atom = desc.from_R(R,lat_and_inv=lat_and_inv,
                callback=None)
        desc = Desc(
                n_atoms,
                interact_cut_off=task['interact_cut_off'],
                max_processes=None,
            )
        
        
        # R is a n_train * 36 matrix 
        E_pred=0
        tem=0
        count = 0
        #for k in range(100):
        R_last = None
        import math
        E_pred = math.inf
        n_step = num_step
        
        E_predict_rec = []
        E_var_rec = []
        R_design = []
        while count<n_step and np.abs(E_pred-E_target)>tol_MAE:
            if count == n_step-1:
                print("last step")
            count+= 1
            tril_perms_lin_mirror = tril_perms_lin
         
            R_atom=R
            #R_val_atom=R_val
            #n_bl=int(n_perms/n_type)
              
            
            #R_mirror=
            R_desc_atom, R_d_desc_atom = desc.from_R(R_atom,lat_and_inv=lat_and_inv,
                    callback=None)
            
            R_desc_val_atom, R_d_desc_val_atom = desc.from_R(R_val_atom,lat_and_inv=lat_and_inv,
                    callback=None)
            
            R_desc_val_atom1=R_desc_val_atom[None]
            R_d_desc_val_atom1=R_d_desc_val_atom[None]
    
            #E_val = task['E_test'].ravel().copy()
    
            
            r_T=self._assemble_kernel_mat_E_test(
                        n_type,
                        R_desc_atom,
                        R_d_desc_atom,
                        R_desc_val_atom1,
                        R_d_desc_val_atom1,
                        tril_perms_lin,
                        tril_perms_lin_mirror,
                        sig_optim,
                        desc,
                        use_E_cstr=task['use_E_cstr'],
                        col_idxs= np.s_[:],
                        callback=None,
                    )
           
                #alphas=R_inv_Y-R_inv_H *theta_hat
            #E_pred= np.repeat([1],n_val)*theta_hat + r_T @ alphas_opt
            
            tem =( theta_hat + r_T @ alphas_opt)*trained_model['E_train_std'] + trained_model['E_train_mean']
            
            # if np.abs(E_pred)<np.abs(tem):
            #     break
            
            E_pred= tem
            
            #if 
            E_predict_rec.append(E_pred[0])
            #tem=tem+1
            # if tem>10:
            print(E_pred)
            #     tem=0
            
            delta=self._deltaE(R_desc_atom,
                               R_d_desc_atom,
                               R_desc_val_atom1, 
                               R_d_desc_val_atom1,
                               tril_perms_lin,
                               tril_perms_lin_mirror,
                               sig_optim,
                               n_type,
                               )
            #lr=1e-12
            
            
            R_cor=self._assemble_kernel_mat_E(
                    n_type,
                    R_desc_atom,
                    R_d_desc_atom,
                    R_desc_atom,
                    R_d_desc_atom,
                    tril_perms_lin,
                    tril_perms_lin_mirror,
                    sig_optim,
                    desc,
                    use_E_cstr=task['use_E_cstr'],
                    col_idxs= np.s_[:],
                    callback=None,
                )
            
           
            lam = task['lam']
            #lam = 1e-10
            R_cor[np.diag_indices_from(R_cor)] += lam 
            
            R_cor1 = R_cor.copy()
            # *** calculate the predicted variance of the current molecular R_val_atom
            # R_inv_r = sp.linalg.solve(
            #                 R_cor1,  r_T.T, overwrite_a=True, overwrite_b=True, check_finite=False
            #             )
            
            #from scipy.linalg import cho_factor, cho_solve
            L = sp.linalg.cho_factor(R_cor1, lower = True)
            
            R_inv_r = sp.linalg.cho_solve(L,r_T.T)
            
        #     L_R, lower_R = sp.linalg.cho_factor(
        #     R_cor1, overwrite_a=True, check_finite=False
        # )
        #     R_inv_r=sp.linalg.cho_solve(   
        #         (L_R, lower_R), r_T.T,overwrite_b=True, check_finite=False)
            
           
            
            # with warnings.catch_warnings():
            #     warnings.simplefilter('ignore')
            #     try:
            #         L_R, lower_R = sp.linalg.cho_factor(
            #         R_cor, overwrite_a=True, check_finite=False
            #     )
            #         R_inv_r=sp.linalg.cho_solve(   
            #             (L_R, lower_R), r_T.T,overwrite_b=True, check_finite=False)
                    
                    
                    
                    
            #     except np.linalg.LinAlgError: 
            #         print('no chol')
            #         R_inv_r = sp.linalg.solve(
            #                 R_cor,  r_T.T, overwrite_a=True, overwrite_b=True, check_finite=False
            #             )
        #     L_R, lower_R = sp.linalg.cho_factor(
        #     R_cor, overwrite_a=True, check_finite=False
        # )
        #     R_inv_r=sp.linalg.cho_solve(   
        #         (L_R, lower_R), r_T.T,overwrite_b=True, check_finite=False)
                     
            
            correlation = 1 - r_T @ R_inv_r + lam
            #print("correlation ",correlation)
            
            predicted_var = var_square_opt * correlation
            #c = 1/lr
            #c = 0
            print('*** predicted_variance is',predicted_var[0][0])
            
            
            loss = np.abs(E_pred-E_target)**2 + c * predicted_var
            gradient_var = 2 * c *  delta @ R_inv_r
            E_var_rec.append(predicted_var[0][0])
            
            R_last = R_val_atom[0,:,:].copy()
            #print('*** loss is',loss[0][0])
            #print('analystic gradient',(np.matmul(delta,alphas_opt).reshape(n_atoms,3)))
            R_val_atom[0,:,:]= R_val_atom[0,:,:]- 2*(E_pred-E_target)*(delta@alphas_opt).reshape(n_atoms,3)*lr  - gradient_var.reshape(n_atoms,3) * lr
            #R_val_atom[0,:,:]= R_val_atom[0,:,:]- 2*(E_pred-E_target)*(delta@alphas_opt).reshape(n_atoms,3)*lr  
            
            R_design.append(R_last)
        return {'R_last':R_last,'E_var_rec':E_var_rec,'E_predict_rec':E_predict_rec,'R_design':np.array(R_design)}
        #return R_last #print(' The delta : '+repr(delta))
            
        #F_hat_val = F_hat_val1*y_std
            #np.mean(np.abs(a-F_val_atom[:,0,0]))
            
            #F_hat_val *=y_std
            
        
        
    def _deltaE(
        self,R_desc_atom,R_d_desc_atom, R_desc_val,R_d_desc_val,tril_perms_lin, tril_perms_lin_mirror, sig, n_type,use_E_cstr=False, exploit_sym=False, cols_m_limit=None
    ):
    
        global glob
        keep_idxs_3n = slice(None)
        # R_desc_atom = np.frombuffer(glob['R_desc']).reshape(glob['R_desc_shape'])
        # R_d_desc_atom = np.frombuffer(glob['R_d_desc']).reshape(glob['R_d_desc_shape'])
        
        # R_desc_val = np.frombuffer(glob['R_desc_val']).reshape(glob['R_desc_shape_val'])
        # R_d_desc_val = np.frombuffer(glob['R_d_desc_val']).reshape(glob['R_d_desc_shape_val'])  
        n_val=1
        
        
        desc_func = glob['desc_func']
        
        n_train, dim_d = R_d_desc_atom.shape[:2]
        # 600; dim_d =66
    
        n_atoms = int((1 + np.sqrt(8 * dim_d + 1)) / 2)
        dim_i = 3 * n_atoms  # 36
        
        #n_perms =int(len(tril_perms_lin) / (dim_d * 6))
        n_perms = int(len(tril_perms_lin) / dim_d)  #12
        n_perm_atom=n_perms
        #blk_j = slice(j*3 , (j + 1) *3)
        #blk_j = slice(j * dim_i, (j + 1) * dim_i)
        k = np.empty((dim_i, n_train))
        for j in range(n_train):
            
            rj_desc_perms = np.reshape(
                np.tile(R_desc_atom[j, :], n_perm_atom)[tril_perms_lin_mirror], (n_perm_atom, -1), order='F'
            )
            rj_d_desc = desc_func.d_desc_from_comp(R_d_desc_atom[j, :, :])[0][
            :, keep_idxs_3n
        ]  
    
            rj_d_desc_perms = np.reshape(
                np.tile(rj_d_desc.T, n_perm_atom)[:, tril_perms_lin_mirror], (-1, dim_d, n_perm_atom)
            )
        
            sqrt5 = np.sqrt(5.0)
            #mat52_base_div = 3 * sig ** 4
            sig_pow2 = sig ** 2
            sig_pow3 = sig ** 3
            
            dim_i_keep = rj_d_desc.shape[1]  # 36
            diff_ab_outer_perms = np.empty((dim_d, dim_i_keep))   # 66 * 36
            #diff_ab_outer_perms_ij = np.empty((n_perm_atom,n_perm_atom,dim_d, dim_i_keep)) 
            diff_ab_perms = np.empty((n_perm_atom, dim_d))# 66 * 36
           # diff_ab_perms = np.empty((n_perm_atom, dim_d))  # 12 * 66
            #ri_d_desc = np.zeros((1, dim_d, dim_i)) # must be zeros! # 1* 66 * 36
            #k = np.empty((dim_i, dim_i_keep))   # 36 * 36
            #k=np.empty((1))
            ri_d_desc = np.zeros((1, dim_d, dim_i))
            
            #index_C=int(index_atom+n_atoms/2*3*index_type ) #2=n_type
            
                
            i=0
            np.subtract(R_desc_val[i, :], rj_desc_perms, out=diff_ab_perms)
            norm_ab_perms = sqrt5 * np.linalg.norm(diff_ab_perms, axis=1)
            d=np.linalg.norm(diff_ab_perms, axis=1)
            #mat52_base_perms = np.exp(-norm_ab_perms / sig) / (-5/(3*sig_pow2)*d-5*sqrt5/(3*sig**3)*(d**2))
            mat52_base_perms = np.exp(-norm_ab_perms / sig) / (-5/(3*sig_pow2)-5*sqrt5/(3*sig_pow3)*(d))

            diff_ab_outer_perms= np.sum(diff_ab_perms * mat52_base_perms[:, None],0)
            
            
            
            
            desc_func.d_desc_from_comp(R_d_desc_val[i, :, :], out=ri_d_desc)
            
            blk_j = slice(0 , dim_i )
            #K[blk_i, blk_j] = ri_d_desc[0].T.dot(diff_ab_outer_perms)
            k[blk_j,j]=np.matmul(ri_d_desc[0].T, diff_ab_outer_perms).copy()
            
           
        
        
        return k
        
    
    def _assemble_kernel_mat_test(
            self,
            n_type,
            R_desc,
            R_d_desc,
            R_desc_val_atom,
            R_d_desc_val_atom,
            tril_perms_lin,
            tril_perms_lin_mirror,
            sig,
            desc,  # TODO: document me
            use_E_cstr=False,
            col_idxs=np.s_[:],  # TODO: document me
            callback=None,
        ):
        r"""
            Compute force field kernel matrix.
        """
        global glob
    
            # Note: This function does not support unsorted (ascending) index arrays.
            # if not isinstance(col_idxs, slice):
            #    assert np.array_equal(col_idxs, np.sort(col_idxs))
        n_val, dim_d = R_d_desc_val_atom.shape[:2]
        n_train, dim_d = R_d_desc.shape[:2]  #R_d_desc.shape (n_train, 66, 3)
        #n_train , dim_d 66
        dim_i = 3 * int((1 + np.sqrt(8 * dim_d + 1)) / 2)  # dim = 3 * 12
    
        # Determine size of kernel matrix.
        #  **** need change
        #K_n_rows = n_val *3#* 6   
        K_n_rows = n_val * dim_i   
        K_n_cols = n_train * dim_i  # * 6  
        
        if use_E_cstr:
            K_n_rows += n_val
            K_n_cols += n_train
        #K_n_cols = len(range(*col_idxs.indices(K_n_rows)))
        exploit_sym = False
        cols_m_limit = None
        is_M_subset = (
                isinstance(col_idxs, slice)
                and (col_idxs.start is None or col_idxs.start % dim_i == 0)
                and (col_idxs.stop is None or col_idxs.stop % dim_i == 0)
                and col_idxs.step is None
            )
        if is_M_subset:
            M_slice_start = (None if col_idxs.start is None else int(col_idxs.start / dim_i))
            M_slice_stop = None if col_idxs.stop is None else int(col_idxs.stop / dim_i)
            M_slice = slice(M_slice_start, M_slice_stop)
    
            #J = range(*M_slice.indices(n_train))
            J = range(*M_slice.indices(n_train))
            if M_slice_start is None:
                exploit_sym = True
                cols_m_limit = M_slice_stop
                
        K = mp.RawArray('d',   K_n_rows * K_n_cols)
        glob['K'], glob['K_shape'] = K, ( K_n_rows, K_n_cols)
        glob['R_desc'], glob['R_desc_shape'] = _share_array(R_desc, 'd')
        glob['R_d_desc'], glob['R_d_desc_shape'] = _share_array(R_d_desc, 'd')
        glob['R_desc_val'], glob['R_desc_shape_val'] = _share_array(R_desc_val_atom, 'd')
        glob['R_d_desc_val'], glob['R_d_desc_shape_val'] = _share_array(R_d_desc_val_atom, 'd')    
    
        glob['desc_func'] = desc
        start = timeit.default_timer()
        pool = Pool(None)
        #pool = Pool(self._max_processes)
        todo, done = K_n_cols, 0
        
        for done_wkr in pool.imap_unordered(
            partial(
                _assemble_kernel_mat_wkr_test,
                tril_perms_lin=tril_perms_lin,
                tril_perms_lin_mirror=tril_perms_lin_mirror,
                sig=sig,
                n_type=n_type,
                use_E_cstr=use_E_cstr,
                exploit_sym=exploit_sym,
                cols_m_limit=cols_m_limit,
            ),
            J,
        ):
            done += done_wkr
            
        pool.close()
        pool.join()  # Wait for the worker processes to terminate (to measure total runtime correctly).
        stop = timeit.default_timer()
        
        glob.pop('K', None)
        glob.pop('R_desc', None)
        glob.pop('R_d_desc', None)
        glob.pop('R_desc_val', None)
        glob.pop('R_d_desc_val', None)
    
        return np.frombuffer(K).reshape(glob['K_shape'])
    
   
    
    def _assemble_kernel_mat_E(
            self,
            n_type,
            R_desc,
            R_d_desc,
            R_desc_val_atom,
            R_d_desc_val_atom,
            tril_perms_lin,
            tril_perms_lin_mirror,
            sig,
            desc,  # TODO: document me
            use_E_cstr=False,
            col_idxs=np.s_[:],  # TODO: document me
            callback=None,
        ):
        r"""
            Compute force field kernel matrix.
        """
        global glob
    
            # Note: This function does not support unsorted (ascending) index arrays.
            # if not isinstance(col_idxs, slice):
            #    assert np.array_equal(col_idxs, np.sort(col_idxs))
        n_val, dim_d = R_d_desc_val_atom.shape[:2]
        n_train, dim_d = R_d_desc.shape[:2]  #R_d_desc.shape (n_train, 66, 3)
        #n_train , dim_d 66
        dim_i = 3 * int((1 + np.sqrt(8 * dim_d + 1)) / 2)  # dim = 3 * 12
    
        # Determine size of kernel matrix.
        #  **** need change
        #K_n_rows = n_val *3#* 6   
        K_n_rows = n_val    
        K_n_cols = n_train   # * 6  
        
        
        #K_n_cols = len(range(*col_idxs.indices(K_n_rows)))
        exploit_sym = False
        cols_m_limit = None
        is_M_subset = (
                isinstance(col_idxs, slice)
                and (col_idxs.start is None or col_idxs.start % dim_i == 0)
                and (col_idxs.stop is None or col_idxs.stop % dim_i == 0)
                and col_idxs.step is None
            )
        if is_M_subset:
            M_slice_start = (None if col_idxs.start is None else int(col_idxs.start / dim_i))
            M_slice_stop = None if col_idxs.stop is None else int(col_idxs.stop / dim_i)
            M_slice = slice(M_slice_start, M_slice_stop)
    
            #J = range(*M_slice.indices(n_train))
            J = range(*M_slice.indices(n_train))
            if M_slice_start is None:
                exploit_sym = True
                cols_m_limit = M_slice_stop
                
        K = mp.RawArray('d',   K_n_rows * K_n_cols)
        glob['K'], glob['K_shape'] = K, ( K_n_rows, K_n_cols)
        glob['R_desc'], glob['R_desc_shape'] = _share_array(R_desc, 'd')
        glob['R_d_desc'], glob['R_d_desc_shape'] = _share_array(R_d_desc, 'd')
        glob['R_desc_val'], glob['R_desc_shape_val'] = _share_array(R_desc_val_atom, 'd')
        glob['R_d_desc_val'], glob['R_d_desc_shape_val'] = _share_array(R_d_desc_val_atom, 'd')    
    
        glob['desc_func'] = desc
        #start = timeit.default_timer()
        pool = Pool(None)
        #pool = Pool(self._max_processes)
        todo, done = K_n_cols, 0
        
        for done_wkr in pool.imap_unordered(
            partial(
                _assemble_kernel_E,
                tril_perms_lin=tril_perms_lin,
                tril_perms_lin_mirror=tril_perms_lin_mirror,
                sig=sig,
                n_type=n_type,
                use_E_cstr=use_E_cstr,
                exploit_sym=exploit_sym,
                cols_m_limit=cols_m_limit,
            ),
            J,
        ):
            done += done_wkr
            
        pool.close()
        pool.join()  # Wait for the worker processes to terminate (to measure total runtime correctly).
        stop = timeit.default_timer()
        
        glob.pop('K', None)
        glob.pop('R_desc', None)
        glob.pop('R_d_desc', None)
        glob.pop('R_desc_val', None)
        glob.pop('R_d_desc_val', None)
    
        return np.frombuffer(K).reshape(glob['K_shape'])
    
    def _assemble_kernel_mat_E_test(
            self,
            n_type,
            R_desc,
            R_d_desc,
            R_desc_val_atom,
            R_d_desc_val_atom,
            tril_perms_lin,
            tril_perms_lin_mirror,
            sig,
            desc,  # TODO: document me
            use_E_cstr=False,
            col_idxs=np.s_[:],  # TODO: document me
            callback=None,
        ):
        r"""
            Compute force field kernel matrix.
        """
        global glob
    
            # Note: This function does not support unsorted (ascending) index arrays.
            # if not isinstance(col_idxs, slice):
            #    assert np.array_equal(col_idxs, np.sort(col_idxs))
        n_val, dim_d = R_d_desc_val_atom.shape[:2]
        n_train, dim_d = R_d_desc.shape[:2]  #R_d_desc.shape (n_train, 66, 3)
        #n_train , dim_d 66
        dim_i = 3 * int((1 + np.sqrt(8 * dim_d + 1)) / 2)  # dim = 3 * 12
    
        # Determine size of kernel matrix.
        #  **** need change
        #K_n_rows = n_val *3#* 6   
        K_n_rows = n_val    
        K_n_cols = n_train   # * 6  
        
        
        #K_n_cols = len(range(*col_idxs.indices(K_n_rows)))
        exploit_sym = False
        cols_m_limit = None
        is_M_subset = (
                isinstance(col_idxs, slice)
                and (col_idxs.start is None or col_idxs.start % dim_i == 0)
                and (col_idxs.stop is None or col_idxs.stop % dim_i == 0)
                and col_idxs.step is None
            )
        if is_M_subset:
            M_slice_start = (None if col_idxs.start is None else int(col_idxs.start / dim_i))
            M_slice_stop = None if col_idxs.stop is None else int(col_idxs.stop / dim_i)
            M_slice = slice(M_slice_start, M_slice_stop)
    
            #J = range(*M_slice.indices(n_train))
            J = range(*M_slice.indices(n_train))
            if M_slice_start is None:
                exploit_sym = True
                cols_m_limit = M_slice_stop
                
        K = mp.RawArray('d',   K_n_rows * K_n_cols)
        glob['K'], glob['K_shape'] = K, ( K_n_rows, K_n_cols)
        glob['R_desc'], glob['R_desc_shape'] = _share_array(R_desc, 'd')
        glob['R_d_desc'], glob['R_d_desc_shape'] = _share_array(R_d_desc, 'd')
        glob['R_desc_val'], glob['R_desc_shape_val'] = _share_array(R_desc_val_atom, 'd')
        glob['R_d_desc_val'], glob['R_d_desc_shape_val'] = _share_array(R_d_desc_val_atom, 'd')    
    
        glob['desc_func'] = desc
        #start = timeit.default_timer()
        pool = Pool(None)
        #pool = Pool(self._max_processes)
        todo, done = K_n_cols, 0
        
        for done_wkr in pool.imap_unordered(
            partial(
                _assemble_kernel_E_test,
                tril_perms_lin=tril_perms_lin,
                tril_perms_lin_mirror=tril_perms_lin_mirror,
                sig=sig,
                n_type=n_type,
                use_E_cstr=use_E_cstr,
                exploit_sym=exploit_sym,
                cols_m_limit=cols_m_limit,
            ),
            J,
        ):
            done += done_wkr
            
        pool.close()
        pool.join()  # Wait for the worker processes to terminate (to measure total runtime correctly).
        stop = timeit.default_timer()
        
        glob.pop('K', None)
        glob.pop('R_desc', None)
        glob.pop('R_d_desc', None)
        glob.pop('R_desc_val', None)
        glob.pop('R_d_desc_val', None)
    
        return np.frombuffer(K).reshape(glob['K_shape'])
    
    
def compile_scirpts_for_physics_based_calculation_IO(R_design):


    [num_molecules, num_atoms,*_] = R_design.shape

    energy_converter_script = './scripts/script_E.c'
    energy_converter_script_path = './scripts/script_E'
    force_converter_script = './scripts/script_F.c'
    cp_force_converter_script = './scripts/cp_script_F.c'
    force_converter_script_path = './scripts/script_F'

    command_compile_E_convertor = ["gcc", energy_converter_script, "-o", energy_converter_script_path]
    compile_E = subprocess.Popen(' '.join(command_compile_E_convertor), shell=True)
    compile_E.wait()


    command_cp_F_convertor = ["cp", force_converter_script, cp_force_converter_script]
    cp_F_convertor = subprocess.Popen(' '.join(command_cp_F_convertor), shell=True)
    cp_F_convertor.wait()
    replace_content = ["\"s/NUMATOM/", str(num_atoms), "/g\""]
    command_replace_num_atom_F_convertor = ["sed", "-i", ''.join(replace_content), cp_force_converter_script]
    replace_F = subprocess.Popen(' '.join(command_replace_num_atom_F_convertor), shell=True)
    replace_F.wait()
    command_compile_F_convertor = ["gcc", cp_force_converter_script, "-o", force_converter_script_path, "-lm"]
    compile_F = subprocess.Popen(' '.join(command_compile_F_convertor), shell=True)
    compile_F.wait()







# function for external physics-based calculation
def run_physics_baed_calculation(R_design, atomic_number, computational_method):

    # attention!!!
    # maybe we can change this into parameter for users to change
    # default location to read element table
    with open('./scripts/element_table', 'rb') as fp:
        element_table = pickle.load(fp)

    # no need to load R_design.npy as file any more because it is one of the inputs
    # # this command can be removed once merge into the main program
    # R_design = np.load('R_design.npy')

    # atomic number of each atom in the molecule also needs to specify
    # for example C O H H would be array([6, 8, 1, 1], dtype=uint8)
    # atomic_number = np.load('atomic_number.npy')



    # it seems the number of predicted molecules is not directly readable in the update.py
    # it seems there is no variables to record how many atoms in one molecules
    # therefore read these info from the shape of R_design
    [num_molecules, num_atoms,*_] = R_design.shape

    simulator_input_filename = "simulator_input.dat"
    simulator_output_filename = "simulator_output.dat"
    # attention!!!
    # this could be changed into a variable so that the number of cores used to run physics-based
    # calculation can be customizedß
    num_parallel_core = 32

    simulation_failed_string = 'SCF failed to converge'
    temp_energy_filename = "temp_energy.dat"
    energy_filename = "energy.dat"
    temp_force_filename = "temp_force.dat"
    force_filename = "force.dat"


    energy_converter_script_path = './scripts/script_E'
    force_converter_script_path = './scripts/script_F'


    E_new_simulation = np.zeros((num_molecules,))
    F_new_simulation = np.zeros((num_molecules,num_atoms,3))


    for index_run in range(num_molecules):
    # for index_run in range(2):

        with open(simulator_input_filename, 'wt') as input_write:
            input_write.write("$molecule\n")
            input_write.write("0 1\n")

            for index_atom in range(len(atomic_number)):
                input_write.write("%s\t" % element_table[atomic_number[index_atom]-1])

                for i_dimension in range(3):
                    # print(R_design[index_run, 0, i_dimension])
                    input_write.write("%.9lf\t" % R_design[index_run, index_atom, i_dimension])
                input_write.write("\n")
            
            # attention!!!
            # future features
            # some of the following calculation settings can be cahnged into variables
            # that allow users to make changes according to their needs
            input_write.write("$end\n")
            input_write.write("\n")
            input_write.write("$rem\n")
            input_write.write("jobtype                force\n")
            input_write.write("exchange               ")
            input_write.write(str(computational_method[0]))
            input_write.write("\n")
            input_write.write("correlation            ")
            input_write.write(str(computational_method[1]))
            input_write.write("\n")
            input_write.write("basis                  ")
            input_write.write(str(computational_method[2]))
            input_write.write("\n")
            input_write.write("SCF_CONVERGENCE 11\n")
            input_write.write("symmetry false\n")
            input_write.write("sym_ignore true\n")
            input_write.write("$end\n")




        # submit jobs to calculation
        command_run_simulation = ["qchem", "-slurm", "-nt", str(num_parallel_core), simulator_input_filename, ">", simulator_output_filename]
        run_simulator = subprocess.Popen(' '.join(command_run_simulation), shell=True)
        run_simulator.wait()

        signal_simulation_success = 1
        with open(simulator_output_filename, "rt") as read_ouput:
        # with open('failed.out', 'rt') as read_ouput:
            for line in read_ouput:
                # print(line)
                if simulation_failed_string in line:
                    signal_simulation_success = 0
                    print('simulation failed')
                    break

        if signal_simulation_success == 1:
            print('simulation succeed')

            # read energy into the matrix
            command_grep_energy = ["grep", "\"Convergence criterion met\"", simulator_output_filename, ">", temp_energy_filename]
            grep_energy = subprocess.Popen(' '.join(command_grep_energy), shell=True)
            grep_energy.wait()
            command_conv_energy = [energy_converter_script_path, temp_energy_filename, energy_filename]
            conv_energy = subprocess.Popen(' '.join(command_conv_energy), shell=True)
            conv_energy.wait()
            single_energy = np.loadtxt(energy_filename)
            os.remove(temp_energy_filename)
            os.remove(energy_filename)
            if np.any(single_energy) is False:
                print('!!!!!!!! no energy read')
            else:
                E_new_simulation[index_run] = single_energy

            if computational_method[0] == 'mp2':
                command_grep_force = ["grep", "-A",  str(math.ceil(num_atoms/6)*4), "\"Full Analytical Gradient of MP2 Energy\"", simulator_output_filename, ">", temp_force_filename]
            elif computational_method[0] == 'PBE':
                command_grep_force = ["grep", "-A",  str(math.ceil(num_atoms/6)*4), "\"Gradient of SCF Energy\"", simulator_output_filename, ">", temp_force_filename]
            grep_force = subprocess.Popen(' '.join(command_grep_force), shell=True)
            grep_force.wait()
            command_force_energy = [force_converter_script_path, temp_force_filename, force_filename]
            conv_force = subprocess.Popen(' '.join(command_force_energy), shell=True)
            conv_force.wait()
            single_force = np.loadtxt(force_filename)
            os.remove(temp_force_filename)
            os.remove(force_filename)
            if np.any(single_force) is False:
                print('!!!!!!!! no force read')
            else:
                if single_force.shape is not F_new_simulation[0,:,:]:
                    F_new_simulation[index_run,:,:] = single_force
                else:
                    print("!!!!!!! wrong dimension in force extraction")
                    print("!!!!!!! please check")



    print(E_new_simulation)
    print(F_new_simulation)

    return E_new_simulation, F_new_simulation
  

