
##!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 23:05:13 2021

@author: lihao
"""
# this version implemented the predicted energy based on the joint dist of E*, E and F*.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-


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
#os.chdir('/Users/HL/Desktop/Study/SFM')

from utils import io
#import hashlib
import numpy as np
import scipy as sp
import multiprocessing as mp
Pool = mp.get_context('fork').Pool

from solvers.analytic_train525 import Analytic
#from solvers.analytic_u_M15 import Analytic
#from solvers.analytic_u_pp import Analytic
from utils import perm
from utils.desc import Desc
from functools import partial
from scipy.stats import norm


import torch
import torch.nn as nn
import linear_operator
import subprocess
import math
import pickle



#global glob
#glob = {}
def _CUR(arr):
    return 0
    

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
    j, tril_perms_lin, tril_perms_lin_mirror, sig, index_diff_atom,use_E_cstr=False, exploit_sym=False, cols_m_limit=None
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
    n_type=len(index_diff_atom)
    # 600; dim_d =66

    n_atoms = int((1 + np.sqrt(8 * dim_d + 1)) / 2)
    dim_i = 3 * n_atoms  # 36
    #dim_ii =3 * n_atoms
     
    #n_perms =int(len(tril_perms_lin) / (dim_d * 6))
    n_perms = int(len(tril_perms_lin) / dim_d)  #12
    n_perm_atom=n_perms
    
    #blk_j = slice(j*3 , (j + 1) *3)
    #blk_j = slice(j * dim_i, (j + 1) * dim_i)

    blk_j = slice(j * dim_i, (j + 1) * dim_i)
       
    
    keep_idxs_3n = slice(None)  # same as [:]
    
    
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
    # 66 * 36
   # diff_ab_perms = np.empty((n_perm_atom, dim_d))  # 12 * 66
    #ri_d_desc = np.zeros((1, dim_d, dim_i)) # must be zeros! # 1* 66 * 36
    #k = np.empty((dim_i, dim_i_keep))   # 36 * 36
    #k=np.empty((1))
    ri_d_desc = np.zeros((1, dim_d, dim_i))
    k = np.zeros((dim_i, dim_i_keep))
    #k = np.empty((dim_i, dim_i_keep))
    
    if use_E_cstr:
        ri_d_desc = np.zeros((1, dim_d, dim_i-1))
        k = np.zeros((dim_i-1, dim_i_keep))
        diff_ab_perms_t = np.empty((n_perm_atom, dim_d))

    #index_C=int(index_atom+n_atoms/2*3*index_type ) #2=n_type
    
    #for i in range(0, j+1):
    for i in range(0, j+1):
        

        blk_i = slice(i * dim_i, (i + 1) * dim_i)
            #blk_i_full = slice(i * dim_i, (i + 1) * dim_i-1)

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
        
        #np.dot(ri_d_desc[0].T, diff_ab_outer_perms, out=k)

        tem_d=np.empty((dim_d, 3)) 
        if(n_atoms<=12):
            np.dot(ri_d_desc[0].T, diff_ab_outer_perms, out=k)
        else:
            #k1 = np.empty((3,3))
            for l in range(0, n_type):
                lenl=len(index_diff_atom[l])
                k1 = np.empty((3*lenl,3*lenl))
                index = np.tile(np.arange(3),lenl)+3*np.repeat(index_diff_atom[l],3)
                
                #index = np.arange(3)+3*l
                
                np.dot(ri_d_desc[0].T[index,:], diff_ab_outer_perms[:,index], out=k1)
                k[np.ix_(index,index)]=k1.copy()

        
        K[blk_i, blk_j] = -k
        K[blk_j, blk_i] = np.transpose(-k)
       
        
        
        
   
    return blk_j.stop - blk_j.start



def _assemble_kernel_mat_wkr_test(
    j,tril_perms_lin, tril_perms_lin_mirror, sig,index_diff_atom, use_E_cstr=False, exploit_sym=False, cols_m_limit=None
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
    n_type=len(index_diff_atom)
    
    n_train, dim_d = R_d_desc.shape[:2]
    n_val, dim_d = R_d_desc_val.shape[:2]
    # dim_d =66

    n_atoms = int((1 + np.sqrt(8 * dim_d + 1)) / 2)
    dim_i = 3 * n_atoms  # 36
    dim_ii =3 * n_atoms
    if use_E_cstr:
        dim_i=3 * n_atoms+1

    n_perms = int(len(tril_perms_lin) / dim_d)
    n_perm_atom=n_perms
    
        
    blk_j = slice(j*dim_i , (j + 1)*dim_i )
    if use_E_cstr:
       blk_j = slice(j * dim_i, (j + 1) * dim_i -1 )

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
    diff_ab_perms_t = np.empty((n_perm_atom, dim_d))
   # diff_ab_perms = np.empty((n_perm_atom, dim_d))  # 12 * 66
    #ri_d_desc = np.zeros((1, dim_d, dim_i)) # must be zeros! # 1* 66 * 36
    #k = np.empty((dim_i, dim_i_keep))   # 36 * 36
    #k=np.empty((1))
    ri_d_desc = np.zeros((1, dim_d, dim_i))
    k = np.zeros((dim_i, dim_i_keep))
    k1 = np.zeros((3, 3))
    if use_E_cstr:
        ri_d_desc = np.zeros((1, dim_d, dim_i-1))
        k = np.zeros((dim_i-1, dim_i_keep))
    #index_C=int(index_atom+n_atoms/2*3*index_type ) #2=n_type
    
    for i in range(0, n_val):
        blk_i = slice(i * dim_i, (i + 1) * dim_i)
        
        if use_E_cstr:
            blk_i = slice(i * dim_i, (i + 1) * dim_i-1)
            blk_i_full = slice(i * dim_i, (i + 1) * dim_i-1)

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
        #k1 = np.empty((3,3))
        for l in range(0, n_type):
            k1 = np.empty((3*len(index_diff_atom[l]),3*len(index_diff_atom[l])))
            index = np.tile(np.arange(3),len(index_diff_atom[l]))+3*np.repeat(index_diff_atom[l],3)
            
            #index = np.arange(3)+3*l
            
            np.dot(ri_d_desc[0].T[index,:], diff_ab_outer_perms[:,index], out=k1)
            k[np.ix_(index,index)]=k1.copy()
        
        
        
        if use_E_cstr:
            ri_desc_perms = np.reshape(
        np.tile(R_desc_val[i, :], n_perm_atom)[tril_perms_lin_mirror], (n_perm_atom, -1), order='F'
    )
            ri_d_desc_perms = np.reshape(
            np.tile(ri_d_desc[0].T, n_perm_atom)[:, tril_perms_lin_mirror], (-1, dim_d, n_perm_atom)
        )
            #diff_ab_perms = R_desc_atom[i, :] - rj_desc_perms
            #norm_ab_perms = sqrt5 * np.linalg.norm(diff_ab_perms, axis=1)

            K_fe = (
                5
                * diff_ab_perms
                / (3 * sig ** 3)
                * (norm_ab_perms[:, None] + sig)
                * np.exp(-norm_ab_perms / sig)[:, None]
            )
            K_fe = np.einsum('ik,jki -> j', K_fe, rj_d_desc_perms)
            #K[blk_i_full, (j + 1) * dim_i-1] = K_fe  # vertical
            #K[blk_j, (i + 1) * dim_i-1] = K_fe 
            
            np.subtract(R_desc[j, :], ri_desc_perms, out=diff_ab_perms_t)
            norm_ab_perms_t = sqrt5 * np.linalg.norm(diff_ab_perms_t, axis=1)
            
            K_fet = (
                5
                * diff_ab_perms_t
                / (3 * sig ** 3)
                * (norm_ab_perms_t[:, None] + sig)
                * np.exp(-norm_ab_perms_t / sig)[:, None]
            )
            
            # K_fet = (
            #     5
            #     * diff_ab_perms
            #     / (3 * sig ** 3)
            #     * (norm_ab_perms[:, None] + sig)
            #     * np.exp(-norm_ab_perms / sig)[:, None]
            # )
            K_fet = np.einsum('ik,jki -> j', K_fet, ri_d_desc_perms)
            # K_fet = np.einsum('ik,kj -> ij', K_fet, ri_d_desc[0])
            # K_fet = np.sum(K_fet,0)

            K[blk_i_full, (j + 1) * dim_i-1] = K_fet
            K[(i + 1) * dim_i-1, blk_j] = K_fe[keep_idxs_3n]  # lower horizontal
            #K[(i + 1) * dim_i-1, blk_j]
            
            #K[(i + 1) * dim_i-1, blk_j] = K_fe[keep_idxs_3n]  # lower horizontal

            # K[(i + 1) * dim_i-1, (j + 1) * dim_i-1] = (
            #     1 + (norm_ab_perms_t / sig) * (1 + norm_ab_perms_t / (3 * sig))
            # ).dot(np.exp(-norm_ab_perms_t / sig))
            
            # K[(i + 1) * dim_i-1, (j + 1) * dim_i-1] = (
            #     1 + (norm_ab_perms / sig) * (1 + norm_ab_perms / (3 * sig))
            # ).dot(np.exp(-norm_ab_perms / sig))
            
            K[(i + 1) * dim_i-1, (j + 1) * dim_i-1]  = (
            1 + (norm_ab_perms / sig) * (1 + norm_ab_perms / (3 * sig))
        ).dot(np.exp(-norm_ab_perms / sig)) 
        
        K[blk_i, blk_j] = -k
       # -----------
    
  
    

        
    return blk_j.stop - blk_j.start

def _assemble_kernel_mat_EF_wkr(
    j, tril_perms_lin, tril_perms_lin_mirror, sig, index_diff_atom,use_E_cstr=False, exploit_sym=False, cols_m_limit=None
):
    
    global glob
    

    R_desc = np.frombuffer(glob['R_desc_atom']).reshape(glob['R_desc_shape_atom'])
    #R_d_desc_atom = np.frombuffer(glob['R_d_desc_atom']).reshape(glob['R_d_desc_shape_atom'])
    R_desc_val_atom = np.frombuffer(glob['R_desc_val_atom']).reshape(glob['R_desc_shape_atom_val'])
    R_d_desc_val_atom = np.frombuffer(glob['R_d_desc_val_atom']).reshape(glob['R_d_desc_shape_atom_val'])
    desc = glob['desc_func']
    
    K = np.frombuffer(glob['K']).reshape(glob['K_shape'])
    
    keep_idxs_3n = slice(None)
    n_train, dim_d = R_desc.shape[:2]
    n_val, dim_d = R_desc_val_atom.shape[:2]
    
    n_atoms = int((1 + np.sqrt(8 * dim_d + 1)) / 2)
    dim_i = 3 * n_atoms  # 36
    
    mat52_base_div = 3 * sig ** 4
    sqrt5 = np.sqrt(5.0)
    sig_pow2 = sig ** 2
    
    R_desc_atom = np.row_stack((R_desc_val_atom,R_desc))
    R_d_desc_atom = R_d_desc_val_atom
    n_perms = int(len(tril_perms_lin) / dim_d)


    blk_j = slice(j*dim_i , (j + 1)*dim_i )
    rj_desc_perms = np.reshape(
np.tile(R_desc_atom[j, :], n_perms)[tril_perms_lin_mirror], (n_perms, -1), order='F'
)
    rj_d_desc = desc.d_desc_from_comp(R_d_desc_atom[j, :, :])[0][
            :, keep_idxs_3n
        ]  # convert descriptor back to full representation
    rj_d_desc_perms = np.reshape(
        np.tile(rj_d_desc.T, n_perms)[:, tril_perms_lin_mirror], (-1, dim_d, n_perms)
    )
    diff_ab_perms = np.empty((n_perms, dim_d))
    for i in range(n_val+n_train):
        blk_i = slice(i*dim_i , (i + 1)*dim_i )
        np.subtract(R_desc_atom[i, :], rj_desc_perms, out=diff_ab_perms)
        norm_ab_perms = sqrt5 * np.linalg.norm(diff_ab_perms, axis=1)
        
        K_fe = (
                5
                * diff_ab_perms
                / (3 * sig ** 3)
                * (norm_ab_perms[:, None] + sig)
                * np.exp(-norm_ab_perms / sig)[:, None]
            )
        K_fe = np.einsum('ik,jki -> j', K_fe, rj_d_desc_perms)
        K[i, blk_j] = K_fe[keep_idxs_3n]

def _assemble_kernel_mat_EE_wkr(
    j, tril_perms_lin, tril_perms_lin_mirror, sig, index_diff_atom,use_E_cstr=False, exploit_sym=False, cols_m_limit=None
):
    R_desc = np.frombuffer(glob['R_desc_atom']).reshape(glob['R_desc_shape_atom'])
    #R_d_desc_atom = np.frombuffer(glob['R_d_desc_atom']).reshape(glob['R_d_desc_shape_atom'])
    R_desc_val_atom = np.frombuffer(glob['R_desc_val_atom']).reshape(glob['R_desc_shape_atom_val'])
    #R_d_desc_val_atom = np.frombuffer(glob['R_d_desc_val_atom']).reshape(glob['R_d_desc_shape_atom_val'])
    #desc = glob['desc_func']
    
    K = np.frombuffer(glob['K']).reshape(glob['K_shape'])
    K_val = np.frombuffer(glob['K_val']).reshape(glob['K_val_shape'])
    
    #keep_idxs_3n = slice(None)
    n_train, dim_d = R_desc.shape[:2]
    n_val, dim_d = R_desc_val_atom.shape[:2]
    
    #n_atoms = int((1 + np.sqrt(8 * dim_d + 1)) / 2)
    #dim_i = 3 * n_atoms  # 36
    
    #mat52_base_div = 3 * sig ** 4
    sqrt5 = np.sqrt(5.0)
    #sig_pow2 = sig ** 2
    
    #R_desc_atom = np.row_stack((R_desc_val_atom,R_desc))
    
    n_perms = int(len(tril_perms_lin) / dim_d)
    rj_desc_perms = np.reshape(
        np.tile(R_desc[j, :], n_perms)[tril_perms_lin_mirror], (n_perms, -1), order='F'
    )   
    diff_ab_perms = np.empty((n_perms, dim_d))
    for i in range(j+1):
        #blk_i = slice(i*dim_i , (i + 1)*dim_i )
        np.subtract(R_desc[i, :], rj_desc_perms, out=diff_ab_perms)
        norm_ab_perms1 = sqrt5 * np.linalg.norm(diff_ab_perms, axis=1)

        K[i,j]  = (
                1 + (norm_ab_perms1 / sig) * (1 + norm_ab_perms1 / (3 * sig))
            ).dot(np.exp(-norm_ab_perms1 / sig)) 
        K[j,i]  = K[i,j].copy()
        
    diff_ab_perms = np.empty((n_perms, dim_d))
    for k in range(n_val):
        #blk_i = slice(i*dim_i , (i + 1)*dim_i )
        np.subtract(R_desc_val_atom[k, :], rj_desc_perms, out=diff_ab_perms)
        norm_ab_perms2 = sqrt5 * np.linalg.norm(diff_ab_perms, axis=1)

        K_val[k,j]  = (
                1 + (norm_ab_perms2 / sig) * (1 + norm_ab_perms2 / (3 * sig))
            ).dot(np.exp(-norm_ab_perms2 / sig)) 
    

class GDMLTrain(object):
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
            index_diff_atom,
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
        
        # if use_E_cstr:
        #     K_n_rows += n_train
        #     K_n_cols += n_train
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
                index_diff_atom=index_diff_atom,
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
            batch_size=1,
            uncertainty=False,
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
        #idxs_test=np.arange(100)
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
            'uncertainty': uncertainty,
            'sig': sig,
            'lam': lam,
            'batch_size':batch_size,
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
        # and the index of same type of atoms 
        n_perms=task_perm.shape[0]
        n_atoms=task_perm.shape[1]
        all_list=list(range(0,n_atoms))
        index_diff_atom=[]
        while len(all_list):
            index_atoms=np.unique(np.where(task_perm==all_list[0])[1])
            index_diff_atom.append(list(index_atoms))
            all_list=[ele for ele in all_list if ele not in index_atoms]
            #all_list.remove(list(index_atoms))
        n_type=len(index_diff_atom)
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
        
    
    def train(self, task,sig_candid_F,sig_candid_E,
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
        n_type = task['n_type']
        
        index_diff_atom = task['index_diff_atom']
        
        tril_perms = np.array([desc.perm(p) for p in task['perms']])

        tril_pos=task['perms']
        
        # tril_perms stores the 12 permutations on the 66 descriptor
        dim_i = 3 * n_atoms #36
        dim_d = desc.dim  #66 on benzene
        perm_offsets = np.arange(n_perms)[:, None] * dim_d
        # perm_offsets a [12,1] matrix stores the [0, 66, 66*2, ..., 12*66]

        tril_perms_lin = (tril_perms + perm_offsets).flatten('F')
    
        lat_and_inv = None
        R = task['R_train']  #.reshape(n_train, -1) 
        
        R_val=task['R_val'] #.reshape(n_val,-1)\
        lam = task['lam']
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
        F_train_atom=[]
        # if task['use_E_cstr']:
        #     #F_train_atom=np.empty((int(n_train*(n_atoms/n_type*3+1)),n_type))
            
        # else:
        #     F_train_atom=np.empty((int(n_train*n_atoms/n_type*3),n_type))
        F_val_atom=[]
            #F_val_atom=task['F_val'].ravel().copy()
        
        E_train = task['E_train'].ravel().copy()
        E_val = task['E_val'].ravel().copy()
        
        for i in range(n_type):
            index=np.array(index_diff_atom[i])

            F_train_atom.append(task['F_train'][:,index,:].reshape(int(n_train*(len(index_diff_atom[i])*3)),order='C'))
            F_val_atom.append(task['F_val'][:,index,:].reshape(int(n_val*(len(index_diff_atom[i])*3)),order='C'))

        #F_train_atom=task['F_train'].ravel().copy()
        #F_val_atom=task['F_val'].ravel().copy()
        
        #E_train_mean = None
        #y = task['F_train'].ravel().copy()
        #y_val= F_val_atom.copy()
        
        ye_val=E_val #- E_val_mean

        #y_std = np.std(y_atom)
        #y_atom /= y_std
        
        
        #sig_candid=np.arange(100,250,40)
        #sig_candid=np.arange(100,300,30) #alkane
        sig_candid=sig_candid_F
        #np.arange(10,100,10) #naphthalene_dft
        #sig_candid1=sig_candid
        sig_candid1=sig_candid_E
        #np.arange(16,35,2)#naphthalene_dft
        #sig_candid1=np.arange(6,15,1)  # this is for energy prediction of uracil
        #sig_candid1=np.arange(6,15,1)  # this is for energy prediction of aspirn
        num_i=sig_candid.shape[0]
        MSA=np.ones((num_i))*1e8
        #RMSE=np.empty((num_i))*1e8
        kernel_time_all=np.zeros((num_i))
        
        for i in range(num_i):
            y_atom= F_train_atom.copy()
            
            
            print('This is '+repr(i)+'th task: sigma='+repr(sig_candid[i]))
            
            if solver == 'analytic':
                #gdml_train: analytic = Analytic(gdml_train, desc, callback=None)
        
                analytic = Analytic(self, desc, callback=callback)
                #alphas = analytic.solve(task, R_desc, R_d_desc, tril_perms_lin, tril_perms_lin_mirror, y)
                alphas, inverse_time, kernel_time = analytic.solve_xyz(task,sig_candid[i], R_desc_atom, R_d_desc_atom, tril_perms_lin, tril_perms_lin_mirror, y_atom)
                #print('Kernel Processing time: '+str(kernel_time)+'seconds')
                #print('training time:   '+str(inverse_time)+'seconds')
                kernel_time_all[i]=inverse_time+kernel_time
            
            # if task['use_E_cstr']:
            #     alphas_E = alphas[-n_train:]
            #     alphas_F = alphas[:-n_train]
            F_hat_val=[]
            F_hat_val_F=[]
            F_hat_val_E=[]
            # if task['use_E_cstr']:
            #     F_hat_val=np.empty((int(n_val*(n_atoms/n_type*3+1)),n_type)) 
            #     F_hat_val_F=np.empty((int(n_val*n_atoms/n_type*3),n_type)) 
            #     F_hat_val_E=np.empty((int(n_val),n_type))
            # else:
            #     F_hat_val=np.empty((int(n_val*(n_atoms/n_type*3)),n_type))
            #     F_hat_val_F=np.empty((int(n_val*n_atoms/n_type*3),n_type)) 
            
            
            K_r_all = self._assemble_kernel_mat_test(
                    index_diff_atom,
                    R_desc_atom,
                    R_d_desc_atom,
                    R_desc_val_atom,
                    R_d_desc_val_atom,
                    tril_perms_lin,
                    tril_perms_lin_mirror,
                    sig_candid[i],
                    desc,
                    use_E_cstr=False,
                    col_idxs= np.s_[:],
                    callback=None,
                )
            
            K_all,tem=self._assemble_kernel_mat(
                index_diff_atom,
                R_desc_atom,
                R_d_desc_atom,
                tril_perms_lin,
                tril_perms_lin_mirror,
                sig_candid[i],
                desc,
                use_E_cstr=False,
                col_idxs=np.s_[:],
                callback=None,
               
            )
            
            F_star=np.empty([n_val*3*n_atoms])
            S2 = []
            for ind_i in range(n_type):
                index_eg=np.tile(np.arange(3),len(index_diff_atom[ind_i]))+3*np.repeat(index_diff_atom[ind_i],3)

                index_x=np.repeat(np.arange(n_val)*(dim_i),3*len(index_diff_atom[ind_i]))+np.tile(index_eg,n_val)
                index_y=np.repeat(np.arange(n_train)*(dim_i),3*len(index_diff_atom[ind_i]))+np.tile(index_eg,n_train)

                    # index_x=np.repeat(np.arange(n_val)*dim_i,3)+np.tile(np.array([3*ind_i,3*ind_i+1,3*ind_i+2]),n_val)
                    # index_y=np.repeat(np.arange(n_train)*dim_i,3)+np.tile(np.array([3*ind_i,3*ind_i+1,3*ind_i+2]),n_train)
                
                K_r=K_r_all[np.ix_(index_x,index_y)]
                F_hat_val_i=np.matmul(K_r,np.array(alphas[ind_i]))
                index_i=np.repeat(np.arange(n_val)*(dim_i),3*len(index_diff_atom[ind_i]))+np.tile(index_eg,n_val)


                F_hat_val_F.append(F_hat_val_i)
                F_star[index_i]=F_hat_val_i
                
                R=K_all[np.ix_(index_y,index_y)]
                R[np.diag_indices_from(R)] += lam
                L, lower = sp.linalg.cho_factor(
                            R, overwrite_a=True, check_finite=False
                        )
                R_inv_r = sp.linalg.cho_solve(
                            (L, lower), K_r.T
                        )
                #R_inv_r = np.matmul(np.linalg.inv(R),  K_r.T)
                #K_star_star=K_r_all_val[np.ix_(index_x,index_x)]-np.matmul(K_r,R_inv_r)
                S2.append(np.matmul(np.array(F_train_atom[ind_i]),np.array(alphas[ind_i]))/(n_train*3*len(index_diff_atom[ind_i])))

                
                    
                
       
            
            #F_hat_val=np.matmul(K_r_all,alphas)
            

            #F_hat_val_E=(-np.matmul(K_r_all_e,alphas))-c
            # F_hat_val = F_hat_val1*y_std
            ae=np.mean(np.abs(  np.concatenate(F_hat_val_F)-np.concatenate(F_val_atom)))
            print(' This is '+repr(i)+'th task: MAE of F='+repr(ae)) 
            #ae=np.mean(np.abs(F_hat_val_F-F_val_atom))
            MSA[i]=ae
            

           
            if ae<=min(MSA):
                alphas_ret=alphas.copy()
                F_star_opt=F_star.copy()
                S2_optim = S2.copy()
                
        kernel_time_ave=np.mean(kernel_time_all)
        print('***-----    Overall training  time: '+str(kernel_time_ave)+'seconds.  -----***')        
                
        alphas_opt=alphas_ret
        sig_optim=sig_candid[MSA==min(MSA)]
        lam_e = task['lam']
        lam_f=task['lam']
        sig_optim_E=0
        #sig_candid1=np.arange(15,18,2)
        if task['use_E_cstr']:
            print('   Starting training for energy:    ') 
            E_time_all=np.zeros((len(sig_candid1)))
            batch_size=task['batch_size']
            MSA_E_arr=[]
            MSA_E_arr_o=[]
            for i in range(len(sig_candid1)):
                #print('This is '+repr(i)+'th task: sigma='+repr(sig_candid[i]))
                start_e = timeit.default_timer()  
                F_hat_val_E_ave,E_L,E_H=self._assemble_kernel_mat_Energy(
                    index_diff_atom,
                    R_desc_atom,
                    R_d_desc_atom,
                    R_desc_val_atom,
                    R_d_desc_val_atom,
                    tril_perms_lin,
                    tril_perms_lin_mirror,
                    sig_candid1[i],
                    lam_e,
                    lam_f,
                    batch_size,
                    desc,
                    F_star_opt,#task['F_val'].ravel().copy(),#
                    E_train,
                    use_E_cstr=task['use_E_cstr'],
                    col_idxs= np.s_[:],
                    callback=None,
                )
                stop_e = timeit.default_timer()  
                E_time_all[i] = stop_e-start_e
                
                MSA_E=np.mean(np.abs(-F_hat_val_E_ave-ye_val))
                RMSE_E=np.sqrt(np.mean((-F_hat_val_E_ave-ye_val)**2))/np.std(ye_val)
                MSA_E_arr.append(MSA_E)
                print('This is '+repr(i)+'th task: sigma='+repr(sig_candid1[i]))
                print(' This is '+repr(i)+'th task: MAE of E='+repr(MSA_E)) 
                print(' This is '+repr(i)+'th task: RMSE of E='+repr(RMSE_E)) 
                
                
            sig_optim_E=sig_candid1[MSA_E_arr==np.min(MSA_E_arr)]
            print('**-----    Overall training time on E: '+str(np.mean(E_time_all))+'seconds.  -----***')  
        
        #RMSE=np.sqrt(np.mean((F_hat_val_F-F_val_atom)**2))/np.std(F_val_atom)
        #print(' Optima task : sigma ='+repr(sig_optim)+':  MAE, RMSE='+repr(MSA[i])+' ; '+repr(RMSE))
        print(' Optima task : sigma ='+repr(sig_optim)+':  MAE of F ='+repr(min(MSA)))
        if task['use_E_cstr']:
            print(' Optima task : sigma of e ='+repr(sig_optim_E)+':  MAE of E ='+repr(np.min(MSA_E_arr)))
        trained_model = {'sig_optim':sig_optim[0], 'sig_optim_E':sig_optim_E,
                         'kernel_time_ave':kernel_time_ave,'alphas_opt':alphas_opt,'S2_optim':S2_optim}
        
        return trained_model
        #return sig_optim[0],sig_optim_E,alphas_opt,kernel_time_ave,S2_optim
    
    def test(self, task,trained_model,
            cprsn_callback=None,
            save_progr_callback=None,  # TODO: document me
            callback=None):
        
        sig_optim= trained_model['sig_optim']
        sig_candid1_opt= trained_model['sig_candid1_opt']
        alphas_opt= trained_model['alphas_opt']
        kernel_time_ave= trained_model['kernel_time_ave']
        
        task = dict(task)
        solver = task['solver_name']
        batch_size=task['batch_size']
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
        index_diff_atom = task['index_diff_atom']
        # tril_perms stores the 12 permutations on the 66 descriptor
        dim_i = 3 * n_atoms #36
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
        # tril_perms_lin stores a vectorized permuations of all 12 permuations' descriptor position
        
        R_atom=R
        R_val_atom=R_val
        #R_mirror=
        R_desc_atom, R_d_desc_atom = desc.from_R(R_atom,lat_and_inv=lat_and_inv,
                callback=None)
        
        R_desc_val_atom, R_d_desc_val_atom = desc.from_R(R_val_atom,lat_and_inv=lat_and_inv,
                callback=None)
        F_train_atom=[]
        # if task['use_E_cstr']:
        #     #F_train_atom=np.empty((int(n_train*(n_atoms/n_type*3+1)),n_type))
            
        # else:
        #     F_train_atom=np.empty((int(n_train*n_atoms/n_type*3),n_type))
        F_val_atom=[]
            #F_val_atom=task['F_val'].ravel().copy()
        
        E_train = task['E_train'].ravel().copy()
        E_val = task['E_test'].ravel().copy()
        uncertainty=task['uncertainty']
        
        for i in range(n_type):
            index=np.array(index_diff_atom[i])

            F_train_atom.append(task['F_train'][:,index,:].reshape(int(n_train*(len(index_diff_atom[i])*3)),order='C'))
            F_val_atom.append(task['F_test'][:,index,:].reshape(int(n_val*(len(index_diff_atom[i])*3)),order='C'))
        ye_val=E_val
        
        #y_atom= F_train_atom.copy()
            
            
        print('This is tesing task : sigma='+repr(sig_optim))
        alphas=alphas_opt
        

        F_hat_val_F=[]
        #F_hat_val_E=[]

        K_r_all = self._assemble_kernel_mat_test(
                index_diff_atom,
                R_desc_atom,
                R_d_desc_atom,
                R_desc_val_atom,
                R_d_desc_val_atom,
                tril_perms_lin,
                tril_perms_lin_mirror,
                sig_optim,
                desc,
                use_E_cstr=False,
                col_idxs= np.s_[:],
                callback=None,
            )
        
        if uncertainty:
            K_r_all_val,tem=self._assemble_kernel_mat(
                index_diff_atom,
                R_desc_val_atom,
                R_d_desc_val_atom,
                tril_perms_lin,
                tril_perms_lin_mirror,
                sig_optim,
                desc,
                use_E_cstr=False,
                col_idxs=np.s_[:],
                callback=None,
               
            )

            K_all,tem=self._assemble_kernel_mat(
                index_diff_atom,
                R_desc_atom,
                R_d_desc_atom,
                tril_perms_lin,
                tril_perms_lin_mirror,
                sig_optim,
                desc,
                use_E_cstr=False,
                col_idxs=np.s_[:],
                callback=None,
               
            )

            
        
        F_star=np.empty([n_val*3*n_atoms])
        F_star_L=[]
        F_star_H=[]
        for ind_i in range(n_type):
            index_eg=np.tile(np.arange(3),len(index_diff_atom[ind_i]))+3*np.repeat(index_diff_atom[ind_i],3)


                
            index_x=np.repeat(np.arange(n_val)*(dim_i),3*len(index_diff_atom[ind_i]))+np.tile(index_eg,n_val)
            index_y=np.repeat(np.arange(n_train)*(dim_i),3*len(index_diff_atom[ind_i]))+np.tile(index_eg,n_train)

 
            K_r=K_r_all[np.ix_(index_x,index_y)]
            F_hat_val_i=np.matmul(K_r,np.array(alphas[ind_i]))
            index_i=np.repeat(np.arange(n_val)*(dim_i),3*len(index_diff_atom[ind_i]))+np.tile(index_eg,n_val)


            F_hat_val_F.append(F_hat_val_i)
            F_star[index_i]=F_hat_val_i
            
            if uncertainty:
                from scipy.stats import norm
                lam=task['lam']
                R=K_all[np.ix_(index_y,index_y)]
                R[np.diag_indices_from(R)] += lam
                L, lower = sp.linalg.cho_factor(
                            R, overwrite_a=True, check_finite=False
                        )
                R_inv_r = sp.linalg.cho_solve(
                            (L, lower), K_r.T
                        )
                #R_inv_r = np.matmul(np.linalg.inv(R),  K_r.T)
                K_star_star=K_r_all_val[np.ix_(index_x,index_x)]-np.matmul(K_r,R_inv_r)
                S2=np.matmul(np.array(F_train_atom[ind_i]),np.array(alphas[ind_i]))/(n_train*3*len(index_diff_atom[ind_i]))
                
                F_star_L.append(F_hat_val_i+norm.ppf(0.025,0,np.sqrt(np.diag(S2*K_star_star))))
                #F_star_L.append(F_hat_val_i+norm.ppf(0.025,0,S2*K_star_star))
                #norm.ppf(0.025)*np.sqrt(S2*np.diag(K_star_star)))
                #qnorm(0.025,mean=0,sd=sqrt(S2_testing))
                F_star_H.append(F_hat_val_i+norm.ppf(0.975,0,np.sqrt(np.diag(S2*K_star_star))))
        
        if uncertainty: 
            P_C=np.sum( np.array(np.concatenate(F_star_L)<=np.concatenate(F_val_atom)) & np.array(np.concatenate(F_star_H)>=np.concatenate(F_val_atom)))/(n_val*3*n_atoms)
            A_L=np.mean( np.concatenate(F_star_H)-np.concatenate(F_star_L))
            print(' The percentage of coverage by 95% CI is '+repr(P_C*100)+' %.') 
            print(' The average length of the 95% CI is '+repr(A_L)) 
        #a=(np.concatenate(F_star_L)<=np.concatenate(F_val_atom) and np.concatenate(F_star_H)>=np.concatenate(F_val_atom))
        ae=np.mean(np.abs(  np.concatenate(F_hat_val_F)-np.concatenate(F_val_atom)))
        RMSE_F=np.sqrt(np.mean((np.concatenate(F_hat_val_F)-np.concatenate(F_val_atom))**2))/np.std(np.concatenate(F_val_atom))
        print(' This is the  MAE of F='+repr(ae)) 
        print(' This is the  RMSE/std of F='+repr(RMSE_F)) 
        print(' This is the testing task: RMSE of F='+repr(np.sqrt(np.mean((np.concatenate(F_hat_val_F)-np.concatenate(F_val_atom))**2)))) 
           
        
        #if uncertainty:
          # P_cover=
        
        
        #ae=np.mean(np.abs(F_hat_val_F-F_val_atom))
        MAE=ae
        
        if task['use_E_cstr']:
            lam_f=task['lam']
            lam_e=task['lam']
            print('   Starting training for energy:    ') 
            MSA_E_arr=[]
            start = timeit.default_timer()
        
                #print('This is '+repr(i)+'th task: sigma='+repr(sig_candid[i]))
            F_hat_val_E_ave,E_L,E_H=self._assemble_kernel_mat_Energy(
                index_diff_atom,
                R_desc_atom,
                R_d_desc_atom,
                R_desc_val_atom,
                R_d_desc_val_atom,
                tril_perms_lin,
                tril_perms_lin_mirror,
                sig_candid1_opt,
                lam_e,
                lam_f,
                batch_size,
                desc,
                F_star,#F_star_opt,
                E_train,
                uncertainty,
                use_E_cstr=task['use_E_cstr'],
                col_idxs= np.s_[:],
                callback=None,
            )
            stop = timeit.default_timer()
            dur_s = (stop - start)
            MSA_E=np.mean(np.abs(-F_hat_val_E_ave-ye_val))
            RMSE_E=np.sqrt(np.mean((-F_hat_val_E_ave-ye_val)**2))/np.std(ye_val)
            MSA_E_arr.append(MSA_E)
            print(' *** Overall training time on E:'+repr(dur_s+kernel_time_ave)+'second; each estimation takes'+repr(dur_s/n_val)+' seconds') 
            print(' This is the testing task: MAE of E='+repr(MSA_E)) 
            print(' This is the testing task: RMSE /stdof E='+repr(RMSE_E)) 
            print(' This is the testing task: RMSE of E='+repr(np.sqrt(np.mean((-F_hat_val_E_ave-ye_val)**2)))) 
            if uncertainty: 
                P_C=np.sum( np.array((E_L)<=(-ye_val)) & np.array((E_H)>=(-ye_val)))/(n_val)
                A_L=np.mean( (E_H)-(E_L))
                print(' Energy: The percentage of coverage  by 95% CI is '+repr(P_C*100)+' %.') 
                print(' Energy: The average length of the 95% CI is '+repr(A_L)) 
       
        
        
       
        return MAE
    
    def correlation_matrix(self,R_val,task,sig,tril_perms_lin,lam):
        
        R_atom = task['R_train']
        n_type=task['n_type']
        n_train, n_atoms = task['R_train'].shape[:2]
        desc = Desc(
                n_atoms,
                interact_cut_off=task['interact_cut_off'],
                max_processes=None,
            )
        R_all = np.append(R_atom,R_val[None]).reshape(n_train+1,n_atoms,-1)
        
        R_desc, R_d_desc = desc.from_R(R_all,lat_and_inv=None,
                callback=None)
        
        #R_desc_val, R_d_desc_val = desc.from_R(R_val[None],lat_and_inv=None,
        #        callback=None)
        index_diff_atom = task['index_diff_atom']
        K,dur_s = self._assemble_kernel_mat(
                index_diff_atom,
                R_desc,
                R_d_desc,
                tril_perms_lin,
                tril_perms_lin,
                sig,
                desc,
                use_E_cstr=False,
                col_idxs=np.s_[:],
                callback=None,
               
            )
        
        K[np.diag_indices_from(K)] += lam
        
        R = K[0:(3*n_atoms*n_train),0:(3*n_atoms*n_train)]
        
        k_star = K[(3*n_atoms*n_train):(3*n_atoms*(n_train+1)),(3*n_atoms*n_train):(3*n_atoms*(n_train+1))]
        
        # K_star,dur_s = self._assemble_kernel_mat(
        #         n_type,
        #         R_desc_val,
        #         R_d_desc_val,
        #         tril_perms_lin,
        #         tril_perms_lin,
        #         sig,
        #         desc,
        #         use_E_cstr=False,
        #         col_idxs=np.s_[:],
        #         callback=None,
               
        #     )
        
        # K_star[np.diag_indices_from(K_star)] += lam
        return torch.from_numpy(R).type(torch.float64),torch.from_numpy(k_star).type(torch.float64)
    
    def predicted_F_tensor(self,k_val_train,alpha_t,task):
        '''
        

        Parameters
        ----------
        k_val_train : TYPE
            DESCRIPTION.
        alpha_t : list of tensor, 
            alpha_t[i] is the alpha[ith-type]
        task : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''

        
        

        
        
        n_type=task['n_type']
        index_diff_atom = task['index_diff_atom']
        
        n_atoms = task['R_train'].shape[1]
        n_val = 1
        n_train = task['R_train'].shape[0]
        dim_i = 3 * n_atoms
        F_hat_val = []
        
        for ind_i in range(n_type):
            
            index_eg=np.tile(np.arange(3),len(index_diff_atom[ind_i]))+3*np.repeat(index_diff_atom[ind_i],3)

            
            index_x=np.repeat(np.arange(n_val)*(dim_i),3*len(index_diff_atom[ind_i]))+np.tile(index_eg,n_val)
            index_y=np.repeat(np.arange(n_train)*(dim_i),3*len(index_diff_atom[ind_i]))+np.tile(index_eg,n_train)

            K_r=k_val_train[np.ix_(index_x,index_y)]
            F_hat_val_i=torch.mm(K_r,alpha_t[ind_i])
            F_hat_val.append(F_hat_val_i)
            
        return F_hat_val
        
       
        
        
        
        
        
        
        
        
    def loss(self,R_val_tensor, k_val_train,alpha_t,R,c_star,sigma_2_hat,F_predict,c,task):
        """
        find the loss of R_val_tensor predicted by the  AFF method

        Parameters
        ----------
        R_val_tensor : TYPE
            DESCRIPTION.
        k_val_train : TYPE
            DESCRIPTION.
        alpha_t : TYPE
            DESCRIPTION.
        R : TYPE
            DESCRIPTION.
        c_star : TYPE
            DESCRIPTION.
        sigma_2_hat : TYPE
            DESCRIPTION.
        F_predict : list of F_predict
            DESCRIPTION.
        c : TYPE
            DESCRIPTION.

        Returns
        -------
        loss : TYPE
            DESCRIPTION.

        """
        # find the loss of R_val_tensor predicted by the  AFF method
        
        #u = torch.linalg.cholesky(R)
        #R_inv_rT = torch.cholesky_solve(R_val_tensor.t(), u)
        n_type=task['n_type']
        index_diff_atom = task['index_diff_atom']
        
        n_atoms =task['R_train'].shape[1]
        n_val = 1
        n_train = task['R_train'].shape[0]
        dim_i = 3 * n_atoms
        
        loss = 0
        l1 = 0
        l2 = 0
        
        for ind_i in range(n_type):
            
            index_eg=np.tile(np.arange(3),len(index_diff_atom[ind_i]))+3*np.repeat(index_diff_atom[ind_i],3)

            
            index_x=np.repeat(np.arange(n_val)*(dim_i),3*len(index_diff_atom[ind_i]))+np.tile(index_eg,n_val)
            index_y=np.repeat(np.arange(n_train)*(dim_i),3*len(index_diff_atom[ind_i]))+np.tile(index_eg,n_train)

            R_ind_i = R[np.ix_(index_y,index_y)]
            R_val_tensor_ind_i = R_val_tensor[np.ix_(index_x,index_y)]
            
            
            R_inv_rT = linear_operator.solve(R_ind_i, rhs= R_val_tensor_ind_i.T, lhs= R_val_tensor_ind_i)
            #R_inv_rT = torch.mm ( torch.cholesky_inverse(u), R_val_tensor.t())
        
        # LU, pivots = torch.linalg.lu_factor(R)
        # R_inv_rT = torch.lu_solve(R_val_tensor.t(), LU, pivots)
        #R_inv_rT = torch.mm(torch.linalg.inv(R), R_val_tensor.t())
        
            c_star_ind_i = c_star[np.ix_(index_x,index_x)]
        
        
            predicted_variance = torch.sub(c_star_ind_i , R_inv_rT) * sigma_2_hat[ind_i] 
        
        #predicted_variance = torch.sub(c_star , torch.mm(R_val_tensor,R_inv_rT)) * sigma_2_hat 
        
         
            l1 += torch.norm(F_predict[ind_i])**2 
            l2 += c *torch.trace(predicted_variance)
            #loss += (l1 + l2)
            #print(' mean loss  '+repr(l1)+'variance loss  '+repr(l2))  
            #print(' mean loss  '+repr(loss))  
       # print('variance',l2)
        
        loss = l1 + l2
        #print(l2)
        #print(' mean loss  '+repr(l1)+'variance loss  '+repr(l2/c))  
        #print(' mean loss  '+repr(loss))  
        return loss
        
        
        
        
    
    def inverse(self,task,trained_model,initial=0,c = 1, n_iter = 30,random_noise = 1e-2):
        
        sig_optim= trained_model['sig_optim']
        alphas_opt= trained_model['alphas_opt']
        sigma_2_hat= trained_model['S2_optim']
        #kernel_time_ave= trained_model['kernel_time_ave']
        
        n_train, n_atoms = task['R_train'].shape[:2]
        n_val, n_atoms = task['R_val'].shape[:2]
        n_perms = task['perms'].shape[0] 
        desc = Desc(
                n_atoms,
                interact_cut_off=task['interact_cut_off'],
                max_processes=None,
            )
        
        tril_perms = np.array([desc.perm(p) for p in task['perms']])

        #tril_pos=task['perms']
        
        # tril_perms stores the 12 permutations on the 66 descriptor
        dim_i = 3 * n_atoms #36
        dim_d = desc.dim  #66 on benzene
        perm_offsets = np.arange(n_perms)[:, None] * dim_d
        # perm_offsets a [12,1] matrix stores the [0, 66, 66*2, ..., 12*66]

        tril_perms_lin = (tril_perms + perm_offsets).flatten('F')
        #(np.random.normal(size = n_atoms*3)*1e-4).reshape(1,-1,3)
        np.random.seed(10)
        R_val = task['R_train'][initial,:,:] + (np.random.normal(size = n_atoms*3)*random_noise).reshape(-1,3)
        R_val_tensor = torch.from_numpy(R_val).type(torch.float64)
        R_train_tensor = torch.from_numpy(task['R_train']).type(torch.float64)
        index_diff_atom = task['index_diff_atom']
        
        k_val_train = self.torch_correlation_val_train(R_val_tensor, R_train_tensor, sig_optim,tril_perms_lin,index_diff_atom)
        
        
        n_type=task['n_type']
        alpha_t = []
        for ind in range(n_type):
            alpha_t.append(torch.from_numpy(alphas_opt[ind]).type(torch.float64).reshape(-1,1))
            
        #alpha_t = torch.from_numpy(alphas_opt).type(torch.float64).reshape(-1,1)
        F_predict = self.predicted_F_tensor(k_val_train, alpha_t, task)
        #torch.mm(k_val_train,alpha_t)
        
        R_val_tensor.requires_grad_()
        optimizer = torch.optim.Adam([R_val_tensor], lr=1e-3)
        #optimizer = torch.optim.SGD([R_val_tensor], lr=1e-4, momentum=0.9)
        step_size = 1e-12
        for index in range(n_iter):
            optimizer.zero_grad()
            
            k_val_train = self.torch_correlation_val_train(R_val_tensor, R_train_tensor, sig_optim,tril_perms_lin,index_diff_atom)
            R,c_star = self.correlation_matrix(R_val,task,sig_optim,tril_perms_lin,task['lam'])
            #F_predict = torch.mm(k_val_train,alpha_t)
            F_predict = self.predicted_F_tensor(k_val_train, alpha_t, task)
            loss = self.loss(k_val_train, k_val_train,alpha_t,R,c_star,sigma_2_hat,F_predict,c,task)
            
            #loss = torch.norm(F_predict)
            if index%20 == 1:
                print(loss)
            #output = model(input)
            #loss = loss_fn(output, target)
            loss.backward()
            # with torch.no_grad():
            #     R_val_tensor  -= step_size * R_val_tensor.grad
            #     R_val_tensor.grad.zero_()
            optimizer.step()
            
            
        print("final F_predict",F_predict)
        return R_val_tensor,F_predict

  
        
    
    def _assemble_kernel_mat_test(
            self,
            index_diff_atom,
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
        #start = timeit.default_timer()
        pool = Pool(mp.cpu_count())
        #pool = Pool(None)
        #pool = Pool(self._max_processes)
        #todo, done = K_n_cols, 0
        
        pool.imap_unordered(
            partial(
                _assemble_kernel_mat_wkr_test,
                tril_perms_lin=tril_perms_lin,
                tril_perms_lin_mirror=tril_perms_lin_mirror,
                sig=sig,
                index_diff_atom=index_diff_atom,
                use_E_cstr=use_E_cstr,
                exploit_sym=exploit_sym,
                cols_m_limit=cols_m_limit,
            ),
            J,
        )
            
        pool.close()
        pool.join()  # Wait for the worker processes to terminate (to measure total runtime correctly).
        #stop = timeit.default_timer()
        
        glob.pop('K', None)
        glob.pop('R_desc', None)
        glob.pop('R_d_desc', None)
        glob.pop('R_desc_val', None)
        glob.pop('R_d_desc_val', None)
    
        return np.frombuffer(K).reshape(glob['K_shape'])
    
    def _assemble_kernel_mat_Energy(
            self,
            index_diff_atom,
            R_desc,
            R_d_desc,
            R_desc_val_atom,
            R_d_desc_val_atom,
            tril_perms_lin,
            tril_perms_lin_mirror,
            sig,
            lam_e,
            lam_f,
            batch_size,
            desc,  # TODO: document me
            F_star,
            E_train,
            uncertainty=False,
            use_E_cstr=False,
            col_idxs=np.s_[:],  # TODO: document me
            callback=None,
        ):
        r"""
            Compute force field kernel matrix.
        """
        n_val, dim_d = R_d_desc_val_atom.shape[:2]
        n_train, dim_d = R_d_desc.shape[:2]  #R_d_desc.shape (n_train, 66, 3)
        n_total=n_val+n_train
        n_atoms = int((1 + np.sqrt(8 * dim_d + 1)) / 2)
        dim_i = 3 * n_atoms
        #n_train , dim_d 66
        n_perms = int(len(tril_perms_lin) / dim_d)
        E_predict = np.empty(n_val)
        E_L = np.empty(n_val)
        E_H = np.empty(n_val)
        
        keep_idxs_3n = slice(None)
        
        #K_EE_all = np.empty((n_val+n_train,n_val+n_train))
        K_EE_train = np.empty((n_train,n_train))
        K_EE_val_train = np.empty((n_val,n_train))
        K_EF_all = np.empty((n_val+n_train,n_val*3*n_atoms))
        
        mat52_base_div = 3 * sig ** 4
        sqrt5 = np.sqrt(5.0)
        sig_pow2 = sig ** 2
        
        R_desc_atom = np.row_stack((R_desc_val_atom,R_desc))
        R_d_desc_atom = R_d_desc_val_atom

        start_1 = timeit.default_timer()  
        K_EF_all = self._assemble_kernel_KEF_mat(
                    index_diff_atom,
                    R_desc,
                    #R_d_desc_atom,
                    R_desc_val_atom,
                    R_d_desc_val_atom,
                    tril_perms_lin,
                    tril_perms_lin_mirror,
                    sig,
                    desc,
                    use_E_cstr=False,
                    col_idxs= np.s_[:],
                    callback=None,
                )
    
   
        
        K_FE_all =  K_EF_all.T.copy()
        stop_1 = timeit.default_timer()  
        #print(first part takes start_1 = timeit.default_timer()  )
        #print(' The EF part takes: '+repr(stop_1-start_1)+'second') 
            
        K_EE_train,K_EE_val_train = self._assemble_kernel_KEE_mat(
                    index_diff_atom,
                    R_desc,
                    #R_d_desc_atom,
                    R_desc_val_atom,
                    #R_d_desc_val_atom,
                    tril_perms_lin,
                    tril_perms_lin_mirror,
                    sig,
                    desc,
                    use_E_cstr=False,
                    col_idxs= np.s_[:],
                    callback=None,
                )
     
        stop_2 = timeit.default_timer()  
        #print(first part takes start_1 = timeit.default_timer()  )
        #print(' The part EF+EE takes: '+repr(stop_2-start_1)+'second') 
           
        n_batch=int(n_val/batch_size)
        
        start_2 = timeit.default_timer() 
        A= K_EE_train.copy()+np.diag(np.repeat(lam_e,n_train))
        L_A, lower_A = sp.linalg.cho_factor(
                A, overwrite_a=True, check_finite=False
            )
        
        A_inv_E= sp.linalg.cho_solve(   
                            (L_A, lower_A), -E_train.copy(),overwrite_b=True, check_finite=False)
        
        H1=np.repeat([1],n_train)
        #H=np.concatenate([np.repeat([1],n_train),np.repeat([0],3*n_atoms*batch_size)]).reshape(-1,1)
          
        
        A_inv_H=sp.linalg.cho_solve(   
                            (L_A, lower_A), H1,overwrite_b=True, check_finite=False)
        
        
        for l in range(n_batch):
            if uncertainty:
                K_es_es=np.empty((batch_size,batch_size))
            dim_i = 3 * n_atoms
            K_FsFs_all = np.empty((batch_size*3*n_atoms,batch_size*3*n_atoms))
            #index1 = np.arange(batch_size)+l*batch_size
            mat52_base_div = 3 * sig ** 4
            for j in range(batch_size):
                blk_j = slice(j*dim_i , (j + 1)*dim_i )
                index_j=j+l*batch_size
                rj_desc_perms = np.reshape(
                np.tile(R_desc_atom[index_j, :], n_perms)[tril_perms_lin_mirror], (n_perms, -1), order='F'
            )
                rj_d_desc = desc.d_desc_from_comp(R_d_desc_atom[index_j, :, :])[0][
                    :, keep_idxs_3n
                ]
                rj_d_desc_perms = np.reshape(
                    np.tile(rj_d_desc.T, n_perms)[:, tril_perms_lin_mirror], (-1, dim_d, n_perms)
                )
                dim_i_keep =dim_i
                diff_ab_perms = np.empty((n_perms, dim_d))
                diff_ab_outer_perms = np.empty((dim_d, dim_i_keep)) 
                ri_d_desc = np.zeros((1, dim_d, dim_i))
                for i in range(j+1):
                    index_i=i+l*batch_size
                    #desc.d_desc_from_comp(R_d_desc_atom[index1[i], :, :], out=ri_d_desc)
                    k = np.empty((dim_i, dim_i))
                    blk_i = slice(i*dim_i , (i + 1)*dim_i )
                    np.subtract(R_desc_atom[index_i, :], rj_desc_perms, out=diff_ab_perms)
                    
                        
                    norm_ab_perms = sqrt5 * np.linalg.norm(diff_ab_perms, axis=1)
                    
                    if uncertainty:
                        K_es_es[i,j]  = (1 + (norm_ab_perms / sig) * (1 + norm_ab_perms / (3 * sig))
            ).dot(np.exp(-norm_ab_perms / sig)) 
                        K_es_es[j,i]  = K_es_es[i,j].copy()
                        
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
                    
                    desc.d_desc_from_comp(R_d_desc_atom[index_i, :, :], out=ri_d_desc)
                    np.dot(ri_d_desc[0].T, diff_ab_outer_perms, out=k)
                    #K_FsFs_all[np.ix_(blk_i,blk_j)]=-k.copy()
                    K_FsFs_all[blk_i,blk_j]=-k.copy()
                    if (blk_i!=blk_j):  # this will never be called with 'keep_idxs_3n' set to anything else than [:]
                        K_FsFs_all[blk_j,blk_i]=-k.T.copy()
                    
                
            #R_E=np.empty((n_train+3*n_atoms*batch_size,n_train+3*n_atoms*batch_size))
            #R_E[np.ix_(np.arange(n_train),np.arange(n_train))]=A
            Y_vec=np.empty((n_train+3*n_atoms*batch_size))
            
            Y_vec[np.arange(n_train)]=-E_train.copy()
            index = np.arange(l*3*n_atoms*batch_size,(l+1)*3*n_atoms*batch_size)
            D=K_FsFs_all.copy() + np.diag(np.repeat(lam_e,batch_size*n_atoms*3))
            
            B=K_EF_all[
                np.ix_(np.arange(n_val,n_total),index)].copy()
            
            C=K_FE_all[
                np.ix_(index,np.arange(n_val,n_total))].copy()
            
            
            A_inv_B=sp.linalg.cho_solve(   
                            (L_A, lower_A), B,overwrite_b=True, check_finite=False)
            # A_inv_B=sp.linalg.solve(
            #                 A,  B, overwrite_a=True, overwrite_b=True, check_finite=False
            #             )
            
            
            
            D_CAB= D- C @ A_inv_B

                
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                try:
                    L_D, lower_D = sp.linalg.cho_factor(
                    D_CAB, overwrite_a=True, check_finite=False
                )
                    D_CAB_inv_C=sp.linalg.cho_solve(   
                        (L_D, lower_D), C,overwrite_b=True, check_finite=False)
                    D_CAB_inv_F=sp.linalg.cho_solve(   
                        (L_D, lower_D), F_star[index].copy(),overwrite_b=True, check_finite=False)
                    H_A_H= H1.T @ A_inv_H +H1.T @ A_inv_B @D_CAB_inv_C @ A_inv_H
                    H_R_EF_inv_E = H1.T @ (A_inv_E + A_inv_B @ D_CAB_inv_C @ A_inv_E)
                    
                    m1= (1/H_A_H) * (H_R_EF_inv_E - H1.T @ A_inv_B @ D_CAB_inv_F )
                    #m1= (1/H_A_H) * (H1.T @ A_inv_E)
                    #print(m1)
                    E_m= -E_train.copy() - m1
                    
                    
                    A_inv_E_M=sp.linalg.cho_solve(
                        (L_A, lower_A), np.array(E_m), overwrite_b=True, check_finite=False
                    )  
                    
                    D_CAB_inv_CAEM=sp.linalg.cho_solve(   
                (L_D, lower_D), (C @ A_inv_E_M),overwrite_b=True, check_finite=False)
                    #S2=(E_m.T @ A_inv_E_M)/n_val
                    
                except np.linalg.LinAlgError: 
                    D_CAB_inv_C  = sp.linalg.solve(
                            D_CAB,  C, overwrite_a=True, overwrite_b=True, check_finite=False
                        )
                    D_CAB_inv_F  = sp.linalg.solve(
                            D_CAB,  F_star[index].copy(), overwrite_a=True, overwrite_b=True, check_finite=False
                        )
                    H_A_H= H1.T @ A_inv_H +H1.T @ A_inv_B @D_CAB_inv_C @ A_inv_H
                    H_R_EF_inv_E = H1.T @ (A_inv_E + A_inv_B @ D_CAB_inv_C @ A_inv_E)
                    
                    m1= (1/H_A_H) * (H_R_EF_inv_E - H1.T @ A_inv_B @ D_CAB_inv_F )
                    #print(m1)
                    E_m= -E_train.copy() - m1
                    
                    
                    A_inv_E_M=sp.linalg.cho_solve(
                        (L_A, lower_A), np.array(E_m), overwrite_b=True, check_finite=False
                    )
                    D_CAB_inv_CAEM  = sp.linalg.solve(
                            D_CAB,  (C @ A_inv_E_M), overwrite_a=True, overwrite_b=True, check_finite=False
                        )
                    #S2=(E_m.T @ A_inv_E_M)/n_val
                    
  
            
            A_inv_BDEM=sp.linalg.cho_solve(   
                (L_A, lower_A), (B @ D_CAB_inv_CAEM),overwrite_b=True, check_finite=False)
            
            A_inv_BDF=sp.linalg.cho_solve(   
                (L_A, lower_A), (B @ D_CAB_inv_F),overwrite_b=True, check_finite=False)
            
            
            RE_inv=np.empty((n_train+3*n_atoms*batch_size))
                        
            RE_inv[np.arange(n_train)]=A_inv_E_M+ A_inv_BDEM - A_inv_BDF
            RE_inv[np.arange(n_train,n_train+3*n_atoms*batch_size)]= -D_CAB_inv_CAEM + D_CAB_inv_F
           
            #R_E=np.block([[A,B],[C,D]])
            
            #Y_vec[np.arange(n_train,n_train+3*n_atoms*batch_size)]=F_star[index].copy()
            
            index_l=np.arange(l*batch_size,(l+1)*batch_size)
            Kr_E=np.empty((batch_size,n_train+3*n_atoms*batch_size))
            Kr_E[np.ix_(np.arange(batch_size),np.arange(n_train))] = K_EE_val_train[np.ix_(index_l,np.arange(n_train))].copy()
            Kr_E[np.ix_(np.arange(batch_size),np.arange(n_train,n_train+3*n_atoms*batch_size))] =  K_EF_all[np.ix_(index_l,np.arange(l*3*n_atoms*batch_size,(l+1)*3*n_atoms*batch_size))].copy()
            

            
            m_hat=m1

            E_predict[index_l]=(m_hat+np.matmul(Kr_E,RE_inv))#[:,0]
            
            if uncertainty:
                R_E_inv_krE = np.empty((n_train+3*n_atoms*batch_size,batch_size))
                K1 = K_EE_val_train[np.ix_(index_l,np.arange(n_train))].T.copy()
                K2 = K_EF_all[np.ix_(index_l,np.arange(l*3*n_atoms*batch_size,(l+1)*3*n_atoms*batch_size))].T.copy()
                
                Y_vec=np.empty((n_train+3*n_atoms*batch_size))
                
                Y_vec[np.arange(n_train)]=E_m.copy()
                Y_vec[np.arange(n_train,n_train+3*n_atoms*batch_size)]=F_star[index].copy()
                
                R_E=np.block([[A,B],[C,D]])
                
                # L2, lower2 = sp.linalg.cho_factor(
                #     R_E, overwrite_a=True, check_finite=False
                # )
                
                # R_E_inv_krE=sp.linalg.cho_solve(   
                #         (L2, lower2), Kr_E.T,overwrite_b=True, check_finite=False)
                
                # R_E_inv_e = sp.linalg.cho_solve(   
                #         (L2, lower2), Y_vec,overwrite_b=True, check_finite=False)
                
                #A_inv_E_M1= np.linalg.solve(A,np.array(E_m))
                
                # A_inv_K1 = sp.linalg.cho_solve(   
                #          (L_A, lower_A), K1,overwrite_b=True, check_finite=False)
                
                # D_CAB_inv_K2 = sp.linalg.cho_solve(   
                #          (L_D, lower_D), K2,overwrite_b=True, check_finite=False)
                # D_CAB_inv_CAK = sp.linalg.cho_solve(   
                #          (L_D, lower_D), C @ A_inv_K1,overwrite_b=True, check_finite=False)
                # A_inv_all= sp.linalg.cho_solve(   
                #          (L_A, lower_A), B @ D_CAB_inv_CAK,overwrite_b=True, check_finite=False)
                # A_inv_all_k2= sp.linalg.cho_solve(   
                #          (L_A, lower_A), B @ D_CAB_inv_K2,overwrite_b=True, check_finite=False)
                
                
                # R_E_inv_krE[np.arange(n_train),:] = A_inv @ K1 + +A_inv @ B @ D_CAB_inv@ C @ A_inv @K1  - A_inv @ B @ D_CAB_inv @ K2
                # R_E_inv_krE[np.arange(n_train,n_train+3*n_atoms*batch_size),:] = -D_CAB_inv @ C @ A_inv @ K1 + D_CAB_inv @ K2
                
                
                R_E_inv_krE = np.linalg.solve(R_E,np.array(Kr_E.T))
                #S2 = (Y_vec  @ R_E_inv_e )/(n_train+3*n_atoms*batch_size)
                S2=(E_m.T @ A_inv_E_M)/n_train
                
                #K_es_es
                c_star_ctar= (K_es_es - Kr_E @ R_E_inv_krE)* S2
                E_L[index_l]=(E_predict[index_l]+norm.ppf(0.025,0,np.sqrt(np.diag(c_star_ctar))))
                E_H[index_l]=(E_predict[index_l]+norm.ppf(0.975,0,np.sqrt(np.diag(c_star_ctar))))
                
            
            
                
                
        stop_2 = timeit.default_timer()  
        #print(first part takes start_1 = timeit.default_timer()  )
        #print(' The last part takes: '+repr(stop_2-start_2)+'second') 
        
        return E_predict,E_L,E_H
    
    def _assemble_kernel_KEF_mat(
            self,
            index_diff_atom,
            R_desc,
            #R_d_desc,
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
        # R_desc_val_atom
        r"""
            Compute force field kernel matrix.
        """
        global glob
    
            # Note: This function does not support unsorted (ascending) index arrays.
            # if not isinstance(col_idxs, slice):
            #    assert np.array_equal(col_idxs, np.sort(col_idxs))
    
        n_train, dim_d = R_desc.shape[:2]  #R_d_desc.shape (n_train, 66, 3)
        #n_train , dim_d 66
        dim_i = 3 * int((1 + np.sqrt(8 * dim_d + 1)) / 2)  # dim = 3 * 12
        n_val,dim_d= R_d_desc_val_atom.shape[:2]
        n_atoms = int((1 + np.sqrt(8 * dim_d + 1)) / 2)
        # Determine size of kernel matrix.
        #  **** need change
        #K_n_rows = n_train*3 #* 6   
        #K_EF_all = np.empty((n_val+n_train,n_val*3*n_atoms))
        K_n_rows = n_val+n_train  
        K_n_cols = n_val*3*n_atoms    
        
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
    
            J = range(*M_slice.indices(n_val))
    
            if M_slice_start is None:
                exploit_sym = True
                cols_m_limit = M_slice_stop
        
        #K = mp.RawArray('d', n_type * K_n_rows * K_n_cols)
        K = mp.RawArray('d',  K_n_rows * K_n_cols)
        glob['K'], glob['K_shape'] = K, (K_n_rows, K_n_cols)
        glob['R_desc_atom'], glob['R_desc_shape_atom'] = _share_array(R_desc, 'd')
        #glob['R_d_desc_atom'], glob['R_d_desc_shape_atom'] = _share_array(R_d_desc, 'd')
        glob['R_desc_val_atom'], glob['R_desc_shape_atom_val'] = _share_array(R_desc_val_atom, 'd')
        glob['R_d_desc_val_atom'], glob['R_d_desc_shape_atom_val'] = _share_array(R_d_desc_val_atom, 'd')
    
        glob['desc_func'] = desc
        #start = timeit.default_timer()
        pool = Pool(mp.cpu_count())
       
        
        pool.imap_unordered(
            partial(
                _assemble_kernel_mat_EF_wkr,
                tril_perms_lin=tril_perms_lin,
                tril_perms_lin_mirror=tril_perms_lin_mirror,
                sig=sig,
                index_diff_atom=index_diff_atom,
                use_E_cstr=use_E_cstr,
                exploit_sym=exploit_sym,
                cols_m_limit=cols_m_limit,
            ),
            J,
        )
       
            
        pool.close()
        pool.join()  # Wait for the worker processes to terminate (to measure total runtime correctly).
        #stop = timeit.default_timer()
        #dur_s = (stop - start)
        
        glob.pop('K', None)
        glob.pop('R_desc_atom', None)
        #glob.pop('R_d_desc_atom', None)
        glob.pop('R_desc_val_atom', None)
        glob.pop('R_d_desc_val_atom', None)
    
        return np.frombuffer(K).reshape(glob['K_shape'])
    
    def torch_p_dist(self,r):
        """
        Compute pairwise Euclidean distance matrix between all atoms.
        Parameters
        ----------
            r : :obj:`tensor.`
                Array of size 3N containing the Cartesian coordinates of
                each atom.
        Returns
        -------
            :obj:`torch.tensor`
                Array of size N x N containing all pairwise distances between atoms.
        """ 
        torch.reshape(r, (-1,3))
        #r = r.reshape(-1, 3)
        #r_tensor = torch.from_numpy(r).type(torch.float64)
        pdist = torch.cdist(r, r, p=2)
        n_atoms = r.shape[0]
        tril_idxs = torch.tril_indices(n_atoms, n_atoms,-1)
        
        return pdist[tril_idxs[0],tril_idxs[1]]
        
        # pdist = nn.PairwiseDistance(p=2)
        # pdist(r)

        # if lat_and_inv is None:
        #     pdist = sp.spatial.distance.pdist(r, 'euclidean')
        # else:
        #     pdist = sp.spatial.distance.pdist(
        #         r, lambda u, v: np.linalg.norm(_pbc_diff(u - v, lat_and_inv))
        #     )

        # tril_idxs = np.tril_indices(n_atoms, k=-1)
        # return sp.spatial.distance.squareform(pdist, checks=False)[tril_idxs]
        

    def torch_r_to_desc(self,r,pdist):
        """
        Generate descriptor Jacobian for a set of atom positions in
        Cartesian coordinates.
        This method can apply the minimum-image convention as periodic
        boundary condition for distances between atoms, given the edge
        length of the (square) unit cell.
        Parameters
        ----------
            r : :obj:`numpy.ndarray`
                Array of size 3N containing the Cartesian coordinates of
                each atom.
            pdist : :obj:`torch.tensor`
                Array of size N x N containing the Euclidean distance
                (2-norm) for each pair of atoms. 1D tensor of N*(N-1)/2
            
        Returns
        -------
            :obj:`tensor.tensor`
                1D tensor of N*(N-1)/2
        """
        return 1/pdist

    def torch_r_to_d_desc(self,r_tensor, pdist):  # TODO: fix documentation!
        """
        Generate descriptor Jacobian for a set of atom positions in
        Cartesian coordinates.
        This method can apply the minimum-image convention as periodic
        boundary condition for distances between atoms, given the edge
        length of the (square) unit cell.
        Parameters
        ----------
            r : :obj:`torch .ndarray`
                Array of size 3N containing the Cartesian coordinates of
                each atom.
            pdist : :obj:`numpy.ndarray`
                Array of size (N * N-1) /2 containing the Euclidean distance
                (2-norm) for each pair of atoms.
            
        Returns
        -------
            :obj:`numpy.ndarray`
                Array of size N(N-1)/2 x 3N containing all partial
                derivatives of the descriptor.
        """
        torch.reshape(r_tensor, (-1,3))
        #r = r.reshape(-1, 3)
        pdiff = r_tensor[:, None] - r_tensor[None, :]  # pairwise differences ri - rj

        n_atoms = r_tensor.shape[0]
        i,j = torch.tril_indices(n_atoms, n_atoms,-1)
        #i, j = np.tril_indices(n_atoms, k=-1)

        pdiff = pdiff[i, j, :]  # lower triangular

        d_desc_elem = pdiff / (torch.pow(pdist, 3))[:, None] #original
        
        

        
            #d_desc_elem = -pdiff / (pdist ** 3)[:, None]
        
        
        return d_desc_elem

    def tensor_from_r(self,r, lat_and_inv=None, coff=None):
        """
        Generate descriptor and its Jacobian for one molecular geometry
        in Cartesian coordinates.
        Parameters
        ----------
            r : :obj:`numpy.ndarray`
                Array of size 3N containing the Cartesian coordinates of
                each atom.
            
        Returns
        -------
            :obj:`tensor .ndarray`
                Descriptor representation as 1D array of size N(N-1)/2
            :obj:`tensor .ndarray`
                Array of size N(N-1)/2 x 3N containing all partial
                derivatives of the descriptor.
        """

        # Add singleton dimension if input is (,3N).
        if r.ndim == 1:
            r = r[None, :]

        pd = self.torch_p_dist(r)

        r_desc = self.torch_r_to_desc(r, pd)
        r_d_desc = self.torch_r_to_d_desc(r, pd)

        return r_desc, r_d_desc
        


    def tensor_d_desc_from_comp(self,R_d_desc,n_atoms):
        '''

        Parameters
        ----------
        R_d_desc : TYPE
            DESCRIPTION.
        n_atoms : TYPE
            DESCRIPTION.

        Returns
        -------
        out : real d D /d_x: M times 66 times 3N
            DESCRIPTION.

        '''
        # NOTE: out must be filled with zeros!
        if R_d_desc.ndim == 2:
            R_d_desc = R_d_desc[None, ...]
     
        
        n = R_d_desc.shape[0]
        dim = (n_atoms * (n_atoms - 1)) // 2
        out = torch.zeros(n, dim, n_atoms, 3,dtype = torch.float64)
        i, j = torch.tril_indices(n_atoms, n_atoms,-1)
        
        dim_range = torch.arange(dim)
        out[:, dim_range, j, :] = R_d_desc
        out[:, dim_range, i, :] = -R_d_desc
        
        out = torch.reshape(out, (-1,dim,3*n_atoms))
        return out

    def reshape_fortran(self, x, shape):
        if len(x.shape) > 0:
            x = x.permute(*reversed(range(len(x.shape))))
        return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))

    def torch_perm(perm):
        
        n = len(perm)
        

    def tensor_correlation_ij(self, r_desc_i,r_d_desc_i,r_desc_j,r_d_desc_j, sig,tril_perms_lin,index_diff_atom):
        """
        Generate the correlation matrix tensor of x_i, and x_j
        Parameters
        ----------
            r : 
            
        Returns
        -------
            correlation matrix 3N * 3N
        """
        dim_d = r_d_desc_i.shape[0]  # N*(N-1) /2 
        n_atoms = int((1 + np.sqrt(8 * dim_d + 1)) / 2)
        dim_i = 3 * n_atoms  # 36
        
        n_perms = int(len(tril_perms_lin) / dim_d)
        
        rj_d_desc = self.tensor_d_desc_from_comp(r_d_desc_j,n_atoms)[0]
        ri_d_desc = self.tensor_d_desc_from_comp(r_d_desc_i,n_atoms)[0]
        
        rj_desc_perms = self.reshape_fortran(
            torch.tile(r_desc_j, (n_perms,))[tril_perms_lin], (n_perms, -1)
        )
        
        rj_d_desc_perms = torch.reshape(
            torch.tile(rj_d_desc.T, (n_perms,))[:, tril_perms_lin], (-1, dim_d, n_perms)
        )
        
        
        mat52_base_div = 3 * sig ** 4
        sqrt5 = np.sqrt(5.0)
        sig_pow2 = sig ** 2
        
        diff_ab_perms = r_desc_i - rj_desc_perms
        
        norm_ab_perms = sqrt5 * torch.linalg.norm(diff_ab_perms, axis = 1)
        
        mat52_base_perms = torch.exp(-norm_ab_perms / sig) / mat52_base_div * 5
        
        diff_ab_outer_perms = torch.einsum('ki,kj->ij',
                                           diff_ab_perms * mat52_base_perms[:, None] * 5,
                                           torch.einsum('ki,jik -> kj', diff_ab_perms, rj_d_desc_perms)
                                           )
        diff_ab_outer_perms -= torch.einsum(
            'ikj,j->ki',
            rj_d_desc_perms,
            (sig_pow2 + sig * norm_ab_perms) * mat52_base_perms,
        )
        
        
        #desc_func.d_desc_from_comp(R_d_desc_val[i, :, :], out=ri_d_desc)

        #K[blk_i, blk_j] = ri_d_desc[0].T.dot(diff_ab_outer_perms)
        #k = torch.mm(torch.t(ri_d_desc), diff_ab_outer_perms)
        
        n_type=len(index_diff_atom)
        
        if(n_atoms<=12):
            k = torch.mm(torch.t(ri_d_desc), diff_ab_outer_perms)
        else:
            k = torch.empty((n_atoms*3,3*n_atoms), dtype=torch.float64)
            #k1 = np.empty((3,3))
            for l in range(0, n_type):
                lenl=len(index_diff_atom[l])
                
                #out = torch.empty((n_atom*3,3*n_train*n_atom), dtype=torch.float64)
                index = torch.tile(torch.arange(3),(lenl,))+3*torch.repeat_interleave(torch.tensor(index_diff_atom[l]),3)
                
               
                
                #index = np.arange(3)+3*l

                k[np.ix_(index,index)] = torch.mm(torch.t(ri_d_desc)[index,:], diff_ab_outer_perms[:,index])
                #k[np.ix_(index,index)]=k1.copy()
        
        return -k
    
    def torch_correlation_val_train(self,R_val_tensor, R_train_tensor, sig,tril_perms_lin,index_diff_atom):
        """
        Return the correlation tensor matrix of R_val with R_train 

        Parameters
        ----------
        R_val : TYPE
            DESCRIPTION.
        R_train : M times N times 3
            DESCRIPTION.
        sig : TYPE
            DESCRIPTION.
        tril_perms_lin : TYPE
            DESCRIPTION.

        Returns
        Tensor: 3N times 3MN
        None.

        """
        n_train = R_train_tensor.shape[0]
        n_atom = R_train_tensor.shape[1]
        dim = 3*n_atom
        out = torch.empty((n_atom*3,3*n_train*n_atom), dtype=torch.float64)
        
        #r_i = R_val
        #r_tensor_i = torch.from_numpy(r_i).type(torch.float32)
        r_tensor_i = R_val_tensor
        r_desc_i,r_d_desc_i = self.tensor_from_r(r_tensor_i)
        
        for j in range(n_train):
        
            r_tensor_j = R_train_tensor[j,:,:]
            #r_tensor_j = torch.from_numpy(r_j).type(torch.float32)
            r_desc_j,r_d_desc_j = self.tensor_from_r(r_tensor_j)
            
            out[0:(n_atom*3),(j*dim):((j+1)*dim)] = self.tensor_correlation_ij(r_desc_i,r_d_desc_i,r_desc_j,r_d_desc_j, sig,tril_perms_lin,index_diff_atom)
        
        return out
    
    def _assemble_kernel_KEE_mat(
            self,
            index_diff_atom,
            R_desc,
            #R_d_desc,
            R_desc_val_atom,
            #R_d_desc_val_atom,
            tril_perms_lin,
            tril_perms_lin_mirror,
            sig,
            desc,  # TODO: document me
            use_E_cstr=False,
            col_idxs=np.s_[:],  # TODO: document me
            callback=None,
        ):
        # R_desc_val_atom
        r"""
            Compute force field kernel matrix.
        """
        global glob
    
            # Note: This function does not support unsorted (ascending) index arrays.
            # if not isinstance(col_idxs, slice):
            #    assert np.array_equal(col_idxs, np.sort(col_idxs))
    
        n_train, dim_d = R_desc.shape[:2]  #R_d_desc.shape (n_train, 66, 3)
        #n_train , dim_d 66
        dim_i = 3 * int((1 + np.sqrt(8 * dim_d + 1)) / 2)  # dim = 3 * 12
        n_val,dim_d= R_desc_val_atom.shape[:2]
        n_atoms = int((1 + np.sqrt(8 * dim_d + 1)) / 2)
        # Determine size of kernel matrix.
        #  **** need change
        #K_n_rows = n_train*3 #* 6   
        #K_EF_all = np.empty((n_val+n_train,n_val*3*n_atoms))
        K_n_rows_val = n_val 
        K_n_cols = n_train    
        
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
        K_val = mp.RawArray('d',  K_n_rows_val * K_n_cols)
        K = mp.RawArray('d',  K_n_cols * K_n_cols)
        glob['K'], glob['K_shape'] = K, (K_n_cols, K_n_cols)
        glob['K_val'], glob['K_val_shape'] = K_val, (K_n_rows_val, K_n_cols)
        glob['R_desc_atom'], glob['R_desc_shape_atom'] = _share_array(R_desc, 'd')
        #glob['R_d_desc_atom'], glob['R_d_desc_shape_atom'] = _share_array(R_d_desc, 'd')
        glob['R_desc_val_atom'], glob['R_desc_shape_atom_val'] = _share_array(R_desc_val_atom, 'd')
        #glob['R_d_desc_val_atom'], glob['R_d_desc_shape_atom_val'] = _share_array(R_d_desc_val_atom, 'd')
    
        glob['desc_func'] = desc
        #start = timeit.default_timer()
        pool = Pool(mp.cpu_count())
       
        
        pool.imap_unordered(
            partial(
                _assemble_kernel_mat_EE_wkr,
                tril_perms_lin=tril_perms_lin,
                tril_perms_lin_mirror=tril_perms_lin_mirror,
                sig=sig,
                index_diff_atom=index_diff_atom,
                use_E_cstr=use_E_cstr,
                exploit_sym=exploit_sym,
                cols_m_limit=cols_m_limit,
            ),
            J,
        )
       
            
        pool.close()
        pool.join()  # Wait for the worker processes to terminate (to measure total runtime correctly).
        #stop = timeit.default_timer()
        #dur_s = (stop - start)
        
        glob.pop('K', None)
        glob.pop('K_val', None)
        glob.pop('R_desc_atom', None)
        #glob.pop('R_d_desc_atom', None)
        glob.pop('R_desc_val_atom', None)
        #glob.pop('R_d_desc_val_atom', None)
    
        return np.frombuffer(K).reshape(glob['K_shape']),np.frombuffer(K_val).reshape(glob['K_val_shape'])

    
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
    
    
   

#dataset=np.load('benzene_old_dft.npz')
#dataset=np.load('../uracil_dft.npz')
# dataset = np.load('/Users/HL/Desktop/Study/SFM/Data_Generation/h2co/h2c0_hao.npz')
# #dataset=np.load('uracil_dft_mu.npz')
# #dataset=np.load('malonaldehyde_ccsd_t-train.npz')
# #dataset=np.load('glucose_alpha.npz')
# #dataset=np.load('alkane.npz')
# #dataset=np.load('naphthalene_dft.npz')
# #dataset=np.load('aspirin_new_musen.npz')
# #
# #dataset=np.load('H2CO_mu.npz')
# gdml_train=GDMLTrain()
# #n_train=np.array([100])
# #n_train=np.array([200,600,1000,1400])
# n_train=np.array([100])
# for i in range(n_train.shape[0]):
#i=0
    # print(' The N_train is '+repr(n_train[i])+'--------------------')
    # # task=gdml_train.create_task(dataset,n_train[i],dataset,200,100,100,1e-12,use_E_cstr=False,batch_size=100,uncertainty=False)
    # # # #task=np.load('task_test.npy',allow_pickle=True).item()
    
    # # # #task=np.load('task_benzene{}.npy'.format(i),allow_pickle=True).item()
    # # task['uncertainty']=False
    # # task['lam']=1e-13
    
    # # np.save('./saved_model/task_h2co_aff.npy', task)
    # task=np.load('./saved_model/task_h2co_aff.npy',allow_pickle=True).item()
    
    # #np.save('task_uracil_75_400{}.npy'.format(n_train[i]), task) 
    # #sig_opt,sig_opt_E,alphas_opt = gdml_train.train(task,np.arange(10,100,10),np.arange(16,35,2))#uracil
    # sig_opt,sig_opt_E,alphas_opt,kernel_time_ave,sigma_2_hat = gdml_train.train(task,np.arange(1,30,10),np.arange(0.1,1,0.1))#uracil
    # #test_MAE=gdml_train.test(task,sig_opt,sig_opt_E,alphas_opt,kernel_time_ave)
    # R_proposed_tensor = gdml_train.inverse(task,sig_opt,alphas_opt,sigma_2_hat,initial=1, c = 1e-5)

#np.save('task_test.npy', task) 
#save_pet(task)   

# import plotly.graph_objects as go
# import numpy as np
# #import plotly.graph_objects as go
# import plotly.io as pio
# #pio.renderers.default = 'svg'  # change it back to spyder
# pio.renderers.default = 'browser'

# # Helix equation
# #t = np.linspace(0, 10, 50)
# R_target = R_proposed_tensor.cpu().detach().numpy()
# #R_target = task['R_test'][0,:,:]
# x, y, z = R_target[:,0],R_target[:,1],R_target[:,2]

# fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z,
#                                   mode='markers')])
# fig.show()
#fig.savefig('Design_structure.png',dpi=300,bbox_inches='tight')




# n_train=100
# # use_sym=False means use GDML, use_sym=True (default ) means use sGDML
# task=gdml_train.create_task(dataset,n_train,dataset,50,50,100,1e-15,use_E_cstr=True,batch_size=10)
# # #task=gdml_train.create_task(dataset,100,dataset,50,100,100,1e-15)

# sig_opt,sig_opt_E,alphas_opt = gdml_train.train(task)

# test_MAE=gdml_train.test(task,sig_opt,sig_opt_E,alphas_opt)
# dataset2=np.load('qchem_uracil.npz')
# dataset=np.load('uracil_dft.npz')
# # np.savez("uracil_dft_mu.npz", E=dataset2['E'], F=dataset2['F'],R=dataset2['R'],z=dataset2['z'],name=dataset['name'],theory=dataset['theory'],md5=dataset['md5'],type=dataset['type'])
# #a=np.load('data_h2co.npz')
# #dataset=np.load('h2co_mp2_avtz_4001.npz')
# #dataset=np.load('h2co_ccsdt_avtz_4001.npz')
# #np.savez("h2co_mp2.npz", E=dataset['E'], F=dataset['F'],R=dataset['R'],z=dataset['Z'][0],name=a['name'],theory=a['theory'],md5=a['md5'],type=a['type'])

# #a=np.load('data_h2co.npz')
# dataset1=np.load('H2CO.npz')
# #dataset=np.load('h2co_ccsdt_avtz_4001.npz')
# np.savez("H2CO_mu.npz", E=dataset1['E'], F=dataset1['F'],R=dataset1['R'],z=dataset1['z'],name=dataset['name'],theory=dataset['theory'],md5=dataset['md5'],type=dataset['type'])








