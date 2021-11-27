#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 15:52:42 2021

@author: lihao
"""
##!/usr/bin/env python3
# -*- coding: utf-8 -*-

# this version implemented the predicted energy based on the joint dist of E*, E and F*.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import timeit
import logging
import warnings
#os.chdir('/home/a510396/testl')
os.chdir('./')
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
from utils.desc_inv import Desc_inv
from functools import partial
from scipy.stats import norm


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
        
    
    def train(self, task,sig_candid_F,sig_candid_E=None,
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
            
            F_star=np.empty([n_val*3*n_atoms])
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

        return sig_optim[0],sig_optim_E,alphas_opt,kernel_time_ave
    
    def test(self, task,sig_optim,
             alphas_opt,
             kernel_time_ave=0,
             sig_candid1_opt=None,
            cprsn_callback=None,
            save_progr_callback=None,  # TODO: document me
            callback=None):
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
        print(' This is the  RMSE of F='+repr(RMSE_F)) 
        
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
            print(' This is the testing task: RMSE of E='+repr(RMSE_E)) 
            if uncertainty: 
                P_C=np.sum( np.array((E_L)<=(-ye_val)) & np.array((E_H)>=(-ye_val)))/(n_val)
                A_L=np.mean( (E_H)-(E_L))
                print(' Energy: The percentage of coverage  by 95% CI is '+repr(P_C*100)+' %.') 
                print(' Energy: The average length of the 95% CI is '+repr(A_L)) 
       
        
        
       
        return MAE
    
    def inverseF(self, task,sig_optim,alphas_opt,ind_initial,F_target,lr,
            cprsn_callback=None,
            save_progr_callback=None,  # TODO: document me
            callback=None):
        task = dict(task)
        solver = task['solver_name']
        batch_size=task['batch_size']
        n_train, n_atoms = task['R_train'].shape[:2]
        n_val=1
        desc = Desc(
                n_atoms,
                interact_cut_off=task['interact_cut_off'],
                max_processes=None,
            )
        desc_inv = Desc_inv(
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
        R_atom = task['R_train']  #.reshape(n_train, -1) 
        #R_val_atom=task['R_test'][ind_initial,None] #.reshape(n_val,-1)
        R_val_atom=task['R_train'][ind_initial,None] #.reshape(n_val,-1)
        tril_perms_lin_mirror = tril_perms_lin


        R_desc_atom, R_d_desc_atom = desc.from_R(R_atom,lat_and_inv=lat_and_inv,
                callback=None)
        F_train_atom=[]
        for i in range(n_type):
            index=np.array(index_diff_atom[i])

            F_train_atom.append(task['F_train'][:,index,:].reshape(int(n_train*(len(index_diff_atom[i])*3)),order='C'))
        cost=5
        cost1=8000
        record=[]
        kk=1
        cost_SAE=1000
        
        R_design= []
        
        while kk<30 and cost_SAE>0.3:
            kk+=1

            R_desc_val_atom, R_d_desc_val_atom = desc.from_R(R_val_atom,lat_and_inv=lat_and_inv,
                     callback=None)
            
            R_d2_desc_1val_all = np.empty([dim_i,n_val, dim_d, 3])
            R_d2_desc_kval_all = np.empty([dim_i,n_val, dim_d, 3])
            
            for i in range(dim_i): ### can be improved later
                R_d2_desc_1val_all[i,:,:,:], R_d2_desc_kval_all[i,:,:,:] = desc_inv.from_R(R_val_atom,index_i=i,lat_and_inv=lat_and_inv,
                    callback=None)
                
            # R_d2_desc_1val, R_d2_desc_kval = desc_inv.from_R(R_val_atom,index_i=0,lat_and_inv=lat_and_inv,
            #         callback=None)
            
            #r_val_d2_desc = np.empty()
            #r_val_d2_desc=desc_inv.d2_desc_from_comp(R_d2_desc_1val, R_d2_desc_kval ,0)
            
            R_desc_val_atom1=R_desc_val_atom[None]
            R_d_desc_val_atom1=R_d_desc_val_atom[None]
            
            
           
            F_val_atom=[]
                #F_val_atom=task['F_val'].ravel().copy()
            
            E_train = task['E_train'].ravel().copy()
            E_val = task['E_test'].ravel().copy()
            uncertainty=task['uncertainty']
    
                
                
            #print('This is tesing task : sigma='+repr(sig_optim))
            alphas=alphas_opt
            
    
            F_hat_val_F=[]
            F_hat_val_target=[]
            #F_hat_val_E=[]
    
            K_r_all = self._assemble_kernel_mat_test(
                    index_diff_atom,
                    R_desc_atom,
                    R_d_desc_atom,
                    R_desc_val_atom1,
                    R_d_desc_val_atom1,
                    tril_perms_lin,
                    tril_perms_lin_mirror,
                    sig_optim,
                    desc,
                    use_E_cstr=False,
                    col_idxs= np.s_[:],
                    callback=None,
                )
            
            delta=self._delta(
                R_desc_atom,
                R_d_desc_atom,
                R_desc_val_atom1,
                R_d_desc_val_atom1,
                R_d2_desc_1val_all,
                R_d2_desc_kval_all,
                #R_d2_desc_1val[None], 
                #R_d2_desc_kval[None],
                tril_perms_lin,
                tril_perms_lin_mirror,
                sig_optim,
                desc,
                desc_inv,
                index_diff_atom)
            
            F_star=np.empty([n_val*3*n_atoms])
            F_hat=np.empty([n_atoms,3])
            
            drl=np.zeros([3*n_atoms])
            for ind_i in range(n_type):
                index_eg=np.tile(np.arange(3),len(index_diff_atom[ind_i]))+3*np.repeat(index_diff_atom[ind_i],3)
    
                index_x=np.repeat(np.arange(n_val)*(dim_i),3*len(index_diff_atom[ind_i]))+np.tile(index_eg,n_val)
                index_y=np.repeat(np.arange(n_train)*(dim_i),3*len(index_diff_atom[ind_i]))+np.tile(index_eg,n_train)
    
     
                K_r=K_r_all[np.ix_(index_x,index_y)]
                F_hat_val_i=np.matmul(K_r,np.array(alphas[ind_i]))
                
                F_hat[index_diff_atom[ind_i],:]=F_hat_val_i.reshape(len(index_diff_atom[ind_i]),-1).copy()
                index_i=np.repeat(np.arange(n_val)*(dim_i),3*len(index_diff_atom[ind_i]))+np.tile(index_eg,n_val)
                
                for l in range(dim_i):
                    delta_l=delta[l,:,:].copy()
                    
                    dF_drl = np.matmul(delta_l[np.ix_(index_x,index_y)],np.array(alphas[ind_i]))
                    drl[l] += np.matmul(2*(F_hat_val_i-F_target[index_eg]),dF_drl)
                
    
                F_hat_val_F.append(F_hat_val_i)
                F_hat_val_target.append(F_target[index_eg])
                #F_hat_val_F.append(F_hat_val_i)
                #F_star[index_i]=F_hat_val_i
                
                      #a=(np.concatenate(F_star_L)<=np.concatenate(F_val_atom) and np.concatenate(F_star_H)>=np.concatenate(F_val_atom))
            cost1=np.sum((  np.concatenate(F_hat_val_F)-np.concatenate(F_hat_val_target))**2)
            cost_SAE=np.sum(np.abs(  np.concatenate(F_hat_val_F)-np.concatenate(F_hat_val_target)))
            cost-=1
            #F_diff= (np.concatenate(F_hat_val_F)-np.concatenate(F_val_atom)) ### 3N vector of Fij- Fij_tilde
        
            # delta should be a 3N * 3N matrix, i_th row respresent the gradient w.r.t r_i
            #delta  #
    
            
            
            #RMSE_F=np.sqrt(np.mean((np.concatenate(F_hat_val_F)-np.concatenate(F_val_atom))**2))/np.std(np.concatenate(F_val_atom))
            print(' the cost is '+repr(cost_SAE)) 
            record.append(cost_SAE)
            #print(drl)
            #print(' the mean of drl is '+repr(np.max(np.abs(drl))))
            
            dy_lr=1/np.max(np.abs(drl))* 1e-2
            #print(np.concatenat(F_hat_val_F))
            #R_val_atom+=lr
            # if cost1<8000:
            #     lr=1e-17
            # if cost1<24356:
            #     lr=1e-18
            R_val_atom_last=R_val_atom[0,:,:].copy()
            R_val_atom[0,:,:]= R_val_atom[0,:,:] - drl.reshape(n_atoms,3)*lr
            R_design.append(R_val_atom[0,:,:])
            #print(drl.reshape(n_atoms,3)*lr)
            #print(R_val_atom[0,:,:])
        
        #print(' This is the  RMSE of F='+repr(RMSE_F)) 
        

        #MAE=ae

        return np.array(R_design),R_val_atom_last,F_hat,record,cost_SAE

    def _delta(
            self, 
            R_desc,
            R_d_desc,
            R_desc_val,
            R_d_desc_val,
            R_d2_desc_1val_all, 
            R_d2_desc_kval_all,
            tril_perms_lin,
            tril_perms_lin_mirror,
            sig,
            desc,
            desc_inv,
            index_diff_atom):
        #global glob

        # R_desc = np.frombuffer(glob['R_desc']).reshape(glob['R_desc_shape'])
        # R_d_desc = np.frombuffer(glob['R_d_desc']).reshape(glob['R_d_desc_shape'])
    
        # R_desc_val = np.frombuffer(glob['R_desc_val']).reshape(glob['R_desc_shape_val'])
        # R_d_desc_val = np.frombuffer(glob['R_d_desc_val']).reshape(glob['R_d_desc_shape_val'])  

        n_val=1
        n_train, dim_d = R_d_desc.shape[:2]  #R_d_desc.shape (n_train, 66, 3)
        #n_train , dim_d 66
        dim_i = 3 * int((1 + np.sqrt(8 * dim_d + 1)) / 2)  # dim = 3 * 12
    
        
        #K = np.frombuffer(glob['K']).reshape(glob['K_shape'])
        K = np.empty((dim_i,dim_i,n_train*dim_i)) # the first dim_i is for index_i
        desc_func = desc
        desc_func_inv = desc_inv
        n_type=len(index_diff_atom)
        
        dim_d = R_d_desc.shape[1]
        #n_val, dim_d = R_d_desc_val.shape[:2]
        # dim_d =66
    
        n_atoms = int((1 + np.sqrt(8 * dim_d + 1)) / 2)
 
        n_perms = int(len(tril_perms_lin) / dim_d)
        n_perm_atom=n_perms
        
        #mat52_base_div = 3 * sig ** 4
        sqrt5 = np.sqrt(5.0)
        sig_pow2 = sig ** 2
        sig_pow3 = sig ** 3
        sig_pow4 = sig ** 4
        sig_pow5 = sig ** 5
        
        i=0
        ri_d_desc = np.zeros((1, dim_d, dim_i))
        ri_d2_desc = np.zeros((dim_i,1, dim_d, dim_i))
        desc_func.d_desc_from_comp(R_d_desc_val[i, :, :], out=ri_d_desc)
        for index_i in range(dim_i): 
            desc_func_inv.d2_desc_from_comp( R_d2_desc_1val_all[index_i,i,:,:], R_d2_desc_kval_all[index_i,i,:,:], index_i,out=ri_d2_desc[index_i,i,:,:])
          
        
        for j in range(n_train):
            
            blk_j = slice(j*dim_i , (j + 1)*dim_i )
        
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

            dim_i_keep = rj_d_desc.shape[1]  # 36
            diff_ab_outer_perms = np.empty((dim_d, dim_i_keep))   # 66 * 36
    
            diff_ab_perms = np.empty((n_perm_atom, dim_d))# 12 * 66
            
            

            #ri_d_desc = np.zeros((1, dim_d, dim_i))
            #ri_d2_desc = np.zeros((1, dim_d, dim_i))
            k = np.zeros((dim_i, dim_i_keep))
            k1 = np.zeros((3, 3))

            blk_i = slice(i * dim_i, (i + 1) * dim_i)
            
                
            #R_desc_val
            np.subtract(R_desc_val[i, :], rj_desc_perms, out=diff_ab_perms)  ### D_pq-D'_pq
            d = np.linalg.norm(diff_ab_perms, axis=1)  
            norm_ab_perms = sqrt5 * np.linalg.norm(diff_ab_perms, axis=1)
            exp_term = np.exp(-norm_ab_perms / sig)
            
            # the coefficient before (Dpq-D')(Dmn'-D)^2, when pq !=mn
            coeff1= -25.0*sqrt5/(3* sig_pow5 *d)
           
            # the coefficient before (Dpq-D')  when pq !=mn
            coeff2= 25.0/(3* sig_pow4) 
            
            # the added coefficient before (Dpq-D')(Dmn'-D)^2, when pq=mn
            coeff3= 0.0
            # the added coefficient before (Dpq-D')  when pq =mn
            coeff4= -5.0*sqrt5 /(d* sig_pow3) +5.0*sqrt5/( sig_pow3)+(25+25*d)/(3*sig_pow4)
            
            
            
            np.einsum(
                'ki,kj->ij',
                (diff_ab_perms ** 2) * exp_term[:, None] * coeff1[:,None]+coeff2 * exp_term[:, None],
                np.einsum('ki,jik -> kj', diff_ab_perms, rj_d_desc_perms),
                out=diff_ab_outer_perms
            )  
            
            diff_DD_perms = np.empty((n_perm_atom,dim_d, dim_d))   #12 * 66 * 66
            
            
            np.einsum(
                'ki,kj->kij', # 12* 66  oper 12 *66
                coeff4[:,None] * exp_term[:, None],
                diff_ab_perms,
                out=diff_DD_perms
            ) 
            
            DD_diag_perms=np.diagonal(diff_DD_perms,axis1=1,axis2=2)  # 12 * 66
            
            DD_times_rj= rj_d_desc_perms * DD_diag_perms.T[None,:] # 36 * 66 * 12
            
            diff_ab_outer_perms += np.sum(DD_times_rj,axis=2).T
            
            # np.einsum(
            #     'ki,kj->ij',
            #     (diff_ab_perms ** 2) * exp_term[:, None] * coeff3[:,None]+coeff4[:,None] * exp_term[:, None],
            #     np.einsum('ki,jik -> kj', diff_ab_perms, rj_d_desc_perms),
            #     out=diff_ab_outer_perms
            # )  
            
           
            ###  above is the dk^3/dD dD'^2   when pq !=mn

            #desc_func.d_desc_from_comp(R_d_desc_val[i, :, :], out=ri_d_desc)
            #desc_func_inv.d2_desc_from_comp( R_d2_desc_1val[i,:,:], R_d2_desc_kval[i,:,:], out=ri_d2_desc)
            #k1 = np.empty((3,3))
            for index_i in range(dim_i):
                for l in range(0, n_type):
                    k1 = np.empty((3*len(index_diff_atom[l]),3*len(index_diff_atom[l])))
                    index = np.tile(np.arange(3),len(index_diff_atom[l]))+3*np.repeat(index_diff_atom[l],3)
                    
                    #index = np.arange(3)+3*l
                    
                    np.dot(ri_d2_desc[index_i,0].T[index,:], diff_ab_outer_perms[:,index], out=k1)
                    k[np.ix_(index,index)]=k1.copy()
                
                K[index_i,blk_i, blk_j] = -k.copy()
            
            return K
            
        
        
    
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

    
    
    
   

#dataset=np.load('benzene_old_dft.npz')
#dataset=np.load('data_h2co.npz')
#dataset=np.load('h2co_ccsdt.npz')
#dataset=np.load('H2CO_mu.npz')
#dataset=np.load('uracil_dft.npz')
#dataset=np.load('uracil_dft_mu.npz')
#dataset=np.load('malonaldehyde_ccsd_t-train.npz')
#dataset=np.load('glucose_alpha.npz')
#dataset=np.load('alkane.npz')
#dataset=np.load('naphthalene_dft.npz')
#dataset=np.load('aspirin_new_musen.npz')
#
gdml_train=GDMLTrain()
#n_train=np.array([100])
#n_train=np.array([200,400,600,800,1000,1200,1400,1600])
#n_train=np.array([100])


# task=gdml_train.create_task(dataset,n_train,dataset,200,500,100,1e-12,use_E_cstr=False,batch_size=10,uncertainty=False)

# sig_opt,sig_opt_E,alphas_opt,kernel_time_ave = gdml_train.train(task,np.arange(1,20,4))#uracil

# test=gdml_train.test(task,sig_opt,alphas_opt,kernel_time_ave)
# #test(self, task,sig_optim,sig_candid1_opt,alphas_opt,kernel_time_ave,

# np.save('task.npy', task) 

# np.save('sig_opt.npy', sig_opt) 

# np.save('alphas_opt.npy', alphas_opt) 


task=np.load('task.npy',allow_pickle=True).item()
sig_opt=np.load('sig_opt.npy')

alphas_opt=np.load('alphas_opt.npy',allow_pickle=True)
#test_MAE=gdml_train.test(task,sig_opt,sig_opt_E,alphas_opt,kernel_time_ave)
F_target=np.zeros((4,3)).reshape(-1)

#np.min(np.sum(np.abs(task['F_train']-F_target),1))
#F_target = task["F_train"][4,:,:].reshape(-1)

#F_target[0]=0

initial=52
n_sam=11
R_target1=np.empty((n_sam,4,3))
F_predict=np.empty((n_sam,4,3))
cost=np.empty((n_sam,1))


#np.array(R_design),R_val_atom_last,F_hat,record,cost_SAE
R_design,R_val_atom_last,F_hat,record,cost_SAE= gdml_train.inverseF( task,sig_opt,alphas_opt,initial,F_target,lr=1e-11,
            cprsn_callback=None,
            save_progr_callback=None,  # TODO: document me
            callback=None)

np.save('R_design.npy',R_design) 



# R_target1[(n_sam-1),:,:]= test_inv.copy()
# F_predict[(n_sam-1),:,:]= F_inv.copy()
# cost[(n_sam-1),0]=cost_SAE.copy()        
        
# np.savez('F_target_200_train_20predict.npy', R_target=R_target1,F_predict=F_predict,cost=cost) 



# np.save('record_Fmu.npy', record) 

# from matplotlib import pyplot as plt

# plt.plot(np.arange(len(record))+1,np.array(record))
# plt.xlabel("iteration time")
# plt.ylabel("cost")

# #---------------plot the R_target------------------------------
#R=task1['R_test']


# R_target=test_inv[None]
# #R_target=task["R_train"][0,:,:][None,:]
# import plotly.graph_objects as go
# import numpy as np
# #import plotly.graph_objects as go
# import plotly.io as pio
# #pio.renderers.default = 'svg'  # change it back to spyder
# pio.renderers.default = 'browser'

# # Helix equation
# #t = np.linspace(0, 10, 50)
# x, y, z = R_target[0,:,0],R_target[0,:,1],R_target[0,:,2]

# fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z,
#                                   mode='markers')])
# fig.show()
# fig.savefig('Design_structure.png',dpi=300,bbox_inches='tight')

    
