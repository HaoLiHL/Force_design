#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  1 22:26:39 2022

@author: lihao
"""
#!/usr/bin/env python3
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

os.chdir('./')

from utils import io
#import hashlib
import numpy as np
import scipy as sp
import multiprocessing as mp
Pool = mp.get_context('fork').Pool

#from solvers.analytic_train525 import Analytic
from solvers.analytic_aff import Analytic

from utils import perm
from utils.desc import Desc
from utils.desc_inv import Desc_inv
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
    diff_ab_perms = np.empty((n_perm_atom, dim_d))# 66 * 36

    ri_d_desc = np.zeros((1, dim_d, dim_i))
    k = np.zeros((dim_i, dim_i_keep))
    #k = np.empty((dim_i, dim_i_keep))
    
    if use_E_cstr:
        ri_d_desc = np.zeros((1, dim_d, dim_i-1))
        k = np.zeros((dim_i-1, dim_i_keep))
        #diff_ab_perms_t = np.empty((n_perm_atom, dim_d))

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
            

            
            np.dot(ri_d_desc[0].T[index,:], diff_ab_outer_perms[:,index], out=k1)
            k[np.ix_(index,index)]=k1.copy()
        
        
        
        if use_E_cstr:
            ri_desc_perms = np.reshape(
        np.tile(R_desc_val[i, :], n_perm_atom)[tril_perms_lin_mirror], (n_perm_atom, -1), order='F'
    )
            ri_d_desc_perms = np.reshape(
            np.tile(ri_d_desc[0].T, n_perm_atom)[:, tril_perms_lin_mirror], (-1, dim_d, n_perm_atom)
        )

            K_fe = (
                5
                * diff_ab_perms
                / (3 * sig ** 3)
                * (norm_ab_perms[:, None] + sig)
                * np.exp(-norm_ab_perms / sig)[:, None]
            )
            K_fe = np.einsum('ik,jki -> j', K_fe, rj_d_desc_perms)

            
            np.subtract(R_desc[j, :], ri_desc_perms, out=diff_ab_perms_t)
            norm_ab_perms_t = sqrt5 * np.linalg.norm(diff_ab_perms_t, axis=1)
            
            K_fet = (
                5
                * diff_ab_perms_t
                / (3 * sig ** 3)
                * (norm_ab_perms_t[:, None] + sig)
                * np.exp(-norm_ab_perms_t / sig)[:, None]
            )

            K_fet = np.einsum('ik,jki -> j', K_fet, ri_d_desc_perms)


            K[blk_i_full, (j + 1) * dim_i-1] = K_fet
            K[(i + 1) * dim_i-1, blk_j] = K_fe[keep_idxs_3n]  # lower horizontal
       
            K[(i + 1) * dim_i-1, (j + 1) * dim_i-1]  = (
            1 + (norm_ab_perms / sig) * (1 + norm_ab_perms / (3 * sig))
        ).dot(np.exp(-norm_ab_perms / sig)) 
        
        K[blk_i, blk_j] = -k
       # -----------
    
  
    

        
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
            sig = 100,
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
        
        print(' Creating the task of '+str(train_dataset['name'])+' with '+repr(n_train)+' configurations ...\n')

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
        
    
    def train(self, task,
              sig_candid_F =np.arange(10,100,20),
              sig_candid_E = np.arange(10,100,20),
              cprsn_callback=None,
              save_progr_callback=None,  # TODO: document me
              callback=None):
    
            #Train a model based on a training task.
            # R_train  M * N * 3
            # F_train  M * N * 3
        print(' Start the training process ... \n')
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
        #R = task['R_train']  
        #R = R_train
        R_val=task['R_val'] 
        R_train = task['R_train']
        # R is a n_train * 36 matrix        
        tril_perms_lin_mirror = tril_perms_lin
        # tril_perms_lin stores a vectorized permuations of all 12 permuations' descriptor position
        
        #R_atom=R_train
        R_val_atom=R_val
        #R_mirror=
        R_desc_atom, R_d_desc_atom = desc.from_R(R_train,lat_and_inv=lat_and_inv,
                callback=None)
        
        R_desc_val_atom, R_d_desc_val_atom = desc.from_R(R_val_atom,lat_and_inv=lat_and_inv,
                callback=None)
        F_train_atom=[]

        F_val_atom=[]

        
        E_train = task['E_train'].ravel().copy()
        E_val = task['E_val'].ravel().copy()
        
        for i in range(n_type):
            index=np.array(index_diff_atom[i])

            F_train_atom.append(task['F_train'][:,index,:].reshape(int(n_train*(len(index_diff_atom[i])*3)),order='C'))
            F_val_atom.append(task['F_val'][:,index,:].reshape(int(n_val*(len(index_diff_atom[i])*3)),order='C'))


        
        ye_val=E_val #- E_val_mean


        sig_candid=sig_candid_F

        sig_candid1=sig_candid_E

        num_i=sig_candid.shape[0]
        MSA=np.ones((num_i))*1e8
        #RMSE=np.empty((num_i))*1e8
        kernel_time_all=np.zeros((num_i))
        
        for i in range(num_i):
            y_atom= F_train_atom.copy()
            
            
            
            
            if solver == 'analytic':
                #gdml_train: analytic = Analytic(gdml_train, desc, callback=None)
        
                analytic = Analytic(self, desc, callback=callback)
                alphas, inverse_time, kernel_time = analytic.solve_xyz(task,sig_candid[i], R_desc_atom, R_d_desc_atom, tril_perms_lin, tril_perms_lin_mirror, y_atom)

                kernel_time_all[i]=inverse_time+kernel_time
            

            #F_hat_val=[]
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

                K_r=K_r_all[np.ix_(index_x,index_y)]
                F_hat_val_i=np.matmul(K_r,np.array(alphas[ind_i]))
                index_i=np.repeat(np.arange(n_val)*(dim_i),3*len(index_diff_atom[ind_i]))+np.tile(index_eg,n_val)


                F_hat_val_F.append(F_hat_val_i)
                F_star[index_i]=F_hat_val_i
                

            ae=np.mean(np.abs(  np.concatenate(F_hat_val_F)-np.concatenate(F_val_atom)))
            print('   ---This is '+repr(i)+'th task: sigma='+repr(sig_candid[i]))
            print('   ---MAE of F='+repr(ae)+'\n') 
            #ae=np.mean(np.abs(F_hat_val_F-F_val_atom))
            MSA[i]=ae
            

           
            if ae<=min(MSA):
                alphas_ret=alphas.copy()
                F_star_opt=F_star.copy()
                
        kernel_time_ave=np.mean(kernel_time_all)
        print('   -----Overall training  time: '+str(kernel_time_ave)+'seconds.----- \n')        
                
        alphas_opt=alphas_ret
        sig_optim=sig_candid[MSA==min(MSA)]
        lam_e = task['lam']
        lam_f=task['lam']
        sig_optim_E=0
        #sig_candid1=np.arange(15,18,2)
        if task['use_E_cstr']:
            print('   Starting training for energy:    ') 
            batch_size=task['batch_size']
            MSA_E_arr=[]
            MSA_E_arr_o=[]
            for i in range(len(sig_candid1)):
                #print('This is '+repr(i)+'th task: sigma='+repr(sig_candid[i]))
                F_hat_val_E_ave=self._assemble_kernel_mat_Energy(
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
                
                
                MSA_E=np.mean(np.abs(-F_hat_val_E_ave-ye_val))
                RMSE_E=np.sqrt(np.mean((-F_hat_val_E_ave-ye_val)**2))/np.std(ye_val)
                MSA_E_arr.append(MSA_E)
                print('   This is '+repr(i)+'th task: sigma='+repr(sig_candid1[i]))
                print('   This is '+repr(i)+'th task: MAE of E='+repr(MSA_E)) 
                print('   This is '+repr(i)+'th task: RMSE of E='+repr(RMSE_E)) 
                
                
            sig_optim_E=sig_candid1[MSA_E_arr==np.min(MSA_E_arr)]
            
        
        #RMSE=np.sqrt(np.mean((F_hat_val_F-F_val_atom)**2))/np.std(F_val_atom)
        #print(' Optima task : sigma ='+repr(sig_optim)+':  MAE, RMSE='+repr(MSA[i])+' ; '+repr(RMSE))
        print('   Optima task : sigma ='+repr(sig_optim[0])+':  MAE of F ='+repr(min(MSA))+'\n')
        if task['use_E_cstr']:
            print('   Optima task : sigma of e ='+repr(sig_optim_E)+':  MAE of E ='+repr(np.min(MSA_E_arr))+'\n')

        #return sig_optim[0],sig_optim_E,alphas_opt
        trained_model = {'sig_F':sig_optim[0], 'sig_E':sig_optim_E,'alpha':alphas_opt}
        
        #print(' Start the training process ... \n')
        print('Finished the training process \n')
        return trained_model
    
    def predict(self, task,trained_model,
            cprsn_callback=None,
            save_progr_callback=None,  # TODO: document me
            callback=None):
        
        print('Start the predicting process ... \n')
        
        sig_optim = trained_model['sig_F']
        sig_candid1_opt = trained_model['sig_E']
        alphas_opt = trained_model['alpha']
        
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
            
            
        #print('This is tesing task : sigma='+repr(sig_optim))
        alphas=alphas_opt
        
        
        # if task['use_E_cstr']:
        #     alphas_E = alphas[-n_train:]
        #     alphas_F = alphas[:-n_train]
        #F_hat_val=[]
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
        F_star_L_arr=np.empty([n_val*3*n_atoms])
        F_star_H_arr=np.empty([n_val*3*n_atoms])
        F_star_L = []
        F_star_H = []
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
                F_star_L_arr[index_i]=F_hat_val_i+norm.ppf(0.025,0,np.sqrt(np.diag(S2*K_star_star)))

                F_star_H.append(F_hat_val_i+norm.ppf(0.975,0,np.sqrt(np.diag(S2*K_star_star))))
                F_star_H_arr[index_i]=F_hat_val_i+norm.ppf(0.975,0,np.sqrt(np.diag(S2*K_star_star)))
        if uncertainty: 
            P_C=np.sum( np.array(np.concatenate(F_star_L)<=np.concatenate(F_val_atom)) & np.array(np.concatenate(F_star_H)>=np.concatenate(F_val_atom)))/(n_val*3*n_atoms)
            A_L=np.mean( np.concatenate(F_star_H)-np.concatenate(F_star_L))
            print('   The percentage of coverage by 95% CI is '+repr(P_C*100)+' %.') 
            print('   The average length of the 95% CI is '+repr(A_L)) 
        #a=(np.concatenate(F_star_L)<=np.concatenate(F_val_atom) and np.concatenate(F_star_H)>=np.concatenate(F_val_atom))
        ae=np.mean(np.abs(  np.concatenate(F_hat_val_F)-np.concatenate(F_val_atom)))
        RMSE_F=np.sqrt(np.mean((np.concatenate(F_hat_val_F)-np.concatenate(F_val_atom))**2))/np.std(np.concatenate(F_val_atom))
        print('   This is the  MAE of F='+repr(ae)) 
        print('   This is the  RMSE of F='+repr(RMSE_F)+'\n') 
        

        
        
        #ae=np.mean(np.abs(F_hat_val_F-F_val_atom))
        MAE=ae
        
        if task['use_E_cstr']:
            lam_f=task['lam']
            lam_e=task['lam']
            print('   Starting training for energy:    ') 
            MSA_E_arr=[]
            start = timeit.default_timer()
        
                #print('This is '+repr(i)+'th task: sigma='+repr(sig_candid[i]))
            F_hat_val_E_ave=self._assemble_kernel_mat_Energy(
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
                use_E_cstr=task['use_E_cstr'],
                col_idxs= np.s_[:],
                callback=None,
            )
            stop = timeit.default_timer()
            dur_s = (stop - start)
            MSA_E=np.mean(np.abs(-F_hat_val_E_ave-ye_val))
            RMSE_E=np.sqrt(np.mean((-F_hat_val_E_ave-ye_val)**2))/np.std(ye_val)
            MSA_E_arr.append(MSA_E)
            #print(' This is the testing task: It takes'+repr(dur_s)+'second; each estimation takes'+repr(dur_s/n_val)+' seconds') 
            print('   This is the testing task: MAE of E='+repr(MSA_E)) 
            print('   This is the testing task: RMSE of E='+repr(RMSE_E)+'/n') 
        
       
        print('Finished! \n')
        prediction = {'predicted_force':F_star.reshape(n_val,n_atoms,-1),
                      'CI_LB':F_star_L_arr.reshape(n_val,n_atoms,-1),
                      'CI_UB':F_star_H_arr.reshape(n_val,n_atoms,-1)
                      }
        
       
        return prediction
    
    
    def inverseF(self, task1,trained_model,ind_initial,F_target,lr,
            cprsn_callback=None,
            save_progr_callback=None,  # TODO: document me
            callback=None):
        
        # everytime after call this funciton , the result returned from the train() function will change 2022/05/05
        sig_optim = trained_model['sig_F']
        #sig_candid1_opt = trained_model['sig_E']
        alphas_opt = trained_model['alpha']
        
        task = dict(task1)
        #solver = task['solver_name']
        #batch_size=task['batch_size']
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
        
        while kk<5 and cost_SAE>0.3:
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
            
            
           
            #F_val_atom=[]
                #F_val_atom=task['F_val'].ravel().copy()
            
            #E_train = task['E_train'].ravel().copy()
            #E_val = task['E_test'].ravel().copy()
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
            
            #F_star=np.empty([n_val*3*n_atoms])
            F_hat=np.empty([n_atoms,3])
            
            drl=np.zeros([3*n_atoms])
            for ind_i in range(n_type):
                index_eg=np.tile(np.arange(3),len(index_diff_atom[ind_i]))+3*np.repeat(index_diff_atom[ind_i],3)
    
                index_x=np.repeat(np.arange(n_val)*(dim_i),3*len(index_diff_atom[ind_i]))+np.tile(index_eg,n_val)
                index_y=np.repeat(np.arange(n_train)*(dim_i),3*len(index_diff_atom[ind_i]))+np.tile(index_eg,n_train)
    
     
                K_r=K_r_all[np.ix_(index_x,index_y)]
                F_hat_val_i=np.matmul(K_r,np.array(alphas[ind_i]))
                
                F_hat[index_diff_atom[ind_i],:]=F_hat_val_i.reshape(len(index_diff_atom[ind_i]),-1).copy()
                #index_i=np.repeat(np.arange(n_val)*(dim_i),3*len(index_diff_atom[ind_i]))+np.tile(index_eg,n_val)
                
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
            cost_SAE=np.sum(np.abs(np.concatenate(F_hat_val_F)-np.concatenate(F_hat_val_target)))
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
            R_val_atom[0,:,:]= R_val_atom_last - drl.reshape(n_atoms,3)*lr
            R_design.append(R_val_atom_last)
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
            use_E_cstr=False,
            col_idxs=np.s_[:],  # TODO: document me
            callback=None,
        ):
        r"""
            Compute force field kernel matrix.
        """
        #global glob
    
            # Note: This function does not support unsorted (ascending) index arrays.
            # if not isinstance(col_idxs, slice):
            #    assert np.array_equal(col_idxs, np.sort(col_idxs))
        #batch_size=int(10)
        n_val, dim_d = R_d_desc_val_atom.shape[:2]
        n_train, dim_d = R_d_desc.shape[:2]  #R_d_desc.shape (n_train, 66, 3)
        n_total=n_val+n_train
        n_atoms = int((1 + np.sqrt(8 * dim_d + 1)) / 2)
        dim_i = 3 * n_atoms
        #n_train , dim_d 66
        n_perms = int(len(tril_perms_lin) / dim_d)
        E_predict = np.empty(n_val)
        
        keep_idxs_3n = slice(None)
        
        K_EE_all = np.empty((n_val+n_train,n_val+n_train))
        K_EF_all = np.empty((n_val+n_train,n_val*3*n_atoms))
        K_FE_all = np.empty((n_val*3*n_atoms,n_val+n_train))
        
        K_FsFs_all = np.empty((n_val*3*n_atoms,n_val*3*n_atoms))
        
        mat52_base_div = 3 * sig ** 4
        sqrt5 = np.sqrt(5.0)
        sig_pow2 = sig ** 2
        
        R_desc_atom = np.row_stack((R_desc_val_atom,R_desc))
        R_d_desc_atom = R_d_desc_val_atom
        for j in range(n_val+n_train):
            
            blk_j = slice(j*dim_i , (j + 1)*dim_i )
            rj_desc_perms = np.reshape(
        np.tile(R_desc_atom[j, :], n_perms)[tril_perms_lin_mirror], (n_perms, -1), order='F'
    )
            
            if j<n_val:
                rj_d_desc = desc.d_desc_from_comp(R_d_desc_atom[j, :, :])[0][
                    :, keep_idxs_3n
                ]  # convert descriptor back to full representation
                # rj_d_desc 66 * 36
    
                rj_d_desc_perms = np.reshape(
                    np.tile(rj_d_desc.T, n_perms)[:, tril_perms_lin_mirror], (-1, dim_d, n_perms)
                )
                
                
            
            diff_ab_perms = np.empty((n_perms, dim_d))
            diff_ab_perms_t = np.empty((n_perms, dim_d))
            dim_i_keep =dim_i
            diff_ab_outer_perms = np.empty((dim_d, dim_i_keep)) 
            ri_d_desc = np.zeros((1, dim_d, dim_i))
            
            for i in range(n_val+n_train):
                blk_i = slice(i*dim_i , (i + 1)*dim_i )
                np.subtract(R_desc_atom[i, :], rj_desc_perms, out=diff_ab_perms)
                norm_ab_perms = sqrt5 * np.linalg.norm(diff_ab_perms, axis=1)
                
                if i<n_val:  
                    desc.d_desc_from_comp(R_d_desc_atom[i, :, :], out=ri_d_desc)
                    ri_desc_perms = np.reshape(
                        np.tile(R_desc_atom[i, :], n_perms)[tril_perms_lin_mirror], (n_perms, -1), order='F'
                    )
                    ri_d_desc_perms = np.reshape(
                np.tile(ri_d_desc[0].T, n_perms)[:, tril_perms_lin_mirror], (-1, dim_d, n_perms)
            )
                    np.subtract(R_desc_atom[j, :], ri_desc_perms, out=diff_ab_perms_t)
                    norm_ab_perms_t = sqrt5 * np.linalg.norm(diff_ab_perms_t, axis=1)
                    
                    K_fet = (
                        5
                        * diff_ab_perms_t
                        / (3 * sig ** 3)
                        * (norm_ab_perms_t[:, None] + sig)
                        * np.exp(-norm_ab_perms_t / sig)[:, None]
                    )
                    K_fet = np.einsum('ik,jki -> j', K_fet, ri_d_desc_perms)
                    K_FE_all[blk_i, j] = K_fet
                
                if j<n_val:
                    K_fe = (
                        5
                        * diff_ab_perms
                        / (3 * sig ** 3)
                        * (norm_ab_perms[:, None] + sig)
                        * np.exp(-norm_ab_perms / sig)[:, None]
                    )
                    K_fe = np.einsum('ik,jki -> j', K_fe, rj_d_desc_perms)
                    K_EF_all[i, blk_j] = K_fe[keep_idxs_3n]
                    
                    
                    
                    k = np.empty((dim_i, dim_i))
                    if i<n_val:
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
                        
                        desc.d_desc_from_comp(R_d_desc_atom[i, :, :], out=ri_d_desc)
                        
                        np.dot(ri_d_desc[0].T, diff_ab_outer_perms, out=k)
                        K_FsFs_all[blk_i,blk_j]=-k
                

                K_EE_all[i,j]  =(
                1 + (norm_ab_perms / sig) * (1 + norm_ab_perms / (3 * sig))
            ).dot(np.exp(-norm_ab_perms / sig)) 
                
        
        n_batch=int(n_val/batch_size)
        
        for l in range(n_batch):
            H=np.concatenate([np.repeat([1],n_train),np.repeat([0],3*n_atoms*batch_size)]).reshape(-1,1)
            #H=np.concatenate(np.repeat([1],n_train)).reshape(-1,1)
            R_E=np.empty((n_train+3*n_atoms*batch_size,n_train+3*n_atoms*batch_size))
            R_E[np.ix_(np.arange(n_train),np.arange(n_train))]=K_EE_all[
                np.ix_(np.arange(n_val,n_total),np.arange(n_val,n_total))].copy()
            
            Kr_E=np.empty((batch_size,n_train+3*n_atoms*batch_size))
            
            Y_vec=np.empty((n_train+3*n_atoms*batch_size))
            
            Y_vec[np.arange(n_train)]=-E_train.copy()
            index = np.arange(l*3*n_atoms*batch_size,(l+1)*3*n_atoms*batch_size)
            R_E[np.ix_(np.arange(n_train),np.arange(n_train,n_train+3*n_atoms*batch_size))]=K_EF_all[
                np.ix_(np.arange(n_val,n_total),index)].copy()
            R_E[np.ix_(np.arange(n_train,n_train+3*n_atoms*batch_size),np.arange(n_train))]=K_FE_all[
                np.ix_(index,np.arange(n_val,n_total))].copy()
            R_E[np.ix_(np.arange(n_train,n_train+3*n_atoms*batch_size),np.arange(n_train,n_train+3*n_atoms*batch_size))]=K_FsFs_all[
                np.ix_(index,index)].copy()
            Y_vec[np.arange(n_train,n_train+3*n_atoms*batch_size)]=F_star[index].copy()
            
            #index_E=np.concatenate((l,np.arange(n_val,n_total)))
            index_l=np.arange(l*batch_size,(l+1)*batch_size)
            Kr_E[np.ix_(np.arange(batch_size),np.arange(n_train))] = K_EE_all[np.ix_(index_l,np.arange(n_val,n_total))].copy()
            Kr_E[np.ix_(np.arange(batch_size),np.arange(n_train,n_train+3*n_atoms*batch_size))] =  K_EF_all[np.ix_(index_l,np.arange(l*3*n_atoms*batch_size,(l+1)*3*n_atoms*batch_size))].copy()
            
            add=np.diag(np.append(np.repeat(lam_e,n_train),np.repeat(lam_f,batch_size*n_atoms*3)))
            R_E +=add
            #R_E[np.diag_indices_from(R_E)] += lam_e
            #m=
            #R_m=R_E[np.ix_(np.arange(n_train),np.arange(n_train))]
           
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                try:
                    L, lower = sp.linalg.cho_factor(
                                    R_E, overwrite_a=True, check_finite=False
                                )
                    alpha_star=sp.linalg.cho_solve(
                        (L, lower), np.array(Y_vec), overwrite_b=True, check_finite=False
                    )
                    
                    alpha_m=sp.linalg.cho_solve(
                        (L, lower), H[:,0], overwrite_b=True, check_finite=False
                    )
                    
                except np.linalg.LinAlgError: 
                    alpha_star  = sp.linalg.solve(
                            R_E,  Y_vec, overwrite_a=True, overwrite_b=True, check_finite=False
                        )
                    alpha_m  = sp.linalg.solve(
                            R_E,  H[:,0], overwrite_a=True, overwrite_b=True, check_finite=False
                        )
                   
            m_inv=1/np.matmul(H.T,alpha_m)
            m=m_inv*np.matmul(H.T,alpha_star)

            #print(m)
            E_predict[index_l]=m+np.matmul(Kr_E,(alpha_star-alpha_m*m).T)
            #E_predict[index_l]=(m+np.matmul(Kr_E,(alpha_star-alpha_m)*m).T)).T-np.matmul(Kr_E,alpha_m)*m

        return E_predict
    
# compile scripts to read energy and force from the output from physcis-baed calculation
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
                    input_write.write("%.8lf\t" % R_design[index_run, index_atom, i_dimension])

                input_write.write("\n")
            
            # attention!!!
            # future features
            # some of the following calculation settings can be cahnged into variables
            # that allow users to make changes according to their needs
            input_write.write("$end\n")
            input_write.write("\n")
            input_write.write("$rem\n")
            input_write.write("jobtype                force\n")
            input_write.write("exchange               HF\n")
            input_write.write("correlation            ")
            input_write.write(str(computational_method[0]))
            input_write.write("\n")
            input_write.write("basis                  ")
            input_write.write(str(computational_method[1]))
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
            #os.remove(temp_energy_filename)
            #os.remove(energy_filename)
            if np.any(single_energy) is False:
                print('!!!!!!!! no energy read')
            else:
                E_new_simulation[index_run] = single_energy

            if computational_method[0] == 'mp2':
                command_grep_force = ["grep", "-A",  str(math.ceil(num_atoms/6)*4), "\"Full Analytical Gradient of MP2 Energy\"", simulator_output_filename, ">", temp_force_filename]
            elif computational_method[0] == 'PBE':
                command_grep_force = ["grep", "-A",  str(math.ceil(num_atoms/6)*4), "\"Calculating analytic gradient of the SCF energy\"", simulator_output_filename, ">", temp_force_filename]
            grep_force = subprocess.Popen(' '.join(command_grep_force), shell=True)
            grep_force.wait()
            command_force_energy = [force_converter_script_path, temp_force_filename, force_filename]
            conv_force = subprocess.Popen(' '.join(command_force_energy), shell=True)
            conv_force.wait()
            single_force = np.loadtxt(force_filename)
            #os.remove(temp_force_filename)
            #os.remove(force_filename)
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
  
  


