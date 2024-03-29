#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 14:57:57 2021

@author: lihao
"""
import numpy as np
import scipy as sp

import multiprocessing as mp

Pool = mp.get_context('fork').Pool

from functools import partial
#from scipy import spatial
import timeit

try:
    import torch
except ImportError:
    _has_torch = False
else:
    _has_torch = True
    
def _pbc_diff(diffs, lat_and_inv, use_torch=False):
    """
    Clamp differences of vectors to super cell.
    Parameters
    ----------
        diffs : :obj:`numpy.ndarray`
            N x 3 matrix of N pairwise differences between vectors `u - v`
        lat_and_inv : tuple of :obj:`numpy.ndarray`
            Tuple of 3 x 3 matrix containing lattice vectors as columns and its inverse.
        use_torch : boolean, optional
            Enable, if the inputs are PyTorch objects.
    Returns
    -------
        :obj:`numpy.ndarray`
            N x 3 matrix clamped differences
    """

    lat, lat_inv = lat_and_inv

    if use_torch and not _has_torch:
        raise ImportError(
            'Optional PyTorch dependency not found! Please run \'pip install sgdml[torch]\' to install it or disable the PyTorch option.'
        )

    if use_torch:
        c = lat_inv.mm(diffs.t())
        diffs -= lat.mm(c.round()).t()
    else:
        c = lat_inv.dot(diffs.T)
        diffs -= lat.dot(np.rint(c)).T

    return diffs

def _pdist(r, lat_and_inv=None):  # TODO: update return (no squareform anymore)
    """
    Compute pairwise Euclidean distance matrix between all atoms.
    Parameters
    ----------
        r : :obj:`numpy.ndarray`
            Array of size 3N containing the Cartesian coordinates of
            each atom.
        lat_and_inv : tuple of :obj:`numpy.ndarray`, optional
            Tuple of 3x3 matrix containing lattice vectors as columns and its inverse.
    Returns
    -------
        :obj:`numpy.ndarray`
            Array of size N x N containing all pairwise distances between atoms.
    """

    r = r.reshape(-1, 3)
    n_atoms = r.shape[0]

    if lat_and_inv is None:
        pdist = sp.spatial.distance.pdist(r, 'euclidean')
    else:
        pdist = sp.spatial.distance.pdist(
            r, lambda u, v: np.linalg.norm(_pbc_diff(u - v, lat_and_inv))
        )

    tril_idxs = np.tril_indices(n_atoms, k=-1)
    return sp.spatial.distance.squareform(pdist, checks=False)[tril_idxs]

def _r_to_desc(r, pdist, coff=None):
    """
    Generate descriptor for a set of atom positions in Cartesian
    coordinates.
    Parameters
    ----------
        r : :obj:`numpy.ndarray`
            Array of size 3N containing the Cartesian coordinates of
            each atom.
        pdist : :obj:`numpy.ndarray`
            Array of size N x N containing the Euclidean distance
            (2-norm) for each pair of atoms.
    Returns
    -------
        :obj:`numpy.ndarray`
            Descriptor representation as 1D array of size N(N-1)/2
    """

    # Add singleton dimension if input is (,3N).
    if r.ndim == 1:
        r = r[None, :]

    if coff is None:

        return 1.0 / pdist
    else:

        coff_dist, coff_slope = coff
        cutoff_factor = -1.0 / (1.0 + np.exp(-coff_slope * (pdist - coff_dist))) + 1

        return cutoff_factor / pdist

def _r_to_d_desc(r, pdist, lat_and_inv=None, coff=None):  # TODO: fix documentation!
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
        pdist : :obj:`numpy.ndarray`
            Array of size N x N containing the Euclidean distance
            (2-norm) for each pair of atoms.
        lat_and_inv : tuple of :obj:`numpy.ndarray`, optional
            Tuple of 3x3 matrix containing lattice vectors as columns and its inverse.
    Returns
    -------
        :obj:`numpy.ndarray`
            Array of size N(N-1)/2 x 3N containing all partial
            derivatives of the descriptor.
    """

    r = r.reshape(-1, 3)
    pdiff = r[:, None] - r[None, :]  # pairwise differences ri - rj

    n_atoms = r.shape[0]
    i, j = np.tril_indices(n_atoms, k=-1)

    pdiff = pdiff[i, j, :]  # lower triangular

    if lat_and_inv is not None:
        pdiff = _pbc_diff(pdiff, lat_and_inv)

    if coff is None:
        #d_desc_elem = -pdiff / (pdist ** 3)[:, None]
        d_desc_elem = pdiff / (pdist ** 3)[:, None] #original
    else:
        coff_dist, coff_slope = coff

        cutoff_factor = 1.0 - 1.0 / (
            np.exp(-coff_slope * (pdist[:, None] - coff_dist)) + 1.0
        )
        cutoff_term = (
            coff_slope
            * np.exp(-coff_slope * (pdist[:, None] - coff_dist))
            / (pdiff * (np.exp(-coff_slope * (pdist[:, None] - coff_dist)) + 1) ** 2)
        )
        cutoff_term[cutoff_term == np.inf] = 0
        d_desc_elem = cutoff_factor * pdiff / (pdist ** 3)[:, None] + cutoff_term

    return d_desc_elem

def _r_to_d2_desc(r, k,pdist, lat_and_inv=None, coff=None):  # TODO: fix documentation!
    """
    already checked!
    Generate descriptor Hessian for a set of atom positions in
    Cartesian coordinates.
    This method can apply the minimum-image convention as periodic
    boundary condition for distances between atoms, given the edge
    length of the (square) unit cell.
    Parameters
    ----------
        r : :obj:`numpy.ndarray`
            Array of size 3N containing the Cartesian coordinates of
            each atom.
        pdist : :obj:`numpy.ndarray`
            Array of size N x N containing the Euclidean distance
            (2-norm) for each pair of atoms.
        d_index: 1:3N represent the d^2/d_ d_index (d_index=1:3N)
        k =d_index % 3  k=0,1,2, means x,y,z
        lat_and_inv : tuple of :obj:`numpy.ndarray`, optional
            Tuple of 3x3 matrix containing lattice vectors as columns and its inverse.
    Returns
    -------
        :obj:`numpy.ndarray`
            Array of size N(N-1)/2 x 3N containing all partial
            derivatives of the descriptor.
    """

    r = r.reshape(-1, 3)
    pdiff = r[:, None] - r[None, :]  # pairwise differences ri - rj

    n_atoms = r.shape[0]
    i, j = np.tril_indices(n_atoms, k=-1)

    pdiff = pdiff[i, j, :]  # lower triangular # (66,3)   

    if lat_and_inv is not None:
        pdiff = _pbc_diff(pdiff, lat_and_inv)

    #if coff is None:
        #d_desc_elem = -pdiff / (pdist ** 3)[:, None]
        # pdist (66,)
        #d_desc_elem = pdiff / (pdist ** 3)[:, None] #original
    
    # if d_i ==d_i_index,  \partial^2 D_ij /\partial r_{ik} \partial r_{jk}
    #checked: d_desc_elem_1
    d_desc_elem_1 = ((pdist ** 2)[:,None] - 3* pdiff ** 2)/(pdist ** 5)[:, None]   # if d_i ==d_i_index
    #k = d_index % 3
    # if d_i !=d_i_index,  \partial^2 D_ij /\partial r_{ik} \partial r_{jl}
    d_desc_elem_k = -(3* pdiff * pdiff[:,k][:,None])/(pdist ** 5)[:, None]
    # d_desc_elem_k: \partial^2 D_ij /\partial r_{ik} \partial r_{jk}
    
    return d_desc_elem_1,d_desc_elem_k

def _r_to_d2_desck(r, k,pdist, lat_and_inv=None, coff=None):  # TODO: fix documentation!
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
        pdist : :obj:`numpy.ndarray`
            Array of size N x N containing the Euclidean distance
            (2-norm) for each pair of atoms.
        d_index: 1:3N represent the d^2/d_ d_index (d_index=1:3N)
        k =d_index % 3  k=0,1,2
        lat_and_inv : tuple of :obj:`numpy.ndarray`, optional
            Tuple of 3x3 matrix containing lattice vectors as columns and its inverse.
    Returns
    -------
        :obj:`numpy.ndarray`
            Array of size N(N-1)/2 x 3N containing all partial
            derivatives of the descriptor.
    """

    r = r.reshape(-1, 3)
    pdiff = r[:, None] - r[None, :]  # pairwise differences ri - rj

    n_atoms = r.shape[0]
    i, j = np.tril_indices(n_atoms, k=-1)

    pdiff = pdiff[i, j, :]  # lower triangular # (66,3)   

    if lat_and_inv is not None:
        pdiff = _pbc_diff(pdiff, lat_and_inv)

    #if coff is None:
        #d_desc_elem = -pdiff / (pdist ** 3)[:, None]
        # pdist (66,)
        #d_desc_elem = pdiff / (pdist ** 3)[:, None] #original
    #d_desc_elem_1 = ((pdist ** 2)[:,None]+3* pdiff ** 2)/(pdist ** 5)[:, None]   # if d_i ==d_i_index
    #k = d_index % 3
    d_desc_elem_k = (3* pdiff * pdiff[:,k][:,None])/(pdist ** 5)[:, None]
    
    return d_desc_elem_k

def _from_r(r, k, lat_and_inv=None, coff=None):
    """
    Generate descriptor and its Jacobian for one molecular geometry
    in Cartesian coordinates.
    Parameters
    ----------
        r : :obj:`numpy.ndarray`
            Array of size 3N containing the Cartesian coordinates of
            each atom.
        lat_and_inv : tuple of :obj:`numpy.ndarray`, optional
            Tuple of 3 x 3 matrix containing lattice vectors as columns and its inverse.
    Returns
    -------
        :obj:`numpy.ndarray`
            Descriptor representation as 1D array of size N(N-1)/2
        :obj:`numpy.ndarray`
            Array of size N(N-1)/2 x 3N containing all partial
            derivatives of the descriptor.
    """

    # Add singleton dimension if input is (,3N).
    if r.ndim == 1:
        r = r[None, :]

    pd = _pdist(r, lat_and_inv)

    r_desc = _r_to_desc(r, pd, coff=coff)
    r_d_desc = _r_to_d_desc(r, pd, lat_and_inv, coff=coff)
    
    #k=0
    r_d2_desc_1,r_d2_desc_k = _r_to_d2_desc(r, k, pd, lat_and_inv, coff=coff)
    # r_d2_desc_1 = _r_to_d2_desc(r, k, pd, lat_and_inv, coff=coff)
    # r_d2_desc_k = _r_to_d2_desck(r, k, pd, lat_and_inv, coff=coff)

    return  r_d2_desc_1, r_d2_desc_k

class Desc_inv(object):
    def __init__(self, n_atoms, interact_cut_off=None, max_processes=None):
        """
        Generate descriptors and their Jacobians for molecular geometries,
        including support for periodic boundary conditions.
        Parameters
        ----------
                n_atoms : int
                        Number of atoms in the represented system.
                max_processes : int, optional
                        Limit the max. number of processes. Otherwise
                        all CPU cores are used. This parameters has no
                        effect if `use_torch=True`.
        """

        self.n_atoms = n_atoms
        self.dim_i = 3 * n_atoms

        # Size of the resulting descriptor vector.
        self.dim = (n_atoms * (n_atoms - 1)) // 2

        # Precompute indices for nonzero entries in desriptor derivatives.
        self.d_desc_mask = np.zeros((n_atoms, n_atoms - 1), dtype=np.int)
        for a in range(n_atoms):  # for each partial derivative
            rows, cols = np.tril_indices(n_atoms, -1)
            self.d_desc_mask[a, :] = np.concatenate(
                [np.where(rows == a)[0], np.where(cols == a)[0]]
            )

        self.dim_range = np.arange(self.dim)  # [0, 1, ..., dim-1]

        # Precompute indices for nonzero entries in desriptor derivatives.

        self.M = np.arange(1, n_atoms)  # indexes matrix row-wise, skipping diagonal
        for a in range(1, n_atoms):
            self.M = np.concatenate((self.M, np.delete(np.arange(n_atoms), a)))

        self.A = np.repeat(
            np.arange(n_atoms), n_atoms - 1
        )  # [0, 0, ..., 1, 1, ..., 2, 2, ...]

        self.max_processes = max_processes

        # NEW: cutoff
        self.coff_dist = (
            interact_cut_off
            if not hasattr(interact_cut_off, '__iter__')
            else interact_cut_off.item()
        )  # TODO: that's a hack :(
        self.coff_slope = 10
        # NEW

        self.tril_indices = np.tril_indices(n_atoms, k=-1)
        


    def from_R(self, R, index_i, lat_and_inv=None, callback=None):
        """
        Generate descriptor and its Jacobian for multiple molecular geometries
        in Cartesian coordinates.
        Parameters
        ----------
            R : :obj:`numpy.ndarray`
                Array of size M x 3N containing the Cartesian coordinates of
                each atom.
            lat_and_inv : tuple of :obj:`numpy.ndarray`, optional
                Tuple of 3 x 3 matrix containing lattice vectors as columns and its inverse.
            callback : callable, optional
                Descriptor and descriptor Jacobian generation status.
                    current : int
                        Current progress (number of completed descriptors).
                    total : int
                        Task size (total number of descriptors to create).
                    sec_disp_str : :obj:`str`, optional
                        Once complete, this string contains the
                        time it took complete this task (seconds).
        Returns
        -------
            :obj:`numpy.ndarray`
                Array of size M x N(N-1)/2 containing the descriptor representation
                for each geometry.
            :obj:`numpy.ndarray`
                Array of size M x N(N-1)/2 x 3N containing all partial
                derivatives of the descriptor for each geometry.
        """

        # Add singleton dimension if input is (,3N).
        if R.ndim == 1:
            R = R[None, :]

        M = R.shape[0]
        k=index_i%3
        if M == 1:
            return _from_r(R, k,lat_and_inv)

        R_desc = np.empty([M, self.dim])
        R_d_desc = np.empty([M, self.dim, 3])
        R_d2_desc_1 = np.empty([M, self.dim, 3])
        R_d2_desc_k = np.empty([M, self.dim, 3])
        
        

        #i=0
        #R_d2_desc_1[i,:,:],R_d2_desc_k[i,:,:]=_from_r(R[0,:],k)
        #Generate descriptor and their Jacobians
        start = timeit.default_timer()
        
        
        pool = Pool(self.max_processes)

        coff = None if self.coff_dist is None else (self.coff_dist, self.coff_slope)

        for i, r_desc_r_d_desc in enumerate(
            pool.imap(partial(_from_r, k=k, lat_and_inv=lat_and_inv, coff=coff), R)
        ):
            R_d2_desc_1[i,:,:],R_d2_desc_k[i,:,:] = r_desc_r_d_desc

            if callback is not None and i < M - 1:
                callback(i, M - 1)

        pool.close()
        pool.join()  # Wait for the worker processes to terminate (to measure total runtime correctly).
        stop = timeit.default_timer()

        if callback is not None:
            dur_s = (stop - start) / 2
            sec_disp_str = 'took {:.1f} s'.format(dur_s) if dur_s >= 0.1 else ''
            callback(M, M, sec_disp_str=sec_disp_str)

        return R_d2_desc_1, R_d2_desc_k

    def perm(self, perm):
        """
        Convert atom permutation to descriptor permutation.
        A permutation of N atoms is converted to a permutation that acts on
        the corresponding descriptor representation. Applying the converted
        permutation to a descriptor is equivalent to permuting the atoms
        first and then generating the descriptor.
        Parameters
        ----------
            perm : :obj:`numpy.ndarray`
                Array of size N containing the atom permutation.
        Returns
        -------
            :obj:`numpy.ndarray`
                Array of size N(N-1)/2 containing the corresponding
                descriptor permutation.
        """

        n = len(perm)

        rest = np.zeros((n, n))
        rest[np.tril_indices(n, -1)] = list(range((n ** 2 - n) // 2))
        rest = rest + rest.T
        rest = rest[perm, :]
        rest = rest[:, perm]

        return rest[np.tril_indices(n, -1)].astype(int)

    # Private

    # multiplies descriptor(s) jacobian with 3N-vector(s) from the right side
    def d_desc_dot_vec(self, R_d_desc, vecs):

        if R_d_desc.ndim == 2:
            R_d_desc = R_d_desc[None, ...]

        if vecs.ndim == 1:
            vecs = vecs[None, ...]

        i, j = self.tril_indices

        vecs = vecs.reshape(vecs.shape[0], -1, 3)
        return np.einsum('kji,kji->kj', R_d_desc, vecs[:, j, :] - vecs[:, i, :])

    # multiplies descriptor(s) jacobian with N(N-1)/2-vector(s) from the left side
    def vec_dot_d_desc(self, R_d_desc, vecs, out=None):

        if R_d_desc.ndim == 2:
            R_d_desc = R_d_desc[None, ...]

        if vecs.ndim == 1:
            vecs = vecs[None, ...]

        assert (
            R_d_desc.shape[0] == 1
            or vecs.shape[0] == 1
            or R_d_desc.shape[0] == vecs.shape[0]
        )  # either multiple descriptors or multiple vectors at once, not both (or the same number of both, than it will must be a multidot)

        n = np.max((R_d_desc.shape[0], vecs.shape[0]))
        i, j = self.tril_indices

        out = np.zeros((n, self.n_atoms, self.n_atoms, 3))
        out[:, i, j, :] = R_d_desc * vecs[..., None]
        out[:, j, i, :] = -out[:, i, j, :]
        return out.sum(axis=1).reshape(n, -1)

        # if out is None or out.shape != (n, self.n_atoms*3):
        #    out = np.zeros((n, self.n_atoms*3))

        # R_d_desc_full = np.zeros((self.n_atoms, self.n_atoms, 3))
        # for a in range(n):

        #   R_d_desc_full[i, j, :] = R_d_desc * vecs[a, :, None]
        #    R_d_desc_full[j, i, :] = -R_d_desc_full[i, j, :]
        #    out[a,:] = R_d_desc_full.sum(axis=0).ravel()

        # return out

    # inflate desc
    # add_to_vec - instead of creating a new array for the expanded descriptor Jacobian, add it onto an existing representation
    def d_desc_from_comp(self, R_d_desc, out=None):
        # R_d_desc 66*3
        # NOTE: out must be filled with zeros!

        if R_d_desc.ndim == 2:
            R_d_desc = R_d_desc[None, ...]

        n = R_d_desc.shape[0]
        i, j = self.tril_indices

        if out is None:
            out = np.zeros((n, self.dim, self.n_atoms, 3))
        else:
            out = out.reshape(n, self.dim, self.n_atoms, 3)

        out[:, self.dim_range, j, :] = R_d_desc
        out[:, self.dim_range, i, :] = -R_d_desc

        return out.reshape(-1, self.dim, self.dim_i)
    
    def d2_desc_from_comp(self, R_d2_desc_1, R_d2_desc_k, index_d, out=None):
        # R_d_desc 66*3
        # NOTE: out must be filled with zeros!
        # R_d2_desc_1 : 

        if R_d2_desc_1.ndim == 2:
            R_d2_desc_1 = R_d2_desc_1[None, ...]
            R_d2_desc_k = R_d2_desc_k[None, ...]

        n = R_d2_desc_1.shape[0]
        i, j = self.tril_indices

        if out is None:
            # out : 1* 66* 12 * 3
            # out: 1* N(N-1)/2 * N * 3
            out = np.zeros((n, self.dim, self.n_atoms, 3))
        else:
            out = out.reshape(n, self.dim, self.n_atoms, 3)
        
        r_n=index_d // 3  # the index of atom
        r_i=index_d % 3   # the index of x,y,z (0,1,2)
        
        # old 2022/06/09
        # out[:, self.dim_range[j==r_n], j[j==r_n], r_i] = R_d2_desc_1[:,self.dim_range[j==r_n],r_i]
        # #out[:, self.dim_range[j==r_n], j[j==r_n], np.arange(3)[np.arange(3)!=r_i]] = R_d2_desc_k[np.ix_(self.dim_range[j==r_n],np.arange(3)[np.arange(3)!=0])]
        # out[:, self.dim_range[j==r_n], j[j==r_n], :][:,:,np.arange(3)[np.arange(3)!=r_i]]= R_d2_desc_k[0,:,:][np.ix_(self.dim_range[j==r_n],np.arange(3)[np.arange(3)!=r_i])]
        
        
        # out[:, self.dim_range[i==r_n], i[i==r_n], r_i] = -R_d2_desc_1[:,self.dim_range[i==r_n],r_i][None]
        # #out[:, self.dim_range[j==r_n], i[i==r_n], np.arange(3)!=r_i] = -R_d2_desc_k[self.dim_range[j==r_n],np.arange(3)[np.arange(3)!=0]]
        # out[:, self.dim_range[i==r_n], i[i==r_n], :][:,:,np.arange(3)[np.arange(3)!=r_i]] = -R_d2_desc_k[0,:,:][np.ix_(self.dim_range[i==r_n],np.arange(3)[np.arange(3)!=r_i])]
        
        # #out[:, self.dim_range, i, :] = -R_d_desc
        
        
        #out[:, self.dim_range[j==r_n], j[j==r_n], np.arange(3)[np.arange(3)!=r_i]] = R_d2_desc_k[np.ix_(self.dim_range[j==r_n],np.arange(3)[np.arange(3)!=0])]
        out[:, self.dim_range[j==r_n], j[j==r_n], :]= R_d2_desc_k[0,:,:][np.ix_(self.dim_range[j==r_n],np.arange(3))]
        out[:, self.dim_range[j==r_n], j[j==r_n], r_i] = R_d2_desc_1[:,self.dim_range[j==r_n],r_i]
        
        # 
        for val in self.dim_range[j==r_n]:
            
            val_atom = i[val]
            
            out[:, val, val_atom, :] = R_d2_desc_k[0,:,:][val,np.arange(3)][None]
            out[:, val, val_atom, r_i] = R_d2_desc_1[0,:,:][val,r_i]
            
            
        #out[:, self.dim_range[j==r_n], j[j==r_n], :]= R_d2_desc_k[0,:,:][np.ix_(self.dim_range[j==r_n],np.arange(3))]
        #out[:, self.dim_range[j==r_n], j[j==r_n], r_i] = R_d2_desc_1[:,self.dim_range[j==r_n],r_i]
        
        

    
        out[:, self.dim_range[i==r_n], i[i==r_n], :] = -R_d2_desc_k[0,:,:][np.ix_(self.dim_range[i==r_n],np.arange(3))]
        out[:, self.dim_range[i==r_n], i[i==r_n], r_i] = -R_d2_desc_1[:,self.dim_range[i==r_n],r_i][None]
        
        for val in self.dim_range[i==r_n]:
            val_atom = j[val]
            out[:, val, val_atom, :] = -R_d2_desc_k[0,:,:][val,np.arange(3)][None]
            out[:, val, val_atom, r_i] = -R_d2_desc_1[0,:,:][val,r_i]
        
        print('done')
        return out.reshape(-1, self.dim, self.dim_i)  # 66 * 36

    def d2_desc_from_comp_1017(self, R_val, index_d, out=None):
        # R_d_desc 66*3
        # NOTE: out must be filled with zeros!
        # R_d2_desc_1 : 
    
        R = R_val.reshape(-1,3)
        n = 1
        
    
        # if out is None:
        #     # out : 1* 66* 12 * 3
        #     # out: 1* N(N-1)/2 * N * 3
        #     out = np.zeros((n, self.dim, self.n_atoms, 3))
        # else:
        #     out = out.reshape(n, self.dim, self.n_atoms, 3)
        out = np.zeros((self.n_atoms,self.n_atoms,self.n_atoms*3))
        out_index = index_d
        out_index_atom = out_index//3
        out_index_xyz = out_index%3 
        for l in range(self.n_atoms*3):
            index_atom = l // 3
            index_xyz = l % 3
            
            for i in range(1,self.n_atoms):
                for j in range(i):
                    d = np.sqrt(np.sum((R[i,:]-R[j,:])**2))
                    if (index_atom == i and out_index_atom==i):
                        if index_xyz==out_index_xyz:
                            out[i,j,l] = - (d**2 - 3*(R[i,index_xyz]-R[j,out_index_xyz] ))/(d**5)
                        else:
                            out[i,j,l] = (3*(R[i,index_xyz]-R[j,index_xyz])*(R[i,out_index_xyz]-R[j,out_index_xyz]))/(d**5)
                    elif (index_atom == i and out_index_atom==j):
                        if index_xyz==out_index_xyz:
                            out[i,j,l] = (d**2 - 3*(R[i,index_xyz]-R[j,out_index_xyz] ))/(d**5)
                        else:
                            out[i,j,l] = (-3*(R[i,index_xyz]-R[j,index_xyz])*(R[i,out_index_xyz]-R[j,out_index_xyz]))/(d**5)
                    
                    if (index_atom == j and out_index_atom==j):
                        if index_xyz==out_index_xyz:
                            out[i,j,l] = - (d**2 - 3*(R[i,index_xyz]-R[j,out_index_xyz] ))/(d**5)
                        else:
                            out[i,j,l] = (3*(R[i,index_xyz]-R[j,index_xyz])*(R[i,out_index_xyz]-R[j,out_index_xyz]))/(d**5)
                    elif (index_atom == j and out_index_atom==i):
                        if index_xyz==out_index_xyz:
                            out[i,j,l] = (d**2 - 3*(R[i,index_xyz]-R[j,out_index_xyz] ))/(d**5)
                        else:
                            out[i,j,l] = (-3*(R[i,index_xyz]-R[j,index_xyz])*(R[i,out_index_xyz]-R[j,out_index_xyz]))/(d**5)
                            
        i, j = self.tril_indices
        res = out[i,j,:].reshape(1,-1, self.n_atoms*3)
        return res

        
        
        
          # 66 * 36

    # deflate desc
    def d_desc_to_comp(self, R_d_desc):

        # Add singleton dimension for single inputs.
        if R_d_desc.ndim == 2:
            R_d_desc = R_d_desc[None, ...]

        n = R_d_desc.shape[0]
        n_atoms = int(R_d_desc.shape[2] / 3)

        R_d_desc = R_d_desc.reshape(n, -1, n_atoms, 3)

        ret = np.zeros((n, n_atoms, n_atoms, 3))
        ret[:, self.M, self.A, :] = R_d_desc[:, self.d_desc_mask.ravel(), self.A, :]

        # only take upper triangle
        i, j = self.tril_indices
        ret = ret[:, i, j, :]

        return ret

#task_tem=np.load('task_file.npy',allow_pickle='TRUE').item()
#desc = Desc(                12,
#                interact_cut_off=task_tem['interact_cut_off'],
#                max_processes=None,
#            )