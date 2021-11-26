#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 14:53:37 2021

@author: lihao
"""
import sys
import timeit
import numpy as np
import scipy.optimize
import scipy.spatial.distance
import multiprocessing as mp
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from functools import partial

from utils.desc import Desc

DONE = 1
NOT_DONE = 0
Pool = mp.get_context('fork').Pool
glob={}

def share_array(arr_np, typecode):
    arr = mp.RawArray(typecode, arr_np.ravel())
    return arr, arr_np.shape


def _bipartite_match_wkr(i, n_train, same_z_cost):

    global glob

    adj_set = np.frombuffer(glob['adj_set']).reshape(glob['adj_set_shape'])
    v_set = np.frombuffer(glob['v_set']).reshape(glob['v_set_shape'])
    match_cost = np.frombuffer(glob['match_cost']).reshape(glob['match_cost_shape'])

    adj_i = scipy.spatial.distance.squareform(adj_set[i, :])
    v_i = v_set[i, :, :]

    match_perms = {}
    for j in range(i + 1, n_train):

        adj_j = scipy.spatial.distance.squareform(adj_set[j, :])
        v_j = v_set[j, :, :]

        cost = -np.fabs(v_i).dot(np.fabs(v_j).T)
        cost += same_z_cost * np.max(np.abs(cost))

        _, perm = scipy.optimize.linear_sum_assignment(cost)

        adj_i_perm = adj_i[:, perm]
        adj_i_perm = adj_i_perm[perm, :]

        score_before = np.linalg.norm(adj_i - adj_j)
        score = np.linalg.norm(adj_i_perm - adj_j)

        match_cost[i, j] = score
        if score >= score_before:
            match_cost[i, j] = score_before
        elif not np.isclose(score_before, score):  # otherwise perm is identity
            match_perms[i, j] = perm
        
    return match_perms


def bipartite_match(R, z, lat_and_inv=None, max_processes=None, callback=None):

    global glob

    n_train, n_atoms, _ = R.shape

    # penalty matrix for mixing atom species
    same_z_cost = np.repeat(z[:, None], len(z), axis=1) - z
    same_z_cost[same_z_cost != 0] = 1



    match_cost = np.zeros((n_train, n_train))

    desc = Desc(n_atoms, max_processes=max_processes)

    adj_set = np.empty((n_train, desc.dim))
    v_set = np.empty((n_train, n_atoms, n_atoms))
    for i in range(n_train):
        r = np.squeeze(R[i, :, :])

        if lat_and_inv is None:
            adj = scipy.spatial.distance.pdist(r, 'euclidean')


            # from ase import Atoms
            # from ase.geometry.analysis import Analysis

            # atoms = Atoms(
            #     z, positions=r
            # )  # only use first molecule in dataset to find connected components (fix me later, maybe) # *0.529177249

            # bonds = Analysis(atoms).all_bonds[0]
            
            #adj = scipy.spatial.distance.squareform(adj)

            #bonded = np.zeros((z.size, z.size))

            #for j, bonded_to in enumerate(bonds):
                #inv_bonded_to = np.arange(n_atoms)
                #inv_bonded_to[bonded_to] = 0

                #adj[j, inv_bonded_to] = 0

            #    bonded[j, bonded_to] = 1

            # bonded = bonded + bonded.T

            # print(bonded)

        else:
            adj = scipy.spatial.distance.pdist(
                r, lambda u, v: np.linalg.norm(desc.pbc_diff(u - v, lat_and_inv))
            )

        w, v = np.linalg.eig(scipy.spatial.distance.squareform(adj))
        v = v[:, w.argsort()[::-1]]

        adj_set[i, :] = adj
        v_set[i, :, :] = v

    glob['adj_set'], glob['adj_set_shape'] = share_array(adj_set, 'd')
    glob['v_set'], glob['v_set_shape'] = share_array(v_set, 'd')
    glob['match_cost'], glob['match_cost_shape'] = share_array(match_cost, 'd')

    if callback is not None:
        callback = partial(callback, disp_str='Bi-partite matching')

    start = timeit.default_timer()
    pool = Pool(max_processes)

    match_perms_all = {}
    for i, match_perms in enumerate(
        pool.imap_unordered(
            partial(_bipartite_match_wkr, n_train=n_train, same_z_cost=same_z_cost),
            list(range(n_train)),
        )
    ):
        match_perms_all.update(match_perms)

        if callback is not None:
            callback(i, n_train)

    pool.close()
    pool.join()  # Wait for the worker processes to terminate (to measure total runtime correctly).
    stop = timeit.default_timer()

    dur_s = (stop - start) / 2
    sec_disp_str = 'took {:.1f} s'.format(dur_s) if dur_s >= 0.1 else ''
    if callback is not None:
        callback(n_train, n_train, sec_disp_str=sec_disp_str)

    match_cost = np.frombuffer(glob['match_cost']).reshape(glob['match_cost_shape'])
    match_cost = match_cost + match_cost.T
    match_cost[np.diag_indices_from(match_cost)] = np.inf
    match_cost = csr_matrix(match_cost)

    return match_perms_all, match_cost


def sync_perm_mat(match_perms_all, match_cost, n_atoms, callback=None):

    if callback is not None:
        callback = partial(
            callback, disp_str='Multi-partite matching (permutation synchronization)'
        )
        callback(NOT_DONE)

    tree = minimum_spanning_tree(match_cost, overwrite=True)

    perms = np.arange(n_atoms, dtype=int)[None, :]
    rows, cols = tree.nonzero()
    for com in zip(rows, cols):
        perm = match_perms_all.get(com)
        if perm is not None:
            perms = np.vstack((perms, perm))
    perms = np.unique(perms, axis=0)

    if callback is not None:
        callback(DONE)

    return perms

def to_cycles(perm):
    pi = {i: perm[i] for i in range(len(perm))}
    cycles = []

    while pi:
        elem0 = next(iter(pi)) # arbitrary starting element
        this_elem = pi[elem0]
        next_item = pi[this_elem]

        cycle = []
        while True:
            cycle.append(this_elem)
            del pi[this_elem]
            this_elem = next_item
            if next_item in pi:
                next_item = pi[next_item]
            else:
                break

        cycles.append(cycle)

    return cycles

def salvage_subgroup(perms):

    n_perms, n_atoms = perms.shape
    lcms = []
    for i in range(n_perms):
        cy_lens = [len(cy) for cy in to_cycles(list(perms[i, :]))]
        lcm = np.lcm.reduce(cy_lens)
        lcms.append(lcm)
    keep_idx = np.argmax(lcms)
    perms = np.vstack((np.arange(n_atoms), perms[keep_idx,:]))

    return perms

def complete_sym_group(perms, n_perms_max=None, disp_str='Permutation group completion', callback=None):

    if callback is not None:
        callback = partial(callback, disp_str=disp_str)
        callback(NOT_DONE)

    perm_added = True
    while perm_added:
        perm_added = False
        n_perms = perms.shape[0]
        for i in range(n_perms):
            for j in range(n_perms):

                new_perm = perms[i, perms[j, :]]
                if not (new_perm == perms).all(axis=1).any():
                    perm_added = True
                    perms = np.vstack((perms, new_perm))

                    # Transitive closure is not converging! Give up and return identity permutation.
                    if n_perms_max is not None and perms.shape[0] == n_perms_max:

                        if callback is not None:
                            callback(
                                DONE,
                                sec_disp_str='transitive closure has failed',
                                done_with_warning=True,
                            )
                        return None

    if callback is not None:
        callback(
            DONE,
            sec_disp_str='found {:d} symmetries'.format(perms.shape[0]),
        )

    return perms



def find_perms(R, z, lat_and_inv=None, callback=None, max_processes=None):

    m, n_atoms = R.shape[:2]

    # Find matching for all pairs.
    match_perms_all, match_cost = bipartite_match(
        R, z, lat_and_inv, max_processes, callback=callback
    )

    # Remove inconsistencies.
    match_perms = sync_perm_mat(match_perms_all, match_cost, n_atoms, callback=callback)

    # Commplete symmetric group.
    # Give up, if transitive closure yields more than 100 unique permutations.
    sym_group_perms = complete_sym_group(match_perms, n_perms_max=100, callback=callback)

    # Limit closure to largest cardinality permutation in the set to get at least some symmetries.
    if sym_group_perms is None:
        match_perms_subset = salvage_subgroup(match_perms)
        sym_group_perms = complete_sym_group(match_perms_subset, n_perms_max=100, disp_str='Closure disaster recovery', callback=callback)

    return sym_group_perms