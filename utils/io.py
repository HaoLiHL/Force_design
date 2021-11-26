#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 21:35:15 2021

@author: lihao
"""
import numpy as np
import hashlib

def dataset_md5(dataset):

    md5_hash = hashlib.md5()

    keys = ['z', 'R']
    if 'E' in dataset:
        keys.append('E')
    keys.append('F')

    # only include new extra keys in fingerprint for 'modern' dataset files
    # 'code_version' was included from 0.4.0.dev1
    # opt_keys = ['lattice', 'e_unit', 'E_min', 'E_max', 'E_mean', 'E_var', 'f_unit', 'F_min', 'F_max', 'F_mean', 'F_var']
    # for k in opt_keys:
    #    if k in dataset:
    #        keys.append(k)

    for k in keys:
        d = dataset[k]
        if type(d) is np.ndarray:
            d = d.ravel()
        md5_hash.update(hashlib.md5(d).digest())

    return md5_hash.hexdigest().encode('utf-8')

