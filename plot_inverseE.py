#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 21:53:54 2022

@author: lihao
"""


import matplotlib.pyplot as plt
import numpy as np
real_E = [-259924.0986058375, -259883.7011980016, -259883.7011980016, -259883.7011980016, -259862.73884425985, -259822.44996232697, -259778.38783015858, -259723.13850497946, -259672.12690350783, -259617.44307099527, -259562.66742504042, -259508.23826693365, -258686.7896734143, -259095.0887622307, -259070.5729461964, -259070.5729461964, -259070.5729461964, -259070.5729461964, -259070.5729461964, -259070.5729461964, -259070.5729461964, -259070.5729461964, -259070.5729461964, -259070.5729461964, -259070.5729461964, -259070.5729461964, -259070.5729461964, -259070.5729461964, -259070.5729461964, -259070.5729461964, -259070.5729461964, -259070.5729461964]
fig, ax = plt.subplots((1))

#plt.plot(np.abs(E_predict_rec))
ax.plot(np.arange(len(real_E)),-np.array(real_E).reshape(-1), color = 'r',marker = 'o')
ax.axhline(y=259000, color='r', linestyle='--', label = 'target')
ax.axhline(y=259912, color='b', linestyle='--', label = 'minimum value from training')
ax.axhline(y=259932, color='b', linestyle='--', label = 'maximum value from training')
ax.set_xlabel('number of evaluation')
ax.set_ylabel('real designed energy from simulator')
ax.legend()
#ax.plot(np.arange(len(E_var_rec1)),np.array(E_var_rec1).reshape(-1), color = 'b')
ax.set_title('c = 0.1')


plt.show()



