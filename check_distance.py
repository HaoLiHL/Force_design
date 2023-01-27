#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 20:57:07 2023

@author: lihao
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 16:56:25 2023

@author: lihao
"""
import numpy as np
import scipy as sp
import multiprocessing as mp
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.colors as plt_color
#from solvers.analytic_u_pp import Analytic
from utils import perm
from utils.desc import Desc

#task = np.load('/Users/HL/Desktop/Study/SFM/proposed_R_salic.npy',allow_pickle=True).item()
task = np.load('/Users/HL/Desktop/Study/SFM/proposed_R_asp.npy',allow_pickle=True).item()
R = task['R_train']

R_test = np.array([2.1893727664,    0.6339981721,   -0.1655362427,
                 2.2867935017,    2.0053394372,   -0.0183626472  ,
                 1.1606808160,    2.7420580054,    0.3209182679,
                 -0.0561840390,    2.1079466147,    0.5077285244,
                 -0.1465765448,    0.7383371993,    0.3559978426,
         0.9727788099,   -0.0232568592,    0.0200145274,
       3.0511501862,    0.0554424198 ,  -0.4221000908,
       3.2297915980,    2.4928781759 ,  -0.1654694953,
       1.2275508612,    3.8059147508,    0.4382171445,
     -0.9370660007,    2.6549467358  ,  0.7730723219,
        -1.3723959273,    0.1503411010 ,   0.6303296086,
       -2.1909122610,   -0.3742506040 ,  -0.3424522747,
       -2.0111294426,   -0.1513964539  , -1.5125659184,
        -3.2752819291,   -1.1885971183 ,   0.2646249402,
       -2.8357287799,   -2.0953144276,    0.6590821842,
       -4.0098866531,   -1.4354223073,   -0.4843391571,
        -3.7366576316,   -0.6531109719,    1.0822540843,
         0.8963405566,   -1.4857017018,   -0.1309494436,
         2.0951516266,   -2.0545678390,   -0.3901732098,
       -0.1016731800,   -2.1745531342,   -0.0405607386,
       2.0430682970 ,  -3.0013832069 ,  -0.4950806971
    ]).reshape(1,21,3)

R_test =  np.array([-6.4300901476,	7.2331696223,	-0.5911664542,
-5.9715435892,	8.555565608,	-0.5610225352,
-4.6508442198,	8.8485158053,	-0.950795729,
-3.7925281465,	7.8139228357,	-1.3511948229,
-4.2511352389,	6.4893931244,	-1.3688682882,
-5.5850764413,	6.1726550283,	-1.0009542423,
-7.4542929189,	6.9931926667,	-0.3021570067,
-6.6426765618,	9.3554913681,	-0.2387048525,
-4.2850410444,	9.8785082166,	-0.9343369018,
-2.7615561704,	8.009896485,	-1.6522815598,
-3.3435317115,	5.5126558623,	-1.8315551262,
-2.9153750375,	4.4799079277,	-0.9354378578,
-3.1093061487,	4.5366674165,	0.2811105678,
-2.2301457462,	3.4023997094,	-1.7159698343,
-3.0114113718,	2.7910314003,	-2.199932976,
-1.6417765253,	2.7704998751,	-1.0376995306,
-1.5924774877,	3.825450286,	-2.5063696617,
-6.109701237,	4.7939567118,	-1.0790541151,
-7.4412451902,	4.7110920658,	-0.6676714226,
-5.5049187571,	3.7717220042,	-1.4729343509,
-7.7189163082,	3.7623559803,	-0.7493333001]).reshape(1,21,3)


R_new = np.append(R,R_test).reshape(R.shape[0]+1,21,-1)

desc = Desc(
        R_new.shape[1],
        interact_cut_off=task['interact_cut_off'],
        max_processes=None,
    )

R_desc_atom, R_d_desc_atom = desc.from_R(R_new,lat_and_inv=None,
                callback=None)

n_molecule = R_desc_atom.shape[0]


distance = np.zeros([n_molecule,n_molecule])
for i in range(n_molecule):
    for j in range(i):
        distance[i,j] = np.linalg.norm(R_desc_atom[i]-R_desc_atom[j])
        distance[j,i] =distance[i,j]
        


fig = plt.figure() #调用figure创建一个绘图对象
#(ax1, ax2) = plt.subplots(1, 2)
#(ax1, ax2) = fig.add_subplot(211)
ax = fig.add_subplot(111)
cax = ax.matshow(distance, cmap='bwr')  #绘制热力图，从-1到1
#ax.axis('off')
#cax = ax.matshow(correlations, cmap='Blues',vmin=np.min(k2), vmax=np.max(k2))  #绘制热力图，从-1到1
fig.colorbar(cax)  #将matshow生成热力图设置为颜色渐变条
#ticks = np.arange(0,9,1) #生成0-9，步长为1
#ax.set_xticks(ticks)  #生成刻度
#ax.set_yticks(ticks)
ax.set_title("Aspirin euclidean: distance between training and proposed")
#ax.set_xticklabels(names) #生成x轴标签
#ax.set_yticklabels(names)
#plt.imshow(I, cmap='gray');


fig.tight_layout()
#fig.savefig('/Users/HL/Desktop/Study/SFM/inverse_force/euclidean_distance_Salicylic.png', format='png',bbox_inches = 'tight',dpi=600,pad_inches = 0.2)
plt.show()