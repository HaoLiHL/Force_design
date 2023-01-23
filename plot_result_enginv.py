#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 21:31:44 2023

@author: lihao
"""

import matplotlib.pyplot as plt
import numpy as np

# ** salic acid c is 0.001 for first loop , then 0.01, initial 396
real_E = [-13477.8941838182, -13474.7828093186, -13474.080263758, -13472.6040433706, -13470.4845820429, -13464.9271783512, -13447.3936958204, -13339.7930673714, -13339.7930673714, -13339.7930673714, -13339.7930673714, -13339.7930673714]
predict_E = [-13477.8941838182, -13476.1088891, -13474.219972562893, -13472.742014114268, -13470.70622408398, -13465.923292076002, -13452.694395879713, -13414.305617998656, -13339.793067760314, -13339.793067566028, -13339.793067500344, -13339.793067468705]

fig, (ax1, ax2) = plt.subplots(1, 2)
#plt.plot(np.abs(E_predict_rec))
ax1.axhline(y=13477.894183, color='b', linestyle='--', label = 'minimum value from training')
ax1.axhline(y=13479.138923046, color='b', linestyle='--', label = 'maximum value from training')
ax1.plot(np.arange(len(real_E)),-np.array(real_E).reshape(-1), color = 'r',marker = 'o',label="real")
ax1.plot(np.arange(len(predict_E)),-np.array(predict_E).reshape(-1), color = 'b',marker = '.',label="predict")
#ax.axhline(y=13000, color='r', linestyle='--', label = 'target')

ax1.set_xlabel('number of evaluation')
ax1.set_ylabel('real designed energy from simulator')
#ax1.legend()
#ax.plot(np.arange(len(E_var_rec1)),np.array(E_var_rec1).reshape(-1), color = 'b')
ax1.set_title('c = 0.001, initial 398')
#plt.show()


# ** salic acid c is 0.001 for first loop , then 0.001, initial 396, target 13400
real_E = [-13477.9150077829, -13477.40149292, -13475.1656999049, -13472.2860361907, -13464.2453360536, -13454.4231769531, -13438.6667866173, -13418.731348135, -13402.1459067965, -13400.0275424556, -13400.0202417105, -13400.0156116581]

predict_E = [-13477.9150077829, -13477.593321196446, -13475.27717775623, -13473.12773090812, -13465.713433948122, -13457.206290390144, -13443.314896332407, -13424.51504275465, -13405.205166181144, -13400.138803031752, -13400.020201560219, -13400.015571257036]

#fig, ax = plt.subplots((1))

#plt.plot(np.abs(E_predict_rec))
ax2.axhline(y=13477.894183, color='b', linestyle='--', label = 'minimum value from training')
ax2.axhline(y=13479.138923046, color='b', linestyle='--', label = 'maximum value from training')
ax2.axhline(y=13400, color='g', linestyle='--', label = 'target')
ax2.plot(np.arange(len(real_E)),-np.array(real_E).reshape(-1), color = 'r',marker = 'o',label="real")
ax2.plot(np.arange(len(predict_E)),-np.array(predict_E).reshape(-1), color = 'b',marker = '.',label="predict")
#ax.axhline(y=13000, color='r', linestyle='--', label = 'target')


ax2.set_xlabel('number of evaluation')
ax2.set_ylabel('real designed energy from simulator')
#ax2.legend()
#ax.plot(np.arange(len(E_var_rec1)),np.array(E_var_rec1).reshape(-1), color = 'b')
ax2.set_title('c = 0.001, initial 396,target 13400')

handles, labels = ax2.get_legend_handles_labels()
lgd=fig.legend(handles, labels=['minimum value from training','maximum value from training','real','predict','target'], loc='upper right', bbox_to_anchor=(1.4,0.8))

fig.tight_layout()
fig.savefig('/Users/HL/Desktop/Study/SFM/inverse_force/sali_env.png', format='png',bbox_inches = 'tight',dpi=600,pad_inches = 0.2)
plt.show()



# ********* ASPIRIN *************
# initial 200, c 0.1. lr 1e-3, target 16000
real_E = [-17625.9965450208, -17610.4848744652, -17609.7434532482, -17607.5510046082, -17606.2785063043, -17604.1917730663, -17602.0873825956, -17599.9869535878, -17599.9737239303, -17599.9753960708, -17599.9792668761, -17599.982971877]

predict_E =  [-17625.9965450208, -17623.59069405102, -17610.358107138418, -17608.426863489494, -17606.78116741483, -17605.012027728837, -17602.8791744656, -17600.737125382755, -17600.01816835138, -17599.997857594262, -17599.993806974202, -17599.993411138203]
fig, (ax1, ax2) = plt.subplots(1, 2)

#plt.plot(np.abs(E_predict_rec))
ax2.axhline(y=17625.34, color='b', linestyle='--', label = 'minimum value from training')
ax2.axhline(y=17626.32, color='b', linestyle='--', label = 'maximum value from training')
ax2.axhline(y=17600, color='g', linestyle='--', label = 'target')
ax2.plot(np.arange(len(real_E)),-np.array(real_E).reshape(-1), color = 'r',marker = 'o',label="real")
ax2.plot(np.arange(len(predict_E)),-np.array(predict_E).reshape(-1), color = 'b',marker = '.',label="predict")
#ax.axhline(y=13000, color='r', linestyle='--', label = 'target')

ax2.set_xlabel('number of evaluation')
ax2.set_ylabel('real designed energy from simulator')
#ax1.legend()
#ax.plot(np.arange(len(E_var_rec1)),np.array(E_var_rec1).reshape(-1), color = 'b')
ax2.set_title('c = 0.1, initial 200,target 17600')






real_E = [-17625.9829083243, -17613.9280460495, -17611.0081987303, -17605.4401281371, -17602.0251286731, -17592.7095956627, -17583.6958847173, -17557.1875303881, -17536.5105243268, -17532.0318975945, -17522.6478510409, -17513.8290114128, -17513.8290114128, -17513.8290114128, -17513.8290114128, -17513.8290114128]

predict_E =  [-17625.9829083243, -17623.58410421844, -17613.48555063823, -17607.471282906245, -17603.444818735767, -17597.441760664202, -17589.27238332936, -17578.732831650897, -17553.56392951537, -17533.786097598084, -17525.402096568152, -17516.33582482469, -17514.102626233493, -17513.982786579138, -17513.935949234667, -17513.910982279314]



#plt.plot(np.abs(E_predict_rec))
ax1.axhline(y=17625.34, color='b', linestyle='--', label = 'minimum value from training')
ax1.axhline(y=17626.32, color='b', linestyle='--', label = 'maximum value from training')
ax1.plot(np.arange(len(real_E)),-np.array(real_E).reshape(-1), color = 'r',marker = 'o',label="real")
ax1.plot(np.arange(len(predict_E)),-np.array(predict_E).reshape(-1), color = 'b',marker = '.',label="predict")
#ax.axhline(y=13000, color='r', linestyle='--', label = 'target')

#ax.axhline(y=17600, color='g', linestyle='--', label = 'target')
ax1.set_xlabel('number of evaluation')
ax1.set_ylabel('real designed energy from simulator')
ax1.set_title('c = 0.1, initial 200')


handles, labels = ax2.get_legend_handles_labels()
lgd=fig.legend(handles, labels=['minimum value from training','maximum value from training','real','predict','target'], loc='upper right', bbox_to_anchor=(1.4,0.8))

fig.tight_layout()
fig.savefig('/Users/HL/Desktop/Study/SFM/inverse_force/asp_env.png', format='png',bbox_inches = 'tight',dpi=600,pad_inches = 0.2)
plt.show()



