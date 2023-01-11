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


real_E = [-259924.0986058375, -259673.10532618748, -259673.10532618748, -259610.02897153745, -259490.04987957043, -259351.3120945087, -259193.62805674123, -259013.16747872464, -258790.9494916843, -258556.96543889656, -258339.34988873932, -258141.53788560213]
fig, ax = plt.subplots((1))

#plt.plot(np.abs(E_predict_rec))
ax.plot(np.arange(len(real_E)),-np.array(real_E).reshape(-1), color = 'r',marker = 'o')
ax.axhline(y=257000, color='r', linestyle='--', label = 'target')
ax.axhline(y=259912, color='b', linestyle='--', label = 'minimum value from training')
ax.axhline(y=259932, color='b', linestyle='--', label = 'maximum value from training')
ax.set_xlabel('number of evaluation')
ax.set_ylabel('real designed energy from simulator')
ax.legend()
#ax.plot(np.arange(len(E_var_rec1)),np.array(E_var_rec1).reshape(-1), color = 'b')
ax.set_title('c = 0.5')


plt.show()

# c is 0.5
real_E = [-259924.0986058375, -259673.10532618748, -259673.10532618748, -259610.02897153745, -259490.04987957043, -259351.3120945087, -259193.62805674123, -259013.16747872464, -258790.9494916843, -258556.96543889656, -258339.34988873932, -258141.53788560213, -257965.93078251986, -256745.94329148944, -256981.97870112295, -257000.8886468779, -257000.8886468779, -257000.04258718918, -257000.00684731556, -257000.00684731556, -257000.00684731556, -256999.9944423905, -256999.9944423905, -256999.9944423905, -256999.9944423905, -256999.9944423905, -256999.9944423905, -256999.9944423905, -256999.9944423905, -256999.9944423905, -256999.9944423905, -256999.9944423905]
predict_E = [-259924.0986058375, -259673.10532618748, -259673.10532618748, -259610.02897153745, -259490.04987957043, -259351.3120945087, -259193.62805674123, -259013.16747872464, -258790.9494916843, -258556.96543889656, -258339.34988873932, -258141.53788560213, -257965.93078251986, -256745.94329148944, -256981.97870112295, -257000.8886468779, -257000.8886468779, -257000.04258718918, -257000.00684731556, -257000.00684731556, -257000.00684731556, -256999.9944423905, -256999.9944423905, -256999.9944423905, -256999.9944423905, -256999.9944423905, -256999.9944423905, -256999.9944423905, -256999.9944423905, -256999.9944423905, -256999.9944423905, -256999.9944423905]

fig, ax = plt.subplots((1))

#plt.plot(np.abs(E_predict_rec))
ax.plot(np.arange(len(real_E)),-np.array(real_E).reshape(-1), color = 'r',marker = 'o',label="real")
ax.plot(np.arange(len(predict_E)),-np.array(predict_E).reshape(-1), color = 'b',marker = '.',label="predict")
ax.axhline(y=257000, color='r', linestyle='--', label = 'target')
ax.axhline(y=259912, color='b', linestyle='--', label = 'minimum value from training')
ax.axhline(y=259932, color='b', linestyle='--', label = 'maximum value from training')
ax.set_xlabel('number of evaluation')
ax.set_ylabel('real designed energy from simulator')
ax.legend()
#ax.plot(np.arange(len(E_var_rec1)),np.array(E_var_rec1).reshape(-1), color = 'b')
ax.set_title('c = 0.5')


plt.show()


# c is 0.1
real_E = [-259924.0986058375, -259673.10532618748, -259582.70444649033, -259436.19206201853, -259274.99595757527, -259104.82361092977, -258907.70894205122, -258581.0211127221, -258224.01761145223, -258020.6016746554, -257848.0571142551, -257508.92552490556, -256957.6856200097, -257022.6957011246, -256993.23491467015, -256994.91249640752, -256996.06765673633, -257003.56172370128, -257003.56172370128, -256999.41156513547, -256999.41156513547, -256999.41156513547, -256999.41156513547, -256999.41156513547, -257000.10022194043, -257000.10022194043, -257000.10022194043, -257000.10022194043, -257000.10022194043, -257000.10022194043, -257000.10022194043, -257000.10022194043]

predict_E = [-259924.0986058375, -259817.89096669,-259603.75082688,-259445.35540964,-259281.22940673,-259110.26655572,-258913.51229073
 ,-258590.00451996,-258232.50960648,-258023.03873018,-257848.95824599,-257511.5433004,-256964.43820131
 ,-257022.55741833,-256993.21231361,-256994.9056672,-256996.06390278,-257003.56308055,-257003.56276407,
 -256999.39706051,-256999.40550078,-256999.40773157,-256999.40876254,-256999.40935648,-257000.09925852,-257000.09934239,-257000.09941282,-257000.09947277,
 -257000.09952449,-257000.09956956,-257000.09960915,-257000.09964419]
fig, ax = plt.subplots((1))

#plt.plot(np.abs(E_predict_rec))
ax.plot(np.arange(len(real_E)),-np.array(real_E).reshape(-1), color = 'r',marker = 'o',label="real")
ax.plot(np.arange(len(predict_E)),-np.array(predict_E).reshape(-1), color = 'b',marker = '.',label="predict")
ax.axhline(y=257000, color='r', linestyle='--', label = 'target')
ax.axhline(y=259912, color='b', linestyle='--', label = 'minimum value from training')
ax.axhline(y=259932, color='b', linestyle='--', label = 'maximum value from training')
ax.set_xlabel('number of evaluation')
ax.set_ylabel('real designed energy from simulator')
ax.legend()
#ax.plot(np.arange(len(E_var_rec1)),np.array(E_var_rec1).reshape(-1), color = 'b')
ax.set_title('c = 0.1')


plt.show()



# salic acid c is 0.01 for first loop , then 0.001 
real_E = [-13477.8941838182, -13474.7828093186, -13474.080263758, -13472.6040433706, -13470.4845820429, -13464.9271783512, -13447.3936958204, -13339.7930673714, -13339.7930673714, -13339.7930673714, -13339.7930673714, -13339.7930673714, -13339.7930673714, -13339.7930673714, -13339.7930673714, -13339.7930673714, -13339.7930673714, -13339.7930673714, -13339.7930673714, -13339.7930673714, -13339.7930673714, -13339.7930673714, -13339.7930673714, -13339.7930673714, -13339.7930673714]
predict_E = [-13477.8941838182, -13476.1088891, -13474.219972562893, -13472.742014114268, -13470.70622408398, -13465.923292076002, -13452.694395879713, -13414.305617998656, -13339.793067760314, -13339.793067566028, -13339.793067500344, -13339.793067468705, -13339.793067448296, -13339.793067436189, -13339.793067427094, -13339.793067420193, -13339.793067414608, -13339.793067410887, -13339.793067407143, -13339.79306740391, -13339.793067401137, -13339.793067399944, -13339.79306739745, -13339.793067395585, -13339.793067394749, -13106.227336846056]
fig, ax = plt.subplots((1))

#plt.plot(np.abs(E_predict_rec))
ax.plot(np.arange(len(real_E)),-np.array(real_E).reshape(-1), color = 'r',marker = 'o',label="real")
ax.plot(np.arange(len(predict_E)),-np.array(predict_E).reshape(-1), color = 'b',marker = '.',label="predict")
ax.axhline(y=13000, color='r', linestyle='--', label = 'target')
ax.axhline(y=13477.894183, color='b', linestyle='--', label = 'minimum value from training')
ax.axhline(y=13479.138923046, color='b', linestyle='--', label = 'maximum value from training')
ax.set_xlabel('number of evaluation')
ax.set_ylabel('real designed energy from simulator')
ax.legend()
#ax.plot(np.arange(len(E_var_rec1)),np.array(E_var_rec1).reshape(-1), color = 'b')
ax.set_title('c = 0.001')


plt.show()


# ** salic acid c is 0.001 for first loop , then 0.01, initial 398 
real_E = [-13477.8941838182, -13474.7828093186, -13474.080263758, -13472.6040433706, -13470.4845820429, -13464.9271783512, -13447.3936958204, -13339.7930673714, -13339.7930673714, -13339.7930673714, -13339.7930673714, -13339.7930673714]
predict_E = [-13477.8941838182, -13476.1088891, -13474.219972562893, -13472.742014114268, -13470.70622408398, -13465.923292076002, -13452.694395879713, -13414.305617998656, -13339.793067760314, -13339.793067566028, -13339.793067500344, -13339.793067468705]
fig, ax = plt.subplots((1))

#plt.plot(np.abs(E_predict_rec))
ax.plot(np.arange(len(real_E)),-np.array(real_E).reshape(-1), color = 'r',marker = 'o',label="real")
ax.plot(np.arange(len(predict_E)),-np.array(predict_E).reshape(-1), color = 'b',marker = '.',label="predict")
#ax.axhline(y=13000, color='r', linestyle='--', label = 'target')
ax.axhline(y=13477.894183, color='b', linestyle='--', label = 'minimum value from training')
ax.axhline(y=13479.138923046, color='b', linestyle='--', label = 'maximum value from training')
ax.set_xlabel('number of evaluation')
ax.set_ylabel('real designed energy from simulator')
ax.legend()
#ax.plot(np.arange(len(E_var_rec1)),np.array(E_var_rec1).reshape(-1), color = 'b')
ax.set_title('c = 0.001')


plt.show()

# ** salic acid c is 0.001 for first loop , then 0.01, initial 396
real_E = [-13477.9150077829, -13473.3463547704, -13472.0881554096, -13469.8479326399, -13460.5665687369, -13460.5665687369, -13417.8229684725, -13417.8229684725, -13417.8229684725, -13417.8229684725, -13417.8229684725, -13417.8229684725, -13417.8229684725, -13417.8229684725, -13417.8229684725, -13417.8229684725, -13417.8229684725, -13417.8229684725, -13417.8229684725, -13417.8229684725, -13417.8229684725, -13417.8229684725]

predict_E = [-13477.9150077829, -13475.40775768, -13472.44387962135, -13470.096654852248, -13463.114037068703, -13460.566568818924, -13439.10629877781, -13417.822969102514, -13417.822968787337, -13417.822968682029, -13417.82296863005, -13417.822968598097, -13417.822968577231, -13417.822968562246, -13417.822968551136, -13417.82296854232, -13417.822968535407, -13417.822968529634, -13417.822968523966, -13417.822968521932, -13417.822968517925, -13417.822968513725]
fig, ax = plt.subplots((1))

#plt.plot(np.abs(E_predict_rec))
ax.plot(np.arange(len(real_E)),-np.array(real_E).reshape(-1), color = 'r',marker = 'o',label="real")
ax.plot(np.arange(len(predict_E)),-np.array(predict_E).reshape(-1), color = 'b',marker = '.',label="predict")
#ax.axhline(y=13000, color='r', linestyle='--', label = 'target')
ax.axhline(y=13477.894183, color='b', linestyle='--', label = 'minimum value from training')
ax.axhline(y=13479.138923046, color='b', linestyle='--', label = 'maximum value from training')
ax.set_xlabel('number of evaluation')
ax.set_ylabel('real designed energy from simulator')
ax.legend()
#ax.plot(np.arange(len(E_var_rec1)),np.array(E_var_rec1).reshape(-1), color = 'b')
ax.set_title('c = 0.01, initial 396')


plt.show()


# ** salic acid c is 0.001 for first loop , then 0.001, initial 396, target 13400
real_E = [-13477.9150077829, -13477.40149292, -13475.1656999049, -13472.2860361907, -13464.2453360536, -13454.4231769531, -13438.6667866173, -13418.731348135, -13402.1459067965, -13400.0275424556, -13400.0202417105, -13400.0156116581]

predict_E = [-13477.9150077829, -13477.593321196446, -13475.27717775623, -13473.12773090812, -13465.713433948122, -13457.206290390144, -13443.314896332407, -13424.51504275465, -13405.205166181144, -13400.138803031752, -13400.020201560219, -13400.015571257036]

fig, ax = plt.subplots((1))

#plt.plot(np.abs(E_predict_rec))
ax.plot(np.arange(len(real_E)),-np.array(real_E).reshape(-1), color = 'r',marker = 'o',label="real")
ax.plot(np.arange(len(predict_E)),-np.array(predict_E).reshape(-1), color = 'b',marker = '.',label="predict")
#ax.axhline(y=13000, color='r', linestyle='--', label = 'target')
ax.axhline(y=13477.894183, color='b', linestyle='--', label = 'minimum value from training')
ax.axhline(y=13479.138923046, color='b', linestyle='--', label = 'maximum value from training')
ax.axhline(y=13400, color='g', linestyle='--', label = 'target')
ax.set_xlabel('number of evaluation')
ax.set_ylabel('real designed energy from simulator')
ax.legend()
#ax.plot(np.arange(len(E_var_rec1)),np.array(E_var_rec1).reshape(-1), color = 'b')
ax.set_title('c = 0.001, initial 396,target 13400')


plt.show()

# ********* ASPIRIN *************
# initial 190, c 0.1.
real_E = [-17625.9965450208, -17610.8561542301, -17609.319067228, -17604.9320795519, -17600.9733768842, -17593.3005921217, -17588.3842727443, -17587.1475970295, -17585.0124008921, -17582.8010285819, -17580.4960135394, -17578.079454171]

predict_E = [-17625.9965450208, -17623.602998537335, -17610.65469833008, -17607.618891051174, -17603.63043467942, -17599.81524863015, -17592.37224771194, -17587.636563818407, -17585.85298092687, -17583.561789902607, -17581.331214135054, -17578.969935840825]

fig, ax = plt.subplots((1))

#plt.plot(np.abs(E_predict_rec))
ax.plot(np.arange(len(real_E)),-np.array(real_E).reshape(-1), color = 'r',marker = 'o',label="real")
ax.plot(np.arange(len(predict_E)),-np.array(predict_E).reshape(-1), color = 'b',marker = '.',label="predict")
#ax.axhline(y=13000, color='r', linestyle='--', label = 'target')
ax.axhline(y=17625.34, color='b', linestyle='--', label = 'minimum value from training')
ax.axhline(y=17626.32, color='b', linestyle='--', label = 'maximum value from training')
#ax.axhline(y=13400, color='g', linestyle='--', label = 'target')
ax.set_xlabel('number of evaluation')
ax.set_ylabel('real designed energy from simulator')
ax.legend()
#ax.plot(np.arange(len(E_var_rec1)),np.array(E_var_rec1).reshape(-1), color = 'b')
ax.set_title('c = 0.1, initial 190,target 16000')


plt.show()

# ********* ASPIRIN *************
# initial 190, c 0.1.
real_E = [-17625.9965450208, -17610.4848744652, -17609.7434532482, -17607.5510046082, -17606.2785063043, -17604.1917730663, -17602.0873825956, -17599.9869535878, -17599.9737239303, -17599.9753960708, -17599.9792668761, -17599.982971877]

predict_E =  [-17625.9965450208, -17623.59069405102, -17610.358107138418, -17608.426863489494, -17606.78116741483, -17605.012027728837, -17602.8791744656, -17600.737125382755, -17600.01816835138, -17599.997857594262, -17599.993806974202, -17599.993411138203]
fig, ax = plt.subplots((1))

#plt.plot(np.abs(E_predict_rec))
ax.plot(np.arange(len(real_E)),-np.array(real_E).reshape(-1), color = 'r',marker = 'o',label="real")
ax.plot(np.arange(len(predict_E)),-np.array(predict_E).reshape(-1), color = 'b',marker = '.',label="predict")
#ax.axhline(y=13000, color='r', linestyle='--', label = 'target')
ax.axhline(y=17625.34, color='b', linestyle='--', label = 'minimum value from training')
ax.axhline(y=17626.32, color='b', linestyle='--', label = 'maximum value from training')
ax.axhline(y=17600, color='g', linestyle='--', label = 'target')
ax.set_xlabel('number of evaluation')
ax.set_ylabel('real designed energy from simulator')
ax.legend()
#ax.plot(np.arange(len(E_var_rec1)),np.array(E_var_rec1).reshape(-1), color = 'b')
ax.set_title('c = 0.1, initial 190,target 17600')


plt.show()








