#import time module
import time

#import cpu library
import numpy as np

#turn off warnings from importing GPU module
import warnings
warnings.filterwarnings("ignore")

#import GPU (pycuda and skcuda) library
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
import numpy as np
import scipy.linalg
import skcuda.linalg as linalg
import pycuda.driver as drv


# initialize linear algebra module in skcuda and shall be put into main program
# in order to acheive better performance
linalg.init()


# explicit array used for testing
# A = np.array([[9, 3, 1, 5], [3, 7, 5, 1], [1, 5, 9, 2], [5, 1, 2, 6]], dtype=np.float64)
# B = np.array([[1, 1, 1, 1], [1, 1, 1, 1]], dtype=np.float64)
# B = B.transpose()



# A.

#A = np.ones((4, 4))

num=100
n_kind=2

A = np.random.rand(num,num)
B = np.random.rand(num, n_kind)

A = np.asarray(A, dtype=np.float64)
B = np.asarray(B, dtype=np.float64)

A = np.dot(A, A.transpose())

cpu_result = np.zeros(B.shape)
gpu_result = np.zeros(B.shape)



start_time = time.time()

for i in range(B.shape[1]):

	c, low = scipy.linalg.cho_factor(A)
	x =  scipy.linalg.cho_solve((c, low), B[:,i])
	cpu_result[:,i] = x


# print(cpu_result)

print("---CPU %s seconds ---" % (time.time() - start_time))




start = drv.Event()
end = drv.Event()
start.record()




for i in range(B.shape[1]):
	# memory transfer from cpu to gpu
	# note memory transfer between cpu and gpu is expensive and shall be avoided/minimized
	# in order to achieve high peformance
	A_gpu = gpuarray.to_gpu(A)
	B_gpu = gpuarray.to_gpu(B[:,i])

	# solve via skcuda
	linalg.cho_solve(A_gpu, B_gpu)
	gpu_result[:,i] = B_gpu.get()




end.record()
end.synchronize()
secs = start.time_till(end)*1e-3

print("---GPU %s seconds ---" % secs)

if np.allclose(cpu_result, gpu_result):
	print("cpu and gpu result match")
else:
	print("Error!!!! cpu and gpu result match")







