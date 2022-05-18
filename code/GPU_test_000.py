from numba import cuda
import numpy as np
import time
np.random.seed(1)
'''
a = np.random.randint(15, (100, 100))
b = np.random.randint(15, (100, 100))
c = np.zeros((100, 100))
'''
n = 200000000
a = np.random.randint(15, size=n)
b = np.random.randint(15, size=n)
c = np.zeros(n)
start_time = time.time()
for i in range(n):
    c[i] = a[i] + b[i]
print("--- %s seconds ---" % (time.time() - start_time))
print(c)

@cuda.jit('void(float32[:], float32[:], float32[:])')
def cuda_addition(a,b,c):
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if i > c.size:
        return
    c[i] = a[i] + b[i]


device = cuda.get_current_device()
start_time = time.time()
d_a = cuda.to_device(a)
d_b = cuda.to_device(b)
d_c = cuda.device_array_like(a)

tpb = device.WARP_SIZE
bpg = int(np.ceil(n/tpb))

cuda_addition[bpg, tpb](d_a, d_b, d_c)

c = d_c.copy_to_host()
print("--- %s seconds ---" % (time.time() - start_time))
print(c)