from numba import cuda, float32, int32
import numpy as np
import time
import math

np.random.seed(1)
'''
a = np.random.randint(15, (100, 100))
b = np.random.randint(15, (100, 100))
c = np.zeros((100, 100))
'''

'''
a = np.array([[1., 2.],
              [3., 4.]]).astype("float32")

b = np.array([[4., 3.],
              [2., 1.]]).astype("float32")
'''
'''
a = np.array([[1, 2, 3]]).astype("float32")

b = np.array([[1],
              [2],
              [3]]).astype("float32")
'''
'''
a = np.array([[1],
              [2],
              [3]]).astype("float32")

b = np.array([[1, 2, 3]]).astype("float32")
n = a.shape[1]
m = a.shape[0]
k = b.shape[1]

'''
n = 10240
m = 1024
k = 128
a = np.random.random(size=(m, n)).astype("float32")
b = np.random.random(size=(n, k)).astype("float32")
#c = np.zeros((1, 1)).astype("float32")
start_time = time.time()


c = a.dot(b)

print("--- %s seconds ---" % (time.time() - start_time))
print(c)
print(c.dtype)

dtype = a.dtype

c = np.zeros((m, k)).astype("float32")

'''
@cuda.jit('void(float32[:,:], float32[:,:], float32[:,:])')
def cuda_matmul(A, B, C):
    t = 14
    #TPB = int32(tr_per_block)
    sA = cuda.shared.array(shape=(t, t), dtype=float32)
    sB = cuda.shared.array(shape=(t, t), dtype=float32)

    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x

    tmp = float32(0.)

    for i in range(bpg):
        sA[tx, ty] = 0
        sB[tx, ty] = 0
        if x < A.shape[0] and (ty + i * t) < A.shape[1]:
            sA[tx, ty] = A[x, ty + i * t]
        if y < B.shape[1] and (tx + i * t) < B.shape[0]:
            sB[tx, ty] = B[tx + i * t, y]

        cuda.syncthreads()

        for j in range(t):
            tmp += sA[tx, j] * sB[j, ty]

        cuda.syncthreads()

    if x < C.shape[0] and y < C.shape[1]:
        C[x, y] = tmp
'''
@cuda.jit('void(float32[:,:], float32[:,:], float32[:,:])')
def cuda_matmul(A, B, C):
    t = 14
    #TPB = int32(tr_per_block)
    sA = cuda.shared.array(shape=(t, t), dtype=float32)
    sB = cuda.shared.array(shape=(t, t), dtype=float32)

    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x

    tmp = float32(0.)

    for i in range(bpg):
        sA[tx, ty] = 0
        sB[tx, ty] = 0
        if x < A.shape[0] and (ty + i * t) < A.shape[1]:
            sA[tx, ty] = A[x, ty + i * t]
        if y < B.shape[1] and (tx + i * t) < B.shape[0]:
            sB[tx, ty] = B[tx + i * t, y]

        cuda.syncthreads()

        for j in range(t):
            tmp += sA[tx, j] * sB[j, ty]

        cuda.syncthreads()

    if x < C.shape[0] and y < C.shape[1]:
        C[x, y] = tmp
device = cuda.get_current_device()
start_time = time.time()
d_a = cuda.to_device(a)
d_b = cuda.to_device(b)
d_c = cuda.device_array_like(c)

#threads_per_block = int(np.sqrt(device.WARP_SIZE))
#print(threads_per_block)
#threads_per_block = int(np.ceil(np.log2(n)))
threads_per_block = int(16)
print(f"{np.ceil(np.log2(n))=}")
print(type(threads_per_block))
tpb = (threads_per_block, threads_per_block)
#block_per_grid_x = int(np.ceil(max(b.shape[0], c.shape[0]) / tpb[0]))
#block_per_grid_y = int(np.ceil(max(a.shape[1], c.shape[1]) / tpb[1]))
block_per_grid_x = int(np.ceil(n / tpb[0]))
#block_per_grid_y = int(np.ceil(k / tpb[1]))
block_per_grid_y = int(np.ceil(max(m, k) / tpb[1]))
BPG = (block_per_grid_x, block_per_grid_y)
print(f"{BPG=}")
print(f"{tpb=}")
start_time1 = time.time()

cuda_matmul[BPG, tpb](d_a, d_b, d_c)
print("--- %s seconds ---" % (time.time() - start_time1))
c = d_c.copy_to_host()
print("--- %s seconds ---" % (time.time() - start_time))
print(c)
'''
@cuda.jit
def fast_matmul(A, B, C):
    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x    # blocks per grid

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    tmp = float32(0.)
    for i in range(bpg):
        # Preload data into shared memory
        sA[tx, ty] = 0
        sB[tx, ty] = 0
        if x < A.shape[0] and (ty+i*TPB) < A.shape[1]:
          sA[tx, ty] = A[x, ty + i * TPB]
        if y < B.shape[1] and (tx+i*TPB) < B.shape[0]:
          sB[tx, ty] = B[tx + i * TPB, y]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Computes partial product on the shared memory
        for j in range(TPB):
            tmp += sA[tx, j] * sB[j, ty]

        # Wait until all threads finish computing
        cuda.syncthreads()
    if x < C.shape[0] and y < C.shape[1]:
        C[x, y] = tmp



#%%

x_h = a
y_h = b
z_h = c

x_d = cuda.to_device(x_h)
y_d = cuda.to_device(y_h)
z_d = cuda.to_device(z_h)

TPB = 32
threadsperblock = (TPB, TPB)
blockspergrid_x = math.ceil(z_h.shape[0] / threadsperblock[0])
blockspergrid_y = math.ceil(z_h.shape[1] / threadsperblock[1])
blockspergrid = (blockspergrid_x, blockspergrid_y)

fast_matmul[blockspergrid, threadsperblock](x_d, y_d, z_d)
z_h = z_d.copy_to_host()
print(z_h)
print(x_h@y_h)
'''



