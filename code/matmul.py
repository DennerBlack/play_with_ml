import numpy as np


def matmul(A, B):
    assert A.shape[1] == B.shape[0]
    C = np.zeros((A.shape[0], B.shape[1]))

    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            temp = 0
            for k in range(A.shape[1]):
                temp += A[i][k] * B[k][j]

            C[i][j] = temp
    return C





n = 10
m = 1
k = 1
a = np.random.random(size=(m, n)).astype("float32")
b = np.random.random(size=(n, k)).astype("float32")
#c = np.zeros((1, 1)).astype("float32")


c = a.dot(b)
print(c)
c = matmul(a, b)
print(c)
"""
A = np.array([[1, 2],
              [3, 4]])

B = np.array([[4, 3],
              [2, 1]])

C = matmul(A, B)

print(f"first test: {C=}")

A = np.array([[1, 2, 3]])

B = np.array([[1],
              [2],
              [3]])

C = matmul(A, B)

print(f"second test: {C=}")

A = np.array([[1],
              [2],
              [3]])

B = np.array([[1, 2, 3]])

C = matmul(A, B)

print(f"third test: {C=}")"""
"""A = np.array([[1, 2],
              [3, 4]])

B = np.array([[4, 3],
              [2, 1]])

C = A.dot(B)

print(f"first test: {C=}")

A = np.array([[1, 2, 3]])

B = np.array([[1],
              [2],
              [3]])

C = A.dot(B)

print(f"second test: {C=}")

A = np.array([[1],
              [2],
              [3]])

B = np.array([[1, 2, 3]])

C = A.dot(B)

print(f"third test: {C=}")"""
