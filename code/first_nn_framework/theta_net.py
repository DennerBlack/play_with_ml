import numpy as np

class Tensor(object):

    def __init__(self, data):
        self.data = np.array(data)

    def __add__(self, other):
        return Tensor(self.data + other.data)

    def __repr__(self):
        return str(self.data.__repr__())

    def __str__(self):
        return str(self.data.__str__())

x = Tensor([1,2,3,4,5,6,7])
print(f"{x=}")
y = x + x
print(f"{y=}")
z = y + np.array([1,2,3,4,5,6,7])
print(f"{z=}")
k = z + Tensor([1,2,3,4,5,6,7])
print(f"{k=}")