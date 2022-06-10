import numpy as np


class Tensor(object):

    def __init__(self, data, creators=None, creation_op=None):
        self.data = np.array(data)
        self.creators = creators
        self.creation_op = creation_op
        self.grad = None

    def backward(self, grad):
        self.grad = grad

        if (self.creation_op == "add"):
            self.creators[0].backward(grad)
            self.creators[1].backward(grad)

    def __add__(self, other):
        print(self)
        print(other)
        return Tensor(self.data + other.data,
                      creators=[self, other],
                      creation_op="add")

    def __repr__(self):
        return str(self.data.__repr__())

    def __str__(self):
        return str(self.data.__str__())


x = Tensor([1,2,3,4,5])
y = Tensor([2,2,2,2,2])

z = x + y + x
z.backward(Tensor(np.array([1,1,1,1,1])))

print(x.grad)
print(y.grad)
print(z.creators)
print(z.creation_op)
