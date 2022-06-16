import numpy as np
import math

class Tensor(object):

    def __init__(self,
                 data,
                 autograd=False,
                 creators=None,
                 creation_op=None,
                 id=None):
        self.data = np.array(data)
        self.creators = creators
        self.creation_op = creation_op
        self.grad = None
        self.autograd = autograd
        self.children = {}
        if id is None:
            id = np.random.randint(0,1000000000)
        self.id = id
        if(creators is not None):
            for c in creators:
                if(self.id not in c.children):
                    c.children[self.id] = 1
                else:
                    c.children[self.id] += 1

    def all_children_grads_accounted_for(self):
        for id, cnt in self.children.items():
            if(cnt != 0):
                return False
        return True

    def backward(self, grad=None, grad_origin=None):
        if(self.autograd):
            if(grad_origin is not None):
                if(self.children[grad_origin.id] == 0):
                    raise Exception("sorry, can't backprop than once")
                else:
                    self.children[grad_origin.id] -= 1
            if(self.grad is None):
                self.grad = grad
            else:
                self.grad += grad

            assert  grad.autograd == False

            if(self.creators is not None and
                    (self.all_children_grads_accounted_for() or
                    grad_origin is None)):
                if(self.creation_op == 'add'):
                    self.creators[0].backward(self.grad, self)
                    self.creators[1].backward(self.grad, self)

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
