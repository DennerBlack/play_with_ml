from thetanet import *
import numpy as np

np.random.seed(0)

data = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), autograd=True)
target = Tensor(np.array([[0], [1], [0], [1]]), autograd=True)

model = Sequential([Linear(2, 3), Linear(3, 1)])

optim = SGD(parameters=model.get_parameters(), alpha=0.05)

for i in range(10):
    pred = model.forward(data)

    loss = ((pred - target) * (pred - target)).sum(0)

    loss.backward(Tensor(np.ones_like(loss.data)))

    optim.step()
    print(loss)


