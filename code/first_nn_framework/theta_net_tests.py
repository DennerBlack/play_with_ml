from thetanet import *
import numpy as np

np.random.seed(0)

# исходные индексы
data = Tensor(np.array([1,2,1,2]), autograd=True)
# целевые индексы
target = Tensor(np.array([0,1,0,1]), autograd=True)
model = Sequential([Embedding(3,3), Tanh(), Linear(3,4)])
criterion = CrossEntropyLoss()
optim = SGD(parameters=model.get_parameters(), alpha=0.1)
for i in range(20):
    pred = model.forward(data)

    loss = criterion.forward(pred, target)
    loss.backward(Tensor(np.ones_like(loss.data)))
    optim.step()
    print(loss)

