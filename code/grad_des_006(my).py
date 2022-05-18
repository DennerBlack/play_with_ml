inp = 1.2
weight = 0.5
goal_pre = 0.8

alpha = 0.1

for i in range(200):
    pred = inp * weight
    error = (pred - goal_pre)**2
    delta = pred - goal_pre
    delta_wei = delta * inp
    weight -= delta_wei*alpha
    print("Error:" + str(error) + " Prediction:" + str(pred))


