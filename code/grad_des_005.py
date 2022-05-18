weight, goal_pred, input = (0.5, 0.8, 2)
alpha = 0.5
err_last = (input * weight - goal_pred) ** 2
for iteration in range(20):
    print("-----\nWeight:" + str(weight))

    pred = input * weight
    error = (pred - goal_pred) ** 2

    assert (error <= err_last), "too much input"

    delta = pred - goal_pred
    weight_delta = delta*input
    weight -= (weight_delta*alpha)
    err_last = (pred - goal_pred) ** 2
    print("Error:" + str(error) + " Prediction:" + str(pred))
    print("Delta:" + str(delta) + " Weight Delta:" + str(weight_delta))

    