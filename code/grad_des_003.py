weight = 0
input = 0.5
goal_pred = 0.8

for iteration in range(40):

    pred = input * weight
    error = (pred - goal_pred) ** 2
    print("Error:" + str(error) + " Prediction:" + str(pred))

    dir_and_amount = (pred - goal_pred) * input
    weight -= dir_and_amount