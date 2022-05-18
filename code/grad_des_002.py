weight = 0.5
input = 0.5
goal_pred = 0.8

step_amount = 0.001

for iteration in range(1101):

    pred = input * weight
    error = (pred - goal_pred) ** 2
    print("Error:" + str(error) + " Prediction:" + str(pred))

    up_pred = input * (weight + step_amount)
    up_error = (up_pred - goal_pred) ** 2

    down_pred = input * (weight - step_amount)
    down_error = (down_pred - goal_pred) ** 2

    if down_error < up_error:
        weight -= step_amount
    if down_error > up_error:
        weight += step_amount
