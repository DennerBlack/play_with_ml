import numpy as np

streetlights = np.array([[1, 0, 1],
                         [0, 1, 1],
                         [0, 0, 1],
                         [1, 1, 1],
                         [0, 1, 1],
                         [1, 0, 1]])

walk_vs_stop = np.array([0, 1, 0, 1, 1, 0])

weights = np.array([0.5, 0.48, -0.7])
alpha = 0.1



for iter in range(40):
    error_for_all_lights = 0
    for row in range(len(walk_vs_stop)):
        input = streetlights[row]
        goal_pred = walk_vs_stop[row]

        pred = input.dot(weights)
        err = (goal_pred - pred)**2
        error_for_all_lights += err
        delta = pred - goal_pred
        weights -= alpha*input*delta
        print("Error:" + str(err) + " Prediction:" + str(pred))
    print("Error:" + str(error_for_all_lights) + "\n")



