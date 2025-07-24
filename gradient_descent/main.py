import numpy as np
import matplotlib.pyplot as plt

def mean_squared_error(y_true, y_predicted):

    #caclulating loss:
    cost = np.sum((y_true-y_predicted)**2 / len(y_true))
    return cost

#gradient descent function
#parameters can be changed based on iterations and output

def gradient_descent(x,y,iterations = 1000, learning_rate=0.0001, stopping_threshold= 1e-6):
    

    current_weight = 0.1
    current_bias = 0.1
    learning_rate = learning_rate
    n = float(len(x))

    costs = []
    weights = []
    previous_cost = None

    for i in range(iterations):

        y_predicted = (current_weight*x) + current_bias
        current_cost = mean_squared_error(y, y_predicted)

        # If the change in cost is less than or equal to 
        # stopping_threshold we stop the gradient descent

        if previous_cost and abs(previous_cost-current_cost) <= stopping_threshold:
            break;