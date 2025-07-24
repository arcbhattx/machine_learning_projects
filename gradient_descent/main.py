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

        previous_cost = current_cost

        costs.append(current_cost)
        weights.append(current_weight)

        
        #calculation gradients:
        weight_derivative = -(2/n) * sum(x * (y-y_predicted))
        bias_derivative = -(2/n) * sum(y-y_predicted)
        

        #updating values:
        current_weight = current_weight - (learning_rate * weight_derivative)
        current_bias = current_bias - (learning_rate * bias_derivative)

        #print for each  iteration:
        print(f"Iteration {i+1}: Cost {current_cost}, Weight \
              {current_weight}, Bias {current_bias}")
        

        plt.figure(figsize=(8,6))
        plt.plot(weights, costs)
        plt.scatter(weights, costs, marker='o', color='red')
        plt.title("Costs vs Weights")
        plt.ylabel("Cost")
        plt.xlabel("Weight")
        plt.show()

        return current_weight,current_bias