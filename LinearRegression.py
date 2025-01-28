import matplotlib.pyplot as plt

def Linear_regression(x, y):
    n = len(x)

    # Summations
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(xi * yi for xi, yi in zip(x, y))
    sum_x2 = sum(xi ** 2 for xi in x)

    # Calculating slope (m) and intercept (b)
    m = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
    b = (sum_y - m * sum_x) / n

    return m, b

def predict(x, m, b):
    return m * x + b

def plot_regression(x, y, m, b):
    
    plt.scatter(x, y, color="black", label="Data points")

    # Plotting regression line
    y_pred = [m * xi + b for xi in x]
    plt.plot(x, y_pred, color="blue", label="Regression Line")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Linear Regression")
    plt.legend()
    plt.grid()
    plt.show()


weights = [140, 155, 159, 179, 192, 200, 212] 
heights = [60, 62, 67, 70, 71, 72, 75]        

# Perform linear regression
slope, intercept = Linear_regression(weights, heights)

plot_regression(weights, heights, slope, intercept)

#Sample usage of Predicting
new_weight = 185
predicted_height = predict(new_weight, slope, intercept)
print(f"Predicted height for weight {new_weight} lbs: {predicted_height:.4f} inches")
