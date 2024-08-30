from re import X
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def estimate_coeffs(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    deviations = (x - x_mean) * (y - y_mean)
    squared_deviations = (x - x_mean) ** 2
    ssxx = np.sum(squared_deviations)
    ssxy = np.sum(deviations)
    w1 = ssxy/ssxx
    w0 = y_mean - w1*x_mean
    return (w0, w1)

def plot_regression_line(x, y, w):

    plt.scatter(x, y, color = "gray", alpha=0.5, marker = "o", s = 30)

    # calculate prediction vector
    y_pred = w[0] + w[1]*x

    # plot the regression line
    plt.plot(x, y_pred, color = "r" )

    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()


def main():
    # Import data from CSV file.
    y,x = np.loadtxt('./homework2_linearRegression_data.csv', delimiter=',', unpack=True)
    plt.scatter(x, y, color = "gray", alpha=0.5, marker = "o", s = 30)
    plt.plot(x, y, color = "none" )
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

    w = estimate_coeffs(X_train, y_train)

    print ( "Estimated Coefficients:" )
    print ( "  w0: " + str(w[0]) + "   w1: " + str(w[1]) )

    # Plot the regression line with data
    plot_regression_line ( X_test, y_test, w )

    y_pred = w[0] + w[1]*X_test
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    print("R-squared: " + str(r_squared))


if __name__ == "__main__":
    main()
