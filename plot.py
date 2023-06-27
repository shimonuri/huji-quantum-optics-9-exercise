import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 22})
# make the plot bigger
plt.rcParams["figure.figsize"] = (20, 10)


def plot_coefficients(r):
    def coefficient(n):
        if n % 2 == 1:
            return 0

        result = 1 / np.sqrt(np.cosh(r))
        result *= np.power(np.tanh(r), n)
        result *= np.math.factorial(2 * n) ** 0.5
        result /= np.math.factorial(n)
        result /= np.power(2, n)
        return result

    abs_coefficient = lambda n: np.abs(coefficient(n) ** 2)
    n = np.arange(0, 32, 2)
    plt.scatter(n, [abs_coefficient(i) for i in n])
    plt.xlabel("n")
    plt.ylabel("Probability of n photons")
    plt.show()


if __name__ == "__main__":
    plot_coefficients(r=10)
