import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 22})


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
    n = np.arange(0, 20, 2)
    plt.scatter(n, [abs_coefficient(i) for i in n], label="Theoretical predication")
    plt.xlabel("n")
    plt.ylabel("Probability of n photons")


def plot_experiment_data():
    data = pd.read_csv(r"data/p_n_chances.csv", names=["n", "p(n)"])
    plt.scatter(data["n"], data["p(n)"], label="Experimental data")
    plt.xlabel("n")
    plt.ylabel("Probability of n photons")


if __name__ == "__main__":
    plot_experiment_data()
    plot_coefficients(r=2.26)
    plt.legend()
    plt.show()
