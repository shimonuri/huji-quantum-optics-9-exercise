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


def plot_squeezed_vs_anti_squeezed_data():
    squeezed_state = pd.read_csv(
        r"data/fig_3_squeezed.csv", names=["alpha", "squeezing (db)"]
    )
    anti_squeezed_state = pd.read_csv(
        r"data/fig_3_anti_squeezed.csv", names=["alpha", "squeezing (db)"]
    )
    plt.scatter(
        anti_squeezed_state["alpha"],
        anti_squeezed_state["squeezing (db)"],
        label="Anti-squeezed state (data)",
    )
    plt.scatter(
        squeezed_state["alpha"],
        squeezed_state["squeezing (db)"],
        label="Squeezed state (data)",
    )
    plt.xlabel(r"$\alpha$")
    plt.ylabel("squeezing (db)")
    plt.axhline(y=0, color="black", linestyle="--", label="Zero point motion")


def plot_squeezed_vs_anti_squeezed_theory():
    anti_squeezed = lambda alpha: -np.log10((1 - alpha) / (1 + alpha)) * 10
    squeezed = lambda alpha: -np.log10((1 + alpha) / (1 - alpha)) * 10
    alpha = np.linspace(0.01, 0.99, 100)
    plt.plot(
        alpha,
        anti_squeezed(alpha),
        label="Theoretical prediction for anti-squeezed",
    )
    plt.plot(alpha, squeezed(alpha), label="Theoretical prediction for squeezed")


def plot_p_n():
    plot_experiment_data()
    plot_coefficients(r=2.26)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    plot_squeezed_vs_anti_squeezed_data()
    plot_squeezed_vs_anti_squeezed_theory()
    plt.legend()
    plt.show()
    # plot_p_n()
