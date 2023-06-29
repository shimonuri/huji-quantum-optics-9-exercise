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


def plot_variance_to_phi_experimental_data():
    data = pd.read_csv(r"data/variance_to_phi.csv", names=["phi", "variance"])
    plt.scatter(data["phi"], data["variance"], label="Experimental data")
    plt.xlabel(r"$\phi$")
    plt.ylabel("Variance [dB]")


def plot_variance_to_phi_theory(efficiency, squeezed_quadrature_variance):
    anti_squeezed_quadrature_variance = -squeezed_quadrature_variance * 3.1
    variance = lambda phi: efficiency * (
        squeezed_quadrature_variance * np.sin(phi) ** 2
        + anti_squeezed_quadrature_variance * np.cos(phi) ** 2
    ) + (1 - efficiency) * (1 / 2)
    phi = np.linspace(0, 2 * np.pi, 100)
    plt.plot(phi, variance(phi), label="Theoretical prediction", color="red")


def plot_variance_to_phi():
    plot_variance_to_phi_experimental_data()
    plot_variance_to_phi_theory(efficiency=0.88, squeezed_quadrature_variance=-2.8)
    plt.xticks(
        [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
        [0, r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"],
    )
    plt.legend()
    plt.show()


def plot_squeezed_vs_anti_squeezed():
    plot_squeezed_vs_anti_squeezed_data()
    plot_squeezed_vs_anti_squeezed_theory()
    plt.legend()
    plt.show()


def plot_histogram(x_b_variance, p_b_variance, efficiency):
    variance_in_db = lambda phi: efficiency * (
        x_b_variance * np.sin(phi) ** 2 + p_b_variance * np.cos(phi) ** 2
    ) + (1 - efficiency) * (1 / 2)
    variance = lambda phi: 10 ** (-variance_in_db(phi) / 10) / 2
    phis = np.linspace(0, 2 * np.pi, 100)
    phi_to_hist = {}
    for phi in phis:
        samples = np.random.normal(0, np.sqrt(variance(phi)), 100000)
        phi_to_hist[phi] = np.histogram(samples, bins=1000, density=True)[0]
    plt.rcParams["image.cmap"] = "hot"
    # plot 2d heat map
    plt.imshow(
        np.array(list(phi_to_hist.values())).T,
        extent=[0, 2 * np.pi, -3, 3],
        aspect="auto",
        origin="lower",
    )
    plt.xlabel(r"$\phi$")
    plt.ylabel("x")
    plt.xticks(
        [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
        [0, r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"],
    )
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    plot_p_n()
    plot_squeezed_vs_anti_squeezed()
    plot_variance_to_phi()
    plot_histogram(x_b_variance=0.25, p_b_variance=0.25, efficiency=0.88)
    plot_histogram(x_b_variance=-2.8, p_b_variance=2.8 * 3.1, efficiency=0.88)
