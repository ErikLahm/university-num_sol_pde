import matplotlib.pyplot as plt
import numpy as np
from nptyping import Float64, NDArray, Shape

from pde_implementation.advection_diffusion import (  # pylint: disable=import-error
    Approximation,
    AvecDiff,
)
from pde_implementation.verification import Verifier  # pylint: disable=import-error

BETA = 1
MU = 0.1
approximation_value = Approximation.SECOND_ORDER


def verify_func(x: float) -> float:
    return (1 / BETA) * (
        x
        - (
            ((np.exp(-BETA / MU) - np.exp((BETA / MU) * (x - 1))))
            / ((np.exp(-BETA / MU) - 1))
        )
    )


# ________________________________________________________________________________
# TASK ONE PART ONE

n = np.arange(20, 80, 10)  # type: ignore
numeric_list: list[NDArray[Shape["N_p1, 1"], Float64]] = []
exact_list: list[NDArray[Shape["N_p1, 1"], Float64]] = []
grid_list: list[NDArray[Shape["1, N_p1"], Float64]] = []
for number in n:
    problem = AvecDiff(
        lower_bound=0,
        upper_bound=1,
        lower_bc=0,
        upper_bc=0,
        mu=0.1,
        beta=1,
        number_of_grid_points=number,
    )
    verification = Verifier(verify_func=verify_func, grid=problem.grid)
    grid_list.append(problem.grid)
    numeric_list.append(problem.solve_system(approx=approximation_value))
    exact_list.append(verification.get_verify_result())

fig, axs = plt.subplots(2, 3)  # type: ignore
plot_number = 0
for i in range(2):
    for j in range(3):
        axs[i, j].plot(grid_list[plot_number], exact_list[plot_number], label="exact")  # type: ignore
        axs[i, j].scatter(  # type: ignore
            grid_list[plot_number],
            numeric_list[plot_number],
            label="numeric",
            color="red",
        )
        axs[i, j].set_xlabel("x")  # type: ignore
        axs[i, j].set_ylabel("y")  # type: ignore
        axs[i, j].set_title(f"N={n[plot_number]}")  # type: ignore
        axs[i, j].legend()  # type: ignore
        plot_number += 1
# plt.show()

# TASK ONE PART TWO

mu = np.linspace(0.001, 0.1, 6)  # type: ignore
numeric_list: list[NDArray[Shape["N_p1, 1"], Float64]] = []
exact_list: list[NDArray[Shape["N_p1, 1"], Float64]] = []
grid_list: list[NDArray[Shape["1, N_p1"], Float64]] = []
for number in mu:
    problem = AvecDiff(
        lower_bound=0,
        upper_bound=1,
        lower_bc=0,
        upper_bc=0,
        mu=number,
        beta=1,
        number_of_grid_points=20,
    )
    MU = number  # type: ignore
    verification = Verifier(verify_func=verify_func, grid=problem.grid)
    grid_list.append(problem.grid)
    numeric_list.append(problem.solve_system(approx=approximation_value))
    exact_list.append(verification.get_verify_result())

fig, axs = plt.subplots(2, 3)  # type: ignore
plot_number = 0
for i in range(2):
    for j in range(3):
        axs[i, j].plot(grid_list[plot_number], exact_list[plot_number], label="exact")  # type: ignore
        axs[i, j].scatter(  # type: ignore
            grid_list[plot_number],
            numeric_list[plot_number],
            label="numeric",
            color="red",
        )
        axs[i, j].set_xlabel("x")  # type: ignore
        axs[i, j].set_ylabel("y")  # type: ignore
        axs[i, j].set_title(f"$\\nu=${mu[plot_number]}")  # type: ignore
        axs[i, j].legend()  # type: ignore
        plot_number += 1
# plt.show()

# TASK ONE PART THREE

n = np.linspace(20, 500, 6, dtype=int)  # type: ignore
numeric_list: list[NDArray[Shape["N_p1, 1"], Float64]] = []
exact_list: list[NDArray[Shape["N_p1, 1"], Float64]] = []
grid_list: list[NDArray[Shape["1, N_p1"], Float64]] = []
for number in n:
    problem = AvecDiff(
        lower_bound=0,
        upper_bound=1,
        lower_bc=0,
        upper_bc=0,
        mu=0.001,
        beta=1,
        number_of_grid_points=number,
    )
    MU = 0.001  # type: ignore
    verification = Verifier(verify_func=verify_func, grid=problem.grid)
    grid_list.append(problem.grid)
    numeric_list.append(problem.solve_system(approx=approximation_value))
    exact_list.append(verification.get_verify_result())

fig, axs = plt.subplots(2, 3)  # type: ignore
plot_number = 0
for i in range(2):
    for j in range(3):
        axs[i, j].plot(grid_list[plot_number], exact_list[plot_number], label="exact")  # type: ignore
        axs[i, j].scatter(  # type: ignore
            grid_list[plot_number],
            numeric_list[plot_number],
            label="numeric",
            color="red",
        )
        axs[i, j].set_xlabel("x")  # type: ignore
        axs[i, j].set_ylabel("y")  # type: ignore
        axs[i, j].set_title(f"N={n[plot_number]}, $\\nu=0.001$")  # type: ignore
        axs[i, j].legend()  # type: ignore
        plot_number += 1
plt.show()  # type: ignore


error_list: list[NDArray[Shape["N_p1, 1"], Float64]] = []
for i, vector in enumerate(numeric_list):
    error_list.append(np.linalg.norm(numeric_list[i] - exact_list[i]))  # type: ignore
