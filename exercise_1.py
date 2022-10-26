import numpy as np
from nptyping import Float64, NDArray, Shape

from dirichlet import Dirichlet
from verification import Verifier


def funct_diri(x: float) -> float:
    # return 1
    return -np.exp(x)


def v_diri(x: float) -> float:
    return 1


def verify_func(x: float) -> float:
    # return -(x**2) / 2 + x / 2 + 1
    return np.exp(x)


problem = Dirichlet(
    lower_bound=-2,
    upper_bound=1,
    lower_bc=np.exp(-2),
    upper_bc=np.exp(1),
    number_of_grid_points=20,
    f_func=funct_diri,
    v_func=v_diri,
)

verification = Verifier(verify_func=verify_func, grid=problem.grid)

n = np.arange(10, 110, 10)  # type: ignore
numeric_list: list[NDArray[Shape["N_p1, 1"], Float64]] = []
exact_list: list[NDArray[Shape["N_p1, 1"], Float64]] = []
for number in n:
    problem = Dirichlet(
        lower_bound=-2,
        upper_bound=1,
        lower_bc=np.exp(-2),
        upper_bc=np.exp(1),
        number_of_grid_points=number,
        f_func=funct_diri,
        v_func=v_diri,
    )
    verification = Verifier(verify_func=verify_func, grid=problem.grid)
    numeric_list.append(problem.solve_system())
    exact_list.append(verification.get_verify_result())

error_list: list[NDArray[Shape["N_p1, 1"], Float64]] = []
for i, vector in enumerate(numeric_list):
    error_list.append(np.linalg.norm(numeric_list[i] - exact_list[i]))  # type: ignore
print(error_list)
