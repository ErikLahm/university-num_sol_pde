import matplotlib.pyplot as plt
import numpy as np
from fem_implementation.error_computations import H1Error
from fem_implementation.fe_1d import FiniteElement1D
from fem_implementation.fem_1d_poisson import FEMPoisson1D
from fem_implementation.fem_grid import Grid
from fem_implementation.local_to_global import LocalToGlobalMap, ProblemType
from fem_implementation.poisson_dirichlet_examples import GetPoissonDirichletProblem
from fem_implementation.test_gauss_quad import GaussLegendQuad  # type: ignore
from nptyping import Float64, NDArray, Shape


def f_integrand(x: NDArray[Shape["N,1"], Float64]) -> NDArray[Shape["N,1"], Float64]:
    return np.exp(x)


def f_primitive(x: float) -> float:
    return np.exp(x)


# ________________________________________________________________________________________
# part one: Gauss-Legendre Quadrature
# ________________________________________________________________________________________
# gauss_test = GaussLegendQuad(domain=(0, 1), degrees=list(range(1, 6)))
# gauss_test.do_test(f_integrand, f_primitive)
# gauss_test.plot()

# ________________________________________________________________________________________
# test Poisson assemble
# ________________________________________________________________________________________
number_of_elements = 10
poisson_example = GetPoissonDirichletProblem("problem 1")
grid = Grid(
    a=poisson_example.a, b=poisson_example.b, num_el=number_of_elements, uniformity=0
)
fe = FiniteElement1D(degree=2)
# fe.plot_basis_pols()
# fe.plot_basis_pols_der()
ltg = LocalToGlobalMap(
    num_elem=number_of_elements,
    num_loc_dof=fe.ndof,
    problem_type=ProblemType.DIRICHLET_DIRICHLET,
)
poisson_assembling = FEMPoisson1D(func=poisson_example.func, grid=grid, fe=fe, ltg=ltg)  # type: ignore
# print(ltg.ltg)
print(fe.local_matrix)
print(poisson_assembling.rhs)
print(poisson_assembling.coeff_matrix)
# print(grid.element_dims)
# ________________________________________________________________________________________
num_sol = np.linalg.solve(poisson_assembling.coeff_matrix, poisson_assembling.rhs)
num_sol = np.append(num_sol, [[0]])  # type: ignore
num_sol = np.insert(num_sol, 0, 0)  # type: ignore
# num_sol = num_sol * 0.35e-12  # why this factor degree 5????
exact_sol = [
    poisson_example.exact_sol(x)[0]
    for x in np.linspace(
        poisson_example.a, poisson_example.b, poisson_assembling.global_dof + 2
    )
]
# ________________________________________________________________________________________
# plot solution:
# ________________________________________________________________________________________
fig, ax = plt.subplots()  # type: ignore
ax.plot(  # type: ignore
    np.linspace(
        poisson_example.a, poisson_example.b, poisson_assembling.global_dof + 2
    ),
    num_sol,
    label="numerical solution",
)
ax.plot(  # type: ignore
    np.linspace(
        poisson_example.a, poisson_example.b, poisson_assembling.global_dof + 2
    ),
    exact_sol,
    label="exact solution",
)
ax.grid(True)  # type: ignore
ax.legend()  # type: ignore
ax.set_title("plotting the numerical and the exact solution")  # type: ignore
plt.show()  # type: ignore
# ________________________________________________________________________________________

# calculate error:
error = H1Error(grid=grid, fe=fe, ltg=ltg, num_sol=num_sol, exa_sol=exact_sol)  # type: ignore
print(f"The L_2 norm of the error is: {error.error_l2}")
error.plot_piecewise(error.fit_num_sol_to_piece_wise())
