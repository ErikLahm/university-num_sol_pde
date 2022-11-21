from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
from nptyping import Float64, NDArray, Shape


@dataclass
class FiniteElement1D:
    degree: int
    ndof: int = field(init=False)
    nodes_ref: NDArray[Shape["Ndof, 1"], Float64] = field(init=False)
    pol: list[NDArray[Shape["1, Ndof"], Float64]] = field(init=False)
    pol_deri: list[NDArray[Shape["1, Ndof"], Float64]] = field(init=False)
    local_matrix: NDArray[Shape["Ndof, Ndof"], Float64] = field(init=False)

    def __post_init__(self):
        self.ndof = self.degree + 1
        self.nodes_ref = np.linspace(0, 1, self.ndof)
        self.pol = self.get_basis_coeffs()
        self.pol_deri = [np.polyder(self.pol[i]) for i in range(self.ndof)]
        self.local_matrix = self.assemble_local_matrix()

    def get_basis_coeffs(self) -> list[NDArray[Shape["Nnodes, Ndof"], Float64]]:
        y_coords = np.identity(self.ndof)
        basis_coeffs = [
            np.polyfit(self.nodes_ref, y_coords[row], self.degree)
            for row in range(self.ndof)
        ]
        return basis_coeffs

    def assemble_local_matrix(self) -> NDArray[Shape["Ndof, Ndof"], Float64]:
        matrix = np.ones(shape=(self.ndof, self.ndof))
        for i in range(self.ndof):
            for j in range(self.ndof):
                anti_deri = np.polyint(np.polymul(self.pol_deri[i], self.pol_deri[j]))
                matrix[i, j] = matrix[i, j] * (
                    np.polyval(anti_deri, 1) - np.polyval(anti_deri, 0)
                )
        return matrix

    def plot_basis_pols(self):
        fig, ax = plt.subplots()  # type: ignore
        x_base = np.linspace(0, 1, 100)
        for i in range(self.ndof):
            ax.plot(x_base, np.polyval(self.pol[i], x_base), label=f"$\\varphi_{i}$")  # type: ignore
        ax.grid(True)  # type: ignore
        ax.legend()  # type: ignore
        ax.set_title(f"the {self.ndof} basis functions in the reference interval")  # type: ignore
        plt.show()  # type: ignore

    def plot_basis_pols_der(self):
        fig, ax = plt.subplots()  # type: ignore
        x_base = np.linspace(0, 1, 100)
        for i in range(self.ndof):
            ax.plot(x_base, np.polyval(self.pol_deri[i], x_base), label=f"$\\varphi\prime_{i}$")  # type: ignore
        ax.grid(True)  # type: ignore
        ax.legend()  # type: ignore
        ax.set_title(f"the {self.ndof} derived basis functions in the reference interval")  # type: ignore
        plt.show()  # type: ignore
