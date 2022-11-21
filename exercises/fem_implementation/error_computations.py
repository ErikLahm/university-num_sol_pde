from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
from nptyping import Float64, NDArray, Shape
from scipy import interpolate  # type: ignore

from .fe_1d import FiniteElement1D
from .fem_grid import Grid
from .local_to_global import LocalToGlobalMap


@dataclass
class H1Error:
    grid: Grid
    fe: FiniteElement1D
    ltg: LocalToGlobalMap
    num_sol: NDArray[Shape["N,1"], Float64]
    exa_sol: NDArray[Shape["N,1"], Float64]
    exa_sol_der: NDArray[Shape["N,1"], Float64] = field(init=False)
    error_h1: float = field(init=False)
    error_l2: float = field(init=False)

    def __post_init__(self):
        self.error_l2 = np.linalg.norm(self.exa_sol - self.num_sol)  # type: ignore

    def fit_num_sol_to_piece_wise(self) -> list[NDArray[Shape["1,Degree_p1"], Float64]]:
        poly_coeffs: list[NDArray[Shape["1,Degree_p1"], Float64]] = []
        for l in range(self.grid.num_el):
            x = [self.affine_map(local_node, l) for local_node in self.fe.nodes_ref]
            y = [
                list(
                    self.num_sol[
                        l * self.fe.degree : self.fe.ndof + (l * self.fe.degree)
                    ]
                )
            ]
            element_coeffs = np.polyfit(x, y[0], self.fe.degree)
            poly_coeffs.append(element_coeffs)
        return poly_coeffs

    def plot_piecewise(
        self, piece_wise_coeff: list[NDArray[Shape["1,Degree_p1"], Float64]]
    ):
        fig, ax = plt.subplots()  # type: ignore
        for l in range(self.grid.num_el):
            x_interval = np.linspace(
                self.grid.vertices[l], self.grid.vertices[l + 1], 50
            )
            ax.plot(  # type: ignore
                x_interval,
                np.polyval(piece_wise_coeff[l], x_interval),
                label=f"Element {l}",
            )
        x_domain = np.linspace(self.grid.a, self.grid.b, 100)  # type: ignore
        ax.grid(True)  # type: ignore
        ax.legend()  # type: ignore
        ax.set_title("Plot of the interpolation polynomial")  # type: ignore
        plt.show()  # type: ignore

    def np_coeffs_to_scipy_ppoly(
        self, piece_wise_coeff: list[NDArray[Shape["1,Degree_p1"], Float64]]
    ) -> interpolate.PPoly:
        print(piece_wise_coeff)
        piece_wise_coeff = np.reshape(  # type: ignore
            piece_wise_coeff, (self.grid.num_el, self.fe.degree + 1)
        )
        print(piece_wise_coeff)
        piece_wise_coeff = piece_wise_coeff.T  # type: ignore
        # poly_coeffs = np.reshape(poly_coeffs, (self.fe.degree + 1, self.grid.num_el))
        print(piece_wise_coeff)
        interpolation_pol = interpolate.PPoly(piece_wise_coeff, self.grid.vertices)
        return interpolation_pol

    def affine_map(self, chi: float, index_elem: int) -> float:
        return self.grid.vertices[index_elem] + self.grid.element_dims[index_elem] * chi
