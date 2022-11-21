from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from nptyping import Float64, NDArray, Shape

from .fe_1d import FiniteElement1D
from .fem_grid import Grid
from .local_to_global import LocalToGlobalMap


@dataclass
class FEMPoisson1D:
    func: Callable[[NDArray[Shape["N,1"], Float64]], NDArray[Shape["N,1"], Float64]]
    grid: Grid
    fe: FiniteElement1D
    ltg: LocalToGlobalMap
    coeff_matrix: NDArray[Shape["N,N"], Float64] = field(init=False)
    rhs: NDArray[Shape["N,1"], Float64] = field(init=False)
    global_dof: int = field(init=False)

    def __post_init__(self):
        self.global_dof = np.amax(self.ltg.ltg) + 1  # type: ignore
        self.rhs = self.assemble_rhs()
        self.coeff_matrix = self.assemble_matrix()

    def assemble_matrix(self) -> NDArray[Shape["N,N"], Float64]:
        a = np.zeros(shape=(self.global_dof, self.global_dof))
        for l in range(self.grid.num_el):
            for k in range(self.fe.ndof):
                a_row = self.ltg.ltg[l, k]
                if a_row < 0:
                    continue
                for j in range(self.fe.ndof):
                    a_col = self.ltg.ltg[l, j]
                    if a_col < 0:
                        continue
                    integral = (
                        1 / self.grid.element_dims[l] * self.fe.local_matrix[k, j]
                    )
                    a[a_row, a_col] += integral
        return a

    def assemble_rhs(self) -> NDArray[Shape["N,1"], Float64]:
        b = np.zeros(shape=(self.global_dof, 1))
        for l in range(self.grid.num_el):
            for d in range(self.fe.ndof):
                b_row: int = self.ltg.ltg[l, d]
                if b_row >= 0:
                    x, w = np.polynomial.legendre.leggauss(self.fe.degree + 1)  # type: ignore
                    s: float = 1 / 2
                    a: float = 1 / 2
                    integral = np.sum(  # type: ignore
                        s
                        * w
                        * self.func(self.affine_map(s * x + a, l))  # type: ignore
                        * np.polyval(s * x + a, self.fe.pol[d])  # type: ignore
                        * self.grid.element_dims[l]  # type: ignore
                    )
                    b[b_row] += integral  # type: ignore
        return b

    def affine_map(self, chi: float, index_elem: int) -> float:
        return self.grid.vertices[index_elem] + self.grid.element_dims[index_elem] * chi
