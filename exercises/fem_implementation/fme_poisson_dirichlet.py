from dataclasses import dataclass
from typing import Callable

import numpy as np
from fem_grid import Grid  # pylint: disable=import-error
from nptyping import Float64, NDArray, Shape

# class Grid(Protocol):
#     def vertices(self) -> NDArray[Shape["N_p1, 1"], Float64]:
#         ...

#     def element_dims(
#         self, vertices: NDArray[Shape["N_p1, 1"], Float64]
#     ) -> NDArray[Shape["N, 1"], Float64]:
#         ...


@dataclass
class PoissonDirichletAssemble:
    grid: Grid
    func: Callable[[float], float]

    @property
    def coef_matrix(self) -> NDArray[Shape["N_m1, N_m1"], Float64]:
        element_dims = self.grid.element_dims
        a_matrix = np.identity(self.grid.num_el - 1)
        for i in range(len(element_dims) - 1):
            a_matrix[i, i] = 1 / element_dims[i] + 1 / element_dims[i + 1]
        for i in range(1, len(element_dims) - 2):
            a_matrix[i, i - 1] = -1 / element_dims[i]  # type: ignore
            a_matrix[i, i + 1] = -1 / element_dims[i + 1]  # type: ignore
        a_matrix[0, 1] = -1 / element_dims[1]  # type: ignore
        a_matrix[self.grid.num_el - 2, self.grid.num_el - 3] = (  # type: ignore
            -1 / element_dims[self.grid.num_el - 1]
        )
        return a_matrix  # type: ignore

    @property
    def f_vector(self) -> NDArray[Shape["N_m1, 1"], Float64]:
        rhs = np.ones(shape=(self.grid.num_el - 1, 1))
        for i in range(self.grid.num_el - 1):
            rhs[i] = self.grid.element_dims[i] * self.func(
                self.grid.vertices[i]
                + (self.grid.vertices[i + 1] - self.grid.vertices[i]) / 2
            )
        return rhs
