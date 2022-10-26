from dataclasses import dataclass
from typing import Callable

import numpy as np
from nptyping import Float64, NDArray, Shape


@dataclass
class Dirichlet:
    lower_bound: int
    upper_bound: int
    lower_bc: float
    upper_bc: float
    number_of_grid_points: int
    f_func: Callable[[float], float]
    v_func: Callable[[float], float]

    @property
    def distance(self) -> float:
        return (self.upper_bound - self.lower_bound) / self.number_of_grid_points

    @property
    def grid(self) -> NDArray[Shape["1, N_p1"], Float64]:
        return np.linspace(
            start=self.lower_bound,
            stop=self.upper_bound,
            num=self.number_of_grid_points + 1,
        )

    @property
    def coef_matrix(self) -> NDArray[Shape["N_m1, N_m1"], Float64]:
        a_matrix = 2 * np.identity(self.number_of_grid_points - 1)
        a_matrix[0, 1] = -1
        a_matrix[self.number_of_grid_points - 2, self.number_of_grid_points - 3] = -1
        for i in range(1, self.number_of_grid_points - 2):
            a_matrix[i, i - 1] = -1
            a_matrix[i, i + 1] = -1
        return a_matrix

    @property
    def b_vector(self) -> NDArray[Shape["N_m1, 1"], Float64]:
        b_vector = np.ones(shape=(self.number_of_grid_points - 1, 1))
        for i in range(self.number_of_grid_points - 1):
            b_vector[i] = self.distance**2 * self.f_func(self.grid[i + 1])
        b_vector[0] = b_vector[0] + self.v_func(self.grid[1]) * self.lower_bc
        b_vector[self.number_of_grid_points - 2] = (
            b_vector[self.number_of_grid_points - 2]
            + self.v_func(self.grid[self.number_of_grid_points - 2]) * self.upper_bc
        )
        return b_vector

    def solve_system(self) -> NDArray[Shape["N_p1, 1"], Float64]:
        x = np.linalg.solve(self.coef_matrix, self.b_vector)
        x = np.insert(x, 0, self.lower_bc, axis=0)  # type: ignore
        x = np.append(x, np.array([[self.upper_bc]]), axis=0)  # type: ignore
        return x
