from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
from nptyping import Float64, NDArray, Shape


class Approximation(Enum):
    FIRST_ORDER = auto()
    SECOND_ORDER = auto()


@dataclass
class AvecDiff:
    lower_bound: float
    upper_bound: float
    lower_bc: float
    upper_bc: float
    mu: float
    beta: float
    number_of_grid_points: int

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
    def p_coeff(self) -> float:
        return (self.beta * self.distance) / (2 * self.mu)

    @property
    def coef_matrix_sec_acc(self) -> NDArray[Shape["N_m1, N_m1"], Float64]:
        a_matrix = 2 * np.identity(self.number_of_grid_points - 1)
        a_matrix[0, 1] = self.p_coeff - 1
        a_matrix[self.number_of_grid_points - 2, self.number_of_grid_points - 3] = (
            -1 - self.p_coeff
        )
        for i in range(1, self.number_of_grid_points - 2):
            a_matrix[i, i - 1] = -1 - self.p_coeff
            a_matrix[i, i + 1] = self.p_coeff - 1
        return a_matrix

    @property
    def coef_matrix_first_acc(self) -> NDArray[Shape["N_m1, N_m1"], Float64]:
        a_matrix = (2 + 2 * self.p_coeff) * np.identity(self.number_of_grid_points - 1)
        a_matrix[0, 1] = -1
        a_matrix[self.number_of_grid_points - 2, self.number_of_grid_points - 3] = (
            -2 * self.p_coeff - 1
        )
        for i in range(1, self.number_of_grid_points - 2):
            a_matrix[i, i - 1] = -2 * self.p_coeff - 1
            a_matrix[i, i + 1] = -1
        return a_matrix

    @property  # TODO: this is for the special cas that bc are equal 0 -> generalise first and last entry
    def b_vector(self) -> NDArray[Shape["N_m1, 1"], Float64]:
        b_vector = (
            self.distance**2
            / self.mu
            * np.ones(shape=(self.number_of_grid_points - 1, 1))
        )
        return b_vector

    def solve_system(self, approx: Approximation) -> NDArray[Shape["N_p1, 1"], Float64]:
        if approx == Approximation.SECOND_ORDER:
            x = np.linalg.solve(self.coef_matrix_sec_acc, self.b_vector)
        elif approx == Approximation.FIRST_ORDER:
            x = np.linalg.solve(self.coef_matrix_first_acc, self.b_vector)
        x = np.insert(x, 0, self.lower_bc, axis=0)  # type: ignore
        x = np.append(x, np.array([[self.upper_bc]]), axis=0)  # type: ignore
        return x
