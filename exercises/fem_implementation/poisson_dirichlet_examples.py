from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class GetPoissonDirichletProblem:
    problem: str
    a: float = field(init=False)
    b: float = field(init=False)
    func: Callable[[float], float] = field(init=False)
    exact_sol: Callable[[float], tuple[float, float]] = field(init=False)

    def __post_init__(self):
        match self.problem:
            case "problem 2":
                self.a = 0
                self.b = 1
                self.func = self.force_func_2  # type: ignore
                self.exact_sol = self.exact_solution_2  # type: ignore
            case "problem 1":
                self.a = 0
                self.b = np.pi
                self.func = self.force_func_1
                self.exact_sol = self.exact_solution_1
            case "problem 3":
                self.a = 0
                self.b = 1
                self.func = self.force_func_3
                self.exact_sol = self.exact_solution_3
            case "problem 4":
                self.a = 0
                self.b = 2
                self.func = self.force_func_4
                self.exact_sol = self.exact_solution_4
            case _:
                print(
                    "ValueError: problem input must be of the form: problem x, where x is a number in {1,2}"
                )

    def exact_solution_1(self, x: float) -> tuple[float, float]:
        exact_sol = np.exp(x) * np.sin(x)
        exact_sol_der = np.exp(x) * (np.sin(x) + np.cos(x))
        return exact_sol, exact_sol_der

    def force_func_1(self, x: float) -> float:
        return -2 * np.cos(x) * np.exp(x)

    def exact_solution_2(self, x: float) -> tuple[float, float]:
        exact_sol_p = np.polyfit(np.linspace(self.a, self.b, 5), [0, 0.5, -1, 2, 0], 4)
        exact_sol = np.polyval(exact_sol_p, x)
        exact_sol_der = np.polyval(np.polyder(exact_sol_p), x)
        return float(exact_sol), float(exact_sol_der)

    def force_func_2(self, x: float) -> float:
        exact_sol_p = np.polyfit(np.linspace(self.a, self.b, 5), [0, 0.5, -1, 2, 0], 4)
        return -np.polyval(np.polyder(np.polyder(exact_sol_p)), x)  # type: ignore

    def exact_solution_3(self, x: float) -> tuple[float, float]:
        exact_sol = np.sin(np.pi * x)
        exact_sol_der = np.pi * np.cos(np.pi * x)
        return exact_sol, exact_sol_der

    def force_func_3(self, x: float) -> float:
        return np.pi**2 * np.sin(np.pi * x)

    def exact_solution_4(self, x: float) -> tuple[float, float]:
        exact_sol = -((1 - x) ** 2) + 1
        exact_sol_der = 2 - 2 * x
        return exact_sol, exact_sol_der

    def force_func_4(self, x: float) -> float:
        return 2
