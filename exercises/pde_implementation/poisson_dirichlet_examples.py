from dataclasses import dataclass
from typing import Callable


@dataclass
class PoissonDirichletProblem:
    a: float
    b: float
    func: Callable[[float], float]
    exact_sol: Callable[[float], float]
