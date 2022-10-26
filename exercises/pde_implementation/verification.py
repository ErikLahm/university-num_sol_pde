from dataclasses import dataclass
from typing import Callable

import numpy as np
from nptyping import Float64, NDArray, Shape


@dataclass
class Verifier:
    verify_func: Callable[[float], float]
    grid: NDArray[Shape["N, 1"], Float64]

    def get_verify_result(self):
        result = np.zeros(shape=(len(self.grid), 1))
        for i, value in enumerate(self.grid):
            result[i] = self.verify_func(value)
        return result
