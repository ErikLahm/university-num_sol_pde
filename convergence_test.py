from dataclasses import dataclass

from nptyping import Float64, NDArray, Shape


@dataclass
class ConvergenceTest:
    n_array: NDArray[Shape["*, 1"], Float64]
    result_array: list[float]

    def compute_error(self):
        pass
