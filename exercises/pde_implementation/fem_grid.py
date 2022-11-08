from dataclasses import dataclass, field

import numpy as np
from nptyping import Float64, NDArray, Shape


@dataclass
class Grid:
    a: float
    b: float
    num_el: int
    uniformity: float = 0
    vertices: NDArray[Shape["N_p1, 1"], Float64] = field(
        default_factory=lambda: np.zeros(shape=(100, 1))
    )

    def __post_init__(self):
        vertices = np.linspace(self.a, self.b, num=self.num_el + 1)
        randomnes = (
            self.uniformity
            * (self.b - self.a)
            / self.num_el
            * (np.random.rand(1, self.num_el - 1) - 0.5)
        )
        randomnes = np.append(randomnes, [[0]])  # type: ignore
        randomnes = np.insert(randomnes, 0, 0)  # type: ignore
        self.vertices = vertices + randomnes

    @property
    def element_dims(self) -> NDArray[Shape["N, 1"], Float64]:
        return np.diff(self.vertices)  # type: ignore
