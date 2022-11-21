from dataclasses import dataclass, field
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from nptyping import Float64, NDArray, Shape


@dataclass
class GaussLegendQuad:
    domain: tuple[float, float]
    degrees: list[int]
    gauss_result: list[float] = field(init=False)
    exact_result: list[float] = field(init=False)
    error: list[float] = field(init=False)

    def gauss_legendre_integrate(
        self,
        func: Callable[
            [NDArray[Shape["N,1"], Float64]], NDArray[Shape["N,1"], Float64]
        ],
        domain: tuple[float, float],
        deg: int,
    ) -> float:
        x, w = np.polynomial.legendre.leggauss(deg)  # type: ignore
        s = (domain[1] - domain[0]) / 2
        a = (domain[1] + domain[0]) / 2
        return np.sum(s * w * func(s * x + a))  # type: ignore

    def exact_integral(
        self,
        func: Callable[[float], float],
        domain: tuple[float, float],
    ):
        return func(domain[1]) - func(domain[0])

    def do_test(
        self,
        func_int: Callable[
            [NDArray[Shape["N,1"], Float64]], NDArray[Shape["N,1"], Float64]
        ],
        func_prim: Callable[[float], float],
    ):
        self.gauss_result = [
            self.gauss_legendre_integrate(func_int, self.domain, deg)
            for deg in self.degrees
        ]
        self.exact_result = np.full(  # type: ignore
            shape=(len(self.gauss_result), 1),
            fill_value=self.exact_integral(func_prim, self.domain),
        )
        self.error = abs(
            np.subtract(  # type: ignore
                np.asarray(self.gauss_result).reshape((len(self.degrees), 1)),
                self.exact_result,
            )
        )

    def plot(self):
        fig, ax = plt.subplots()  # type: ignore
        ax.plot(self.degrees, self.error, "+-", color="red", label="error")  # type: ignore
        plt.yscale(value="log")  # type: ignore
        ax.grid(True)  # type: ignore
        ax.legend()  # type: ignore
        ax.set_title(  # type: ignore
            "Error between exact solution of $\int_0^1 e^{x}dx$ \n and Gauss-Legendre Quadrature"  # type: ignore
        )
        plt.show()  # type: ignore
