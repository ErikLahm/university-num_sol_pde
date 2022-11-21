from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np
from nptyping import Float64, NDArray, Shape


class ProblemType(Enum):
    DIRICHLET_DIRICHLET = auto()
    DIRICHLET_NEUMANN = auto()
    NEUMANN_DIRICHLET = auto()
    NEUMANN_NEUMANN = auto()


@dataclass
class LocalToGlobalMap:
    num_elem: int
    num_loc_dof: int
    problem_type: ProblemType
    ltg: NDArray[Shape["N_elements,N_local_dofs"], Float64] = field(init=False)

    def __post_init__(self):
        self.ltg = self.assemble_ltg()

    def assemble_ltg(self) -> NDArray[Shape["N_elements,N_local_dofs"], Float64]:
        ltg = np.arange(int(self.num_elem * self.num_loc_dof)).reshape(  # type: ignore
            (self.num_elem, self.num_loc_dof)
        )
        for i, row in enumerate(ltg):
            row -= i
        match self.problem_type:
            case ProblemType.DIRICHLET_DIRICHLET:
                for element in ltg:
                    element -= 1
                ltg[0, 0] = -1
                ltg[self.num_elem - 1, self.num_loc_dof - 1] = -1
            case ProblemType.DIRICHLET_NEUMANN:
                for element in ltg:
                    element -= 1
                ltg[0, 0] = -1
            case ProblemType.NEUMANN_DIRICHLET:
                ltg[self.num_elem - 1, self.num_loc_dof - 1] = -1
            case ProblemType.NEUMANN_NEUMANN:
                return ltg  # type: ignore
        return ltg  # type: ignore
