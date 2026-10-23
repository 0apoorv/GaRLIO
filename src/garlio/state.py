from dataclasses import dataclass, field

import numpy as np

from .so3 import Exp


@dataclass
class ImuState:
    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    v: np.ndarray = field(default_factory=lambda: np.zeros(3))
    p: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bg: np.ndarray = field(default_factory=lambda: np.zeros(3))
    ba: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def copy(self) -> "ImuState":
        return ImuState(
            R=self.R.copy(),
            v=self.v.copy(),
            p=self.p.copy(),
            bg=self.bg.copy(),
            ba=self.ba.copy(),
        )

    def boxplus(self, dx: np.ndarray) -> "ImuState":
        dx = np.asarray(dx, dtype=float).reshape(15)
        d_theta, d_v, d_p, d_bg, d_ba = np.split(dx, [3, 6, 9, 12])
        return ImuState(
            R=self.R @ Exp(d_theta),
            v=self.v + d_v,
            p=self.p + d_p,
            bg=self.bg + d_bg,
            ba=self.ba + d_ba,
        )
