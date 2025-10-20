from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np
from scipy.integrate import solve_ivp

from .model import PBPKParams, pbpk_ode


@dataclass
class SimResult:
	t: np.ndarray
	y: np.ndarray


def simulate(params: PBPKParams, y0: np.ndarray, t_end: float, dt: float, u_fn: Callable[[float], float], method: str = "Radau") -> SimResult:
	t_eval = np.arange(0.0, t_end + 1e-9, dt)
	fun = lambda t, y: pbpk_ode(t, y, params, u_fn)
	sol = solve_ivp(fun, (0.0, t_end), y0, method=method, t_eval=t_eval, rtol=1e-6, atol=1e-8)
	return SimResult(t=sol.t, y=sol.y.T)


