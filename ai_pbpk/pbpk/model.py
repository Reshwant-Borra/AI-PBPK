from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np


@dataclass
class PBPKParams:
	# Placeholder parameters for 8 compartments
	# k vector contains transfer/elimination rates
	k: np.ndarray  # shape (Nparams,)
	V: np.ndarray  # compartment volumes, shape (8,)


def pbpk_ode(t: float, y: np.ndarray, params: PBPKParams, u_fn: Callable[[float], float]) -> np.ndarray:
	# 8-compartment linear model with input u(t) to plasma (compartment 0)
	# dy/dt = A*y + B*u
	A = np.zeros((8, 8))
	# Simple chain with bidirectional links and elimination from plasma
	for i in range(7):
		A[i, i] -= params.k[i]
		A[i+1, i] += params.k[i]
		A[i, i+1] += params.k[i] * 0.2
		A[i+1, i+1] -= params.k[i] * 0.2
	A[0, 0] -= params.k[7]

	B = np.zeros(8)
	B[0] = 1.0 / max(params.V[0], 1e-9)

	dydt = A @ y + B * float(u_fn(t))
	return dydt


