from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import minimize

from ..pbpk.model import PBPKParams
from ..pbpk.solvers import simulate
from .objectives import objective_trajectory


@dataclass
class MPCConfig:
	Np: int
	Nc: int
	Ts: float
	w1: float
	w2: float
	u_max: float
	dose_total: float


class MPCController:
	def __init__(self, params: PBPKParams, mpc_cfg: MPCConfig):
		self.params = params
		self.cfg = mpc_cfg
		self.dose_budget = mpc_cfg.dose_total

	def optimize_step(self, y0: np.ndarray) -> float:
		Np = self.cfg.Np
		Ts = self.cfg.Ts
		u_max = self.cfg.u_max
		w1, w2 = self.cfg.w1, self.cfg.w2

		# Decision variables: piecewise-constant control over horizon
		x0 = np.zeros(Np)
		bounds = [(0.0, u_max)] * Np

		def u_profile(t: float, u_seq: np.ndarray) -> float:
			idx = min(int(t // Ts), Np - 1)
			return float(u_seq[idx])

		def objective(u_seq: np.ndarray) -> float:
			if np.sum(u_seq) * Ts > self.dose_budget + 1e-9:
				# Large penalty if exceeding budget
				return 1e6 + 1e3 * (np.sum(u_seq) * Ts - self.dose_budget)
			u_fn = lambda t: u_profile(t, u_seq)
			res = simulate(self.params, y0=y0, t_end=Np * Ts, dt=Ts, u_fn=u_fn)
			y_plasma = res.y[:, 0]
			y_tumor = res.y[:, -1]
			return objective_trajectory(y_tumor, y_plasma, u_seq, w1, w2, Ts)

		cons = []

		res = minimize(objective, x0, bounds=bounds, method="SLSQP", options={"maxiter": 100, "ftol": 1e-6})
		u_opt = np.clip(res.x, 0.0, u_max)
		# Enforce remaining dose budget
		dose_this_step = float(u_opt[0] * Ts)
		if dose_this_step > self.dose_budget:
			dose_this_step = self.dose_budget
		self.dose_budget -= dose_this_step
		return dose_this_step / Ts


