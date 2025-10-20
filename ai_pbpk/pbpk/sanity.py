from __future__ import annotations

import numpy as np


def check_mass_balance(y: np.ndarray, u_series: np.ndarray, dt: float, tol: float = 1e-4) -> bool:
	# For a closed system with input only, ensure non-negativity and reasonable total mass trend
	if np.any(y < -1e-9):
		return False
	# Weak mass balance: total mass at end <= input mass + small numerical slack
	input_mass = np.trapz(u_series, dx=dt)
	total_end = float(np.sum(y[-1]))
	return total_end <= input_mass * (1 + tol) + tol


