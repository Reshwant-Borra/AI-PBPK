from __future__ import annotations

from typing import Dict

import numpy as np


def auc(y: np.ndarray, dt: float) -> float:
	return float(np.trapz(y, dx=dt))


def percent_injected_dose(y_tumor: np.ndarray, V_tumor: float, u_series: np.ndarray, dt: float) -> float:
	input_mass = float(np.trapz(u_series, dx=dt))
	if input_mass <= 0:
		return 0.0
	# Approximate terminal mass in tumor as concentration * volume
	terminal_mass = float(y_tumor[-1] * V_tumor)
	return 100.0 * terminal_mass / input_mass


def tumor_plasma_auc_ratio(y_tumor: np.ndarray, y_plasma: np.ndarray, dt: float) -> float:
	auc_t = auc(y_tumor, dt)
	auc_p = max(auc(y_plasma, dt), 1e-12)
	return float(auc_t / auc_p)


