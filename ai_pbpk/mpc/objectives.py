from __future__ import annotations

import numpy as np


def objective_trajectory(y_tumor: np.ndarray, y_plasma: np.ndarray, u: np.ndarray, w1: float, w2: float, dt: float) -> float:
	# Maximize tumor exposure, penalize plasma AUC; here we return negative for minimization
	tumor_auc = float(np.trapz(y_tumor, dx=dt))
	plasma_auc = float(np.trapz(y_plasma, dx=dt))
	return -(w1 * tumor_auc - w2 * plasma_auc)


