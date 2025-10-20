from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
	rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
	r2 = float(r2_score(y_true, y_pred))
	return {"rmse": rmse, "r2": r2}


