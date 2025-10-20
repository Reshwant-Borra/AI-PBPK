from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from scipy import stats


@dataclass
class StatTestResult:
	p_value: float
	ci_low: float
	ci_high: float
	effect_size: float
	method: str


def paired_test(a: np.ndarray, b: np.ndarray, alpha: float = 0.05) -> StatTestResult:
	delta = a - b
	method = "wilcoxon"
	try:
		if len(delta) >= 3:
			w, pnorm = stats.shapiro(delta)
			if pnorm >= 0.05:
				method = "ttest_rel"
				t_stat, p = stats.ttest_rel(a, b)
				mean = float(np.mean(delta))
				sd = float(np.std(delta, ddof=1))
				n = len(delta)
				se = sd / np.sqrt(n)
				tcrit = stats.t.ppf(1 - alpha / 2, df=n - 1)
				ci = (mean - tcrit * se, mean + tcrit * se)
				effect = mean / (sd + 1e-12)
				return StatTestResult(p_value=float(p), ci_low=float(ci[0]), ci_high=float(ci[1]), effect_size=float(effect), method=method)
		# Fallback to Wilcoxon
		wstat, p = stats.wilcoxon(a, b, zero_method='wilcox', correction=True)
		q1, q3 = np.percentile(delta, [2.5, 97.5])
		effect = float(np.median(delta)) / (np.std(delta, ddof=1) + 1e-12)
		return StatTestResult(p_value=float(p), ci_low=float(q1), ci_high=float(q3), effect_size=float(effect), method=method)
	except Exception:
		# Very small samples
		return StatTestResult(p_value=1.0, ci_low=0.0, ci_high=0.0, effect_size=0.0, method="na")


