from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BaselineConfig:
	t_end: float
	dt: float
	dose_total: float


def bolus_profile(t: float, dose_total: float, dt: float) -> float:
	# Instantaneous bolus approximated over first step
	return dose_total / dt if t < dt else 0.0


def infusion_profile(t: float, dose_total: float, t_end: float) -> float:
	return dose_total / t_end if 0.0 <= t <= t_end else 0.0


