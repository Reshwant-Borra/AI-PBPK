from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .theme import apply_theme


def plot_timeseries(t: np.ndarray, series_dict: dict, out_path: Path, title: str):
	apply_theme()
	fig, ax = plt.subplots(1, 1, figsize=(6, 3.5))
	for label, arr in series_dict.items():
		ax.plot(t, arr, label=label)
	ax.set_xlabel("Time (h)")
	ax.set_ylabel("Concentration (a.u.)")
	ax.set_title(title)
	ax.legend()
	out_path.parent.mkdir(parents=True, exist_ok=True)
	fig.tight_layout()
	fig.savefig(out_path)
	plt.close(fig)


def plot_dose(t: np.ndarray, u_dict: dict, out_path: Path, title: str):
	apply_theme()
	fig, ax = plt.subplots(1, 1, figsize=(6, 2.8))
	for label, u in u_dict.items():
		ax.step(t, u, where='post', label=label)
	ax.set_xlabel("Time (h)")
	ax.set_ylabel("Dose rate (a.u.)")
	ax.set_title(title)
	ax.legend()
	out_path.parent.mkdir(parents=True, exist_ok=True)
	fig.tight_layout()
	fig.savefig(out_path)
	plt.close(fig)


