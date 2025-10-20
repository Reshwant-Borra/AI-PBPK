from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from .theme import apply_theme


def plot_pareto(points: np.ndarray, out_path: Path, title: str):
	apply_theme()
	fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.2))
	ax.scatter(points[:, 0], points[:, 1], c='C0', s=25, alpha=0.8)
	ax.set_xlabel("Tumor AUC (a.u.)")
	ax.set_ylabel("Plasma AUC (a.u.)")
	ax.set_title(title)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	fig.tight_layout()
	fig.savefig(out_path)
	plt.close(fig)


