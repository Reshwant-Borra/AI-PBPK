from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from .theme import apply_theme


def plot_cohort_box(values_dict: dict, out_path: Path, title: str):
	apply_theme()
	labels = list(values_dict.keys())
	data = [np.asarray(values_dict[k]) for k in labels]
	fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.2))
	ax.boxplot(data, labels=labels)
	ax.set_title(title)
	ax.set_ylabel("Metric value")
	out_path.parent.mkdir(parents=True, exist_ok=True)
	fig.tight_layout()
	fig.savefig(out_path)
	plt.close(fig)


