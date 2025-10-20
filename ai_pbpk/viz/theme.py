from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns


def apply_theme():
	sns.set_theme(style="whitegrid")
	plt.rcParams.update({
		"figure.dpi": 120,
		"savefig.dpi": 120,
		"axes.titlesize": 12,
		"axes.labelsize": 11,
		"legend.fontsize": 9,
	})


