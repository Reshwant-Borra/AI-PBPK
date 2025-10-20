from __future__ import annotations

from pathlib import Path
from datetime import datetime
import os

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def build_report():
	root = Path('.')
	fig_dir = Path('reports/figures')
	fig_dir.mkdir(parents=True, exist_ok=True)
	pdf_path = Path('reports/summary.pdf')
	with PdfPages(pdf_path) as pdf:
		# Title page
		fig, ax = plt.subplots(figsize=(8.25, 11.0))
		ax.axis('off')
		ax.text(0.5, 0.8, 'AI-PBPK-MPC Summary', ha='center', va='center', fontsize=20)
		ax.text(0.5, 0.75, datetime.now().strftime('%Y-%m-%d %H:%M'), ha='center')
		ax.text(0.5, 0.70, 'Data: Nano-Tumor Excel (see ai_pbpk/config/data.yaml)', ha='center')
		pdf.savefig(fig)
		plt.close(fig)

		# Append known figures if present
		for name in [
			'G1_e2_timeseries.png',
			'G2_e2_dose.png',
			'G3_cohort.png',
			'G4_pareto.png',
		]:
			p = fig_dir / name
			if p.exists():
				fig = plt.figure(figsize=(8.25, 5.5))
				img = plt.imread(p)
				plt.imshow(img)
				plt.axis('off')
				pdf.savefig(fig)
				plt.close(fig)
	print(str(pdf_path))


if __name__ == '__main__':
	build_report()


