from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
from xgboost import XGBRegressor
import pandas as pd

from .metrics import regression_metrics


@dataclass
class TrainResult:
	metrics: Dict[str, float]
	model_path: Path


def train_xgb(X_train, y_train, X_val, y_val, artifacts_dir: Path, params: Dict[str, Any]) -> TrainResult:
	model = XGBRegressor(**params, n_jobs=0, random_state=42)
	model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
	y_val_pred = model.predict(X_val)
	metrics = regression_metrics(y_val, y_val_pred)
	artifacts_dir.mkdir(parents=True, exist_ok=True)
	model_path = artifacts_dir / "xgb_model.json"
	model.save_model(model_path)
	# feature importance
	fi = getattr(model, "feature_importances_", None)
	if fi is not None:
		joblib.dump(fi, artifacts_dir / "feature_importances.joblib")
	return TrainResult(metrics=metrics, model_path=model_path)


def predict(model_path: Path, X) -> np.ndarray:
	model = XGBRegressor()
	model.load_model(model_path)
	return model.predict(X)


def train_from_parquet(features_path: Path, artifacts_dir: Path, params: Dict[str, Any]) -> TrainResult:
	from ..data_prep.preprocess import prepare_from_parquet
	splits = prepare_from_parquet(features_path, artifacts_dir=artifacts_dir, seed=42)
	res = train_xgb(splits.X_train, splits.y_train, splits.X_val, splits.y_val, artifacts_dir=artifacts_dir, params=params)
	# Save simple CSV metrics
	import csv
	metrics_csv = artifacts_dir / "reports" / "tables" / "ai_metrics.csv"
	metrics_csv.parent.mkdir(parents=True, exist_ok=True)
	with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(["metric", "value"])
		for k, v in res.metrics.items():
			writer.writerow([k, v])
	# Feature importance bar plot
	try:
		model = XGBRegressor()
		model.load_model(res.model_path)
		fi = getattr(model, "feature_importances_", None)
		if fi is not None:
			import matplotlib.pyplot as plt
			fig_path = artifacts_dir / "reports" / "figures" / "ai_feature_importance.png"
			fig_path.parent.mkdir(parents=True, exist_ok=True)
			plt.figure(figsize=(6,3))
			plt.bar(range(len(fi)), fi)
			plt.tight_layout()
			plt.savefig(fig_path)
			plt.close()
	except Exception:
		pass
	return res


