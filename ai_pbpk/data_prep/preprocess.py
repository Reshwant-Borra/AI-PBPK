from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import pyarrow.parquet as pq
import pandas as pd


@dataclass
class SplitOutput:
	X_train: pd.DataFrame
	X_val: pd.DataFrame
	X_test: pd.DataFrame
	y_train: np.ndarray
	y_val: np.ndarray
	y_test: np.ndarray
	preprocessor: ColumnTransformer


def stratified_split(df: pd.DataFrame, target_col: str, stratify_by: str, test_size: float, val_size: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	# First split off test
	train_val, test = train_test_split(df, test_size=test_size, stratify=df[stratify_by], random_state=seed)
	# Then split train/val
	val_ratio = val_size / (1.0 - test_size)
	train, val = train_test_split(train_val, test_size=val_ratio, stratify=train_val[stratify_by], random_state=seed)
	return train, val, test


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
	numeric_features = [c for c in df.columns if df[c].dtype != "object" and c not in {"k_params"}]
	categorical_features = [c for c in df.columns if df[c].dtype == "object"]

	numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
	categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])

	preprocessor = ColumnTransformer(
		transformers=[
			("num", numeric_transformer, numeric_features),
			("cat", categorical_transformer, categorical_features),
		]
	)
	return preprocessor


def prepare_features(df: pd.DataFrame, target_col: str, artifacts_dir: Path, seed: int) -> SplitOutput:
	train, val, test = stratified_split(df, target_col, stratify_by="core_material", test_size=0.15, val_size=0.15, seed=seed)
	preprocessor = build_preprocessor(train.drop(columns=[target_col]))

	X_train = preprocessor.fit_transform(train.drop(columns=[target_col]))
	X_val = preprocessor.transform(val.drop(columns=[target_col]))
	X_test = preprocessor.transform(test.drop(columns=[target_col]))

	y_train = train[target_col].to_numpy()
	y_val = val[target_col].to_numpy()
	y_test = test[target_col].to_numpy()

	artifacts_dir.mkdir(parents=True, exist_ok=True)
	joblib.dump(preprocessor, artifacts_dir / "preprocessor.joblib")

	return SplitOutput(
		X_train=X_train, X_val=X_val, X_test=X_test,
		y_train=y_train, y_val=y_val, y_test=y_test,
		preprocessor=preprocessor,
	)


def prepare_from_parquet(features_path: Path, artifacts_dir: Path, seed: int) -> SplitOutput:
	df = pd.read_parquet(features_path)
	assert "k_params" in df.columns, "Expected 'k_params' target in features parquet"
	return prepare_features(df, target_col="k_params", artifacts_dir=artifacts_dir, seed=seed)


