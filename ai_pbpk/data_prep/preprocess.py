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
class PreparedData:
	X_train: pd.DataFrame
	X_val: pd.DataFrame
	X_test: pd.DataFrame
	y_train: np.ndarray
	y_val: np.ndarray
	y_test: np.ndarray
	preprocessor: ColumnTransformer
	# Indices of original dataframe rows for validation/testing
	train_index: np.ndarray | None = None
	val_index: np.ndarray | None = None
	test_index: np.ndarray | None = None

# Backward compatible alias
SplitOutput = PreparedData


def stratified_split(df: pd.DataFrame, target_col: str, stratify_by: str, test_size: float, val_size: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	# First split off test
	train_val, test = train_test_split(df, test_size=test_size, stratify=df[stratify_by], random_state=seed)
	# Then split train/val
	val_ratio = val_size / (1.0 - test_size)
	train, val = train_test_split(train_val, test_size=val_ratio, stratify=train_val[stratify_by], random_state=seed)
	return train, val, test


def _make_onehot() -> OneHotEncoder:
	# sklearn >=1.2 uses sparse_output; older uses sparse
	try:
		return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
	except TypeError:
		return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
	numeric_features = [c for c in df.columns if df[c].dtype != "object" and c not in {"k_params"}]
	categorical_features = [c for c in df.columns if df[c].dtype == "object"]

	numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
	categorical_transformer = Pipeline(steps=[("onehot", _make_onehot())])

	preprocessor = ColumnTransformer(
		transformers=[
			("num", numeric_transformer, numeric_features),
			("cat", categorical_transformer, categorical_features),
		]
	)
	return preprocessor


def prepare_features(df: pd.DataFrame, target_col: str, artifacts_dir: Path, seed: int = 42) -> PreparedData:
	# Non-stratified 70/15/15 split
	train_df, rest_df = train_test_split(df, test_size=0.30, random_state=seed)
	val_df, test_df = train_test_split(rest_df, test_size=0.50, random_state=seed)

	preprocessor = build_preprocessor(train_df.drop(columns=[target_col]))
	X_train = preprocessor.fit_transform(train_df.drop(columns=[target_col]))
	X_val = preprocessor.transform(val_df.drop(columns=[target_col]))
	X_test = preprocessor.transform(test_df.drop(columns=[target_col]))

	y_train = train_df[target_col].to_numpy()
	y_val = val_df[target_col].to_numpy()
	y_test = test_df[target_col].to_numpy()

	artifacts_dir.mkdir(parents=True, exist_ok=True)
	joblib.dump(preprocessor, artifacts_dir / "preprocessor.joblib")

	return PreparedData(
		X_train=X_train, X_val=X_val, X_test=X_test,
		y_train=y_train, y_val=y_val, y_test=y_test,
		preprocessor=preprocessor,
		train_index=getattr(train_df, 'index', None).to_numpy() if hasattr(train_df, 'index') else None,
		val_index=getattr(val_df, 'index', None).to_numpy() if hasattr(val_df, 'index') else None,
		test_index=getattr(test_df, 'index', None).to_numpy() if hasattr(test_df, 'index') else None,
	)


def prepare_features_stratified(df: pd.DataFrame, target_col: str, artifacts_dir: Path, seed: int = 42) -> PreparedData:
	# Stratified 70/15/15 by core_material
	train_df, rest_df = train_test_split(df, test_size=0.30, stratify=df["core_material"], random_state=seed)
	val_df, test_df = train_test_split(rest_df, test_size=0.50, stratify=rest_df["core_material"], random_state=seed)

	preprocessor = build_preprocessor(train_df.drop(columns=[target_col]))
	X_train = preprocessor.fit_transform(train_df.drop(columns=[target_col]))
	X_val = preprocessor.transform(val_df.drop(columns=[target_col]))
	X_test = preprocessor.transform(test_df.drop(columns=[target_col]))

	y_train = train_df[target_col].to_numpy()
	y_val = val_df[target_col].to_numpy()
	y_test = test_df[target_col].to_numpy()

	artifacts_dir.mkdir(parents=True, exist_ok=True)
	joblib.dump(preprocessor, artifacts_dir / "preprocessor.joblib")

	return PreparedData(
		X_train=X_train, X_val=X_val, X_test=X_test,
		y_train=y_train, y_val=y_val, y_test=y_test,
		preprocessor=preprocessor,
		train_index=getattr(train_df, 'index', None).to_numpy() if hasattr(train_df, 'index') else None,
		val_index=getattr(val_df, 'index', None).to_numpy() if hasattr(val_df, 'index') else None,
		test_index=getattr(test_df, 'index', None).to_numpy() if hasattr(test_df, 'index') else None,
	)


def prepare_from_parquet(features_path: Path, artifacts_dir: Path, seed: int) -> PreparedData:
	df = pd.read_parquet(features_path)
	assert "k_params" in df.columns, "Expected 'k_params' target in features parquet"
	# Production path uses stratified splits
	return prepare_features_stratified(df, target_col="k_params", artifacts_dir=artifacts_dir, seed=seed)


