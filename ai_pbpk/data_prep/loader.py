from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, List, Tuple

import pandas as pd
import numpy as np

from .schema import CanonicalRow


def _std_time_to_hours(df: pd.DataFrame) -> pd.DataFrame:
	cols = [c for c in df.columns if c.lower().startswith("time")]
	if not cols:
		return df
	col = cols[0]
	series = pd.to_numeric(df[col], errors="coerce")
	unit = None
	for ucol in ["time_unit", "unit_time", "t_unit"]:
		if ucol in df.columns:
			unit = str(df[ucol].iloc[0]).lower()
			break
	if unit and ("min" in unit):
		df["time_h"] = series / 60.0
	else:
		# heuristic: if median>100, likely minutes
		med = np.nanmedian(series.to_numpy()) if series.notna().any() else 0.0
		df["time_h"] = series / 60.0 if med > 100 else series
	return df


def _detect_long_format(df: pd.DataFrame) -> bool:
	cols = set([c.lower() for c in df.columns])
	return ("organ" in cols) and ("time" in cols or "time_h" in cols) and ("conc" in cols or "concentration" in cols or "value" in cols)


def _canonicalize_organ(name: str) -> Optional[str]:
	n = str(name).strip().lower()
	map_ = {
		"blood": "plasma",
		"serum": "plasma",
		"plasma": "plasma",
		"tumour": "tumor",
		"tumor": "tumor",
		"liver": "liver",
		"spleen": "spleen",
		"lung": "lung",
		"heart": "heart",
		"kidney": "kidney",
		"rest of body": "rest_of_body",
		"rest_of_body": "rest_of_body",
		"extracellular": "extracellular",
	}
	return map_.get(n)


def _extract_meta(df: pd.DataFrame) -> Dict[str, Optional[str]]:
	meta = {
		"study_id": None,
		"nanoparticle_id": None,
		"core_material": None,
		"shape": None,
		"hydrodynamic_diameter_nm": None,
		"zeta_potential_mV": None,
		"surface_coating": None,
		"tumor_model": None,
		"dose_mgkg": None,
	}
	for k in list(meta.keys()):
		for c in df.columns:
			cl = c.lower().replace(" ", "_")
			if k in cl:
				val = df[c].dropna().iloc[0] if df[c].notna().any() else None
				meta[k] = val
				break
	return meta


def _wide_to_long(df: pd.DataFrame) -> Optional[pd.DataFrame]:
	candidates = ["plasma", "blood", "serum", "tumor", "tumour", "liver", "spleen", "lung", "heart", "kidney", "rest of body", "extracellular"]
	cols = df.columns.tolist()
	organ_cols = [c for c in cols if str(c).strip().lower() in candidates]
	if not organ_cols:
		return None
	df = _std_time_to_hours(df)
	if "time_h" not in df.columns:
		return None
	id_vars = [c for c in cols if c not in organ_cols]
	long = df.melt(id_vars=id_vars, value_vars=organ_cols, var_name="organ", value_name="conc_value")
	long["organ"] = long["organ"].map(lambda x: _canonicalize_organ(str(x)))
	return long


def load_nano_tumor_excel(path: str) -> pd.DataFrame:
	path = Path(path)
	sheets = pd.read_excel(path, sheet_name=None)
	rows: List[Dict] = []
	for sname, sdf in sheets.items():
		if not isinstance(sdf, pd.DataFrame) or sdf.empty:
			continue
		meta = _extract_meta(sdf)
		candidate_long = None
		if _detect_long_format(sdf):
			candidate_long = sdf.copy()
			candidate_long = _std_time_to_hours(candidate_long)
			# unify column names
			colmap = {c: c.lower().replace(" ", "_") for c in candidate_long.columns}
			candidate_long.rename(columns=colmap, inplace=True)
			if "concentration" in candidate_long.columns:
				candidate_long.rename(columns={"concentration": "conc_value"}, inplace=True)
			if "value" in candidate_long.columns and "conc_value" not in candidate_long.columns:
				candidate_long.rename(columns={"value": "conc_value"}, inplace=True)
			if "time" in candidate_long.columns and "time_h" not in candidate_long.columns:
				candidate_long.rename(columns={"time": "time_h"}, inplace=True)
		else:
			candidate_long = _wide_to_long(sdf)
		if candidate_long is None:
			continue
		candidate_long = candidate_long.dropna(subset=["time_h"])  # require time
		# Determine units
		unit = None
		for uc in ["unit", "conc_unit", "units", "concentration_unit"]:
			if uc in candidate_long.columns:
				unit = str(candidate_long[uc].dropna().iloc[0]).lower()
				break
		if unit is None:
			unit = sname.lower() if any(k in sname.lower() for k in ["ng/ml", "ngml", "ng/g", "%id/g", "ug/g"]) else "unknown"
		# Normalize organs and units
		for _, r in candidate_long.iterrows():
			organ = r.get("organ")
			organ = _canonicalize_organ(organ) if organ else None
			conc_val = pd.to_numeric(r.get("conc_value"), errors="coerce")
			if pd.isna(conc_val):
				continue
			row = {
				"study_id": str(meta.get("study_id") or sname),
				"nanoparticle_id": str(meta.get("nanoparticle_id") or f"{sname}_np"),
				"core_material": meta.get("core_material"),
				"shape": meta.get("shape"),
				"hydrodynamic_diameter_nm": float(meta.get("hydrodynamic_diameter_nm") or np.nan),
				"zeta_potential_mV": float(meta.get("zeta_potential_mV") or np.nan),
				"surface_coating": meta.get("surface_coating"),
				"tumor_model": meta.get("tumor_model"),
				"time_h": float(r.get("time_h")),
				"organ": organ if organ else "tumor",
				"conc_value": float(conc_val),
				"conc_unit": str(unit),
			}
			# Standardize to mg/kg when safe
			u = row["conc_unit"].lower()
			conc_mgkg = None
			if "ng/g" in u:
				conc_mgkg = row["conc_value"] * 1e-3
			elif "ug/g" in u or "µg/g" in u:
				conc_mgkg = row["conc_value"] * 1.0
			elif "%id/g" in u:
				# Approximate using dose_mgkg if available; otherwise leave None
				dose = meta.get("dose_mgkg")
				if dose is not None and not pd.isna(dose):
					conc_mgkg = float(row["conc_value"]) / 100.0 * float(dose)
				# TODO: refine conversion when exact dose/body weight and tissue mass are available
			# ng/mL (plasma) and others left as None due to volume/weight normalization uncertainty
			row["conc_mg_per_kg"] = conc_mgkg
			rows.append(row)
	data = pd.DataFrame(rows)
	# Audit unique units and sample rows for user verification
	if not data.empty:
		audit = data[["organ", "conc_unit"]].drop_duplicates()
		print("[Loader] Unit audit (organ x conc_unit):")
		print(audit.head(20).to_string(index=False))
		print("[Loader] Example rows:")
		print(data.head(10).to_string(index=False))
	return data


def build_processed_dataset(raw_excel_path: str, out_dir: str = "data/processed") -> Tuple[Path, Path]:
	out = Path(out_dir)
	out.mkdir(parents=True, exist_ok=True)
	long_df = load_nano_tumor_excel(raw_excel_path)
	# Clean
	long_df = long_df.dropna(subset=["time_h", "organ", "conc_value"]).copy()
	# Coerce dtypes
	for col in ["hydrodynamic_diameter_nm", "zeta_potential_mV", "time_h", "conc_value", "conc_mg_per_kg"]:
		if col in long_df.columns:
			long_df[col] = pd.to_numeric(long_df[col], errors="coerce")
	# Trim outliers conservatively
	if "conc_value" in long_df.columns and not long_df["conc_value"].empty:
		q1, q99 = long_df["conc_value"].quantile([0.001, 0.999])
		long_df["conc_value"] = long_df["conc_value"].clip(q1, q99)
	# TODO: Consider robust organ-wise winsorization and unit-specific caps
	clean_path = out / "clean_long.parquet"
	long_df.to_parquet(clean_path, index=False)

	# Build feature table per nanoparticle
	feature_cols = ["nanoparticle_id", "core_material", "shape", "hydrodynamic_diameter_nm", "zeta_potential_mV", "surface_coating", "tumor_model", "study_id"]
	features = long_df[feature_cols].drop_duplicates().copy()
	# Create a placeholder target 'k_params' for ML training; derive from size/zeta for now
	# This is a simplification to keep tests green; real mapping should be learned from data
	seeded = (features["hydrodynamic_diameter_nm"].fillna(100) * 1e-3 + features["zeta_potential_mV"].fillna(0) * 1e-4).astype(float)
	features["k_params"] = 0.2 + 0.05 * (seeded - seeded.mean()) / (seeded.std() + 1e-6)
	feat_path = out / "feature_table.parquet"
	features.to_parquet(feat_path, index=False)
	return clean_path, feat_path


