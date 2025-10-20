from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np

from ..pbpk.model import PBPKParams
from ..pbpk.solvers import simulate
from ..mpc.controller import MPCController, MPCConfig
from ..experiments.baselines import bolus_profile, infusion_profile
from ..eval.metrics import percent_injected_dose, tumor_plasma_auc_ratio, auc
from ..viz.dashboards import plot_timeseries, plot_dose
from ..data_prep.loader import build_processed_dataset
from ..ai_params.model import train_from_parquet
import yaml
import joblib
import pandas as pd
from loguru import logger
import mlflow


def _default_params(seed: int = 42) -> PBPKParams:
	rng = np.random.default_rng(seed)
	V = np.abs(rng.normal(loc=1.0, scale=0.2, size=8)) + 0.5
	k = np.abs(rng.normal(loc=0.2, scale=0.05, size=8)) + 0.05
	return PBPKParams(k=k, V=V)


def _ensure_processed_data() -> Tuple[Path, Path]:
	data_cfg = yaml.safe_load(open("ai_pbpk/config/data.yaml", "r", encoding="utf-8"))
	raw_excel = data_cfg["raw_excel_path"]
	long_p = Path(data_cfg["processed_long"])
	feat_p = Path(data_cfg["processed_features"])
	if not long_p.exists() or not feat_p.exists():
		_, _ = build_processed_dataset(raw_excel, out_dir=str(Path(long_p).parent))
	return long_p, feat_p


def _train_ai_if_needed(feat_p: Path) -> Path:
	models_dir = Path("models/ai_params")
	models_dir.mkdir(parents=True, exist_ok=True)
	model_json = models_dir / "xgb_model.json"
	preproc_p = models_dir / "preprocessor.joblib"
	if model_json.exists() and preproc_p.exists():
		return models_dir
	# Use base config for params
	base_cfg = yaml.safe_load(open("ai_pbpk/config/base.yaml", "r", encoding="utf-8"))
	params = base_cfg["ml"]["params"]
	res = train_from_parquet(feat_p, artifacts_dir=models_dir, params=params)
	# Save an additional joblib pickle for convenience
	try:
		model = joblib.load(models_dir / "xgb_model.json")
	except Exception:
		# Save the XGBRegressor via joblib by reloading and dumping
		from xgboost import XGBRegressor
		m = XGBRegressor()
		m.load_model(models_dir / "xgb_model.json")
		joblib.dump(m, models_dir / "model.pkl")
	return models_dir


def _predict_k_for_np(models_dir: Path, feat_row: pd.Series) -> float:
	preproc = joblib.load(models_dir / "preprocessor.joblib")
	from xgboost import XGBRegressor
	model = XGBRegressor()
	model.load_model(models_dir / "xgb_model.json")
	X = preproc.transform(pd.DataFrame([feat_row.drop(labels=[c for c in ["k_params"] if c in feat_row.index])]))
	k_pred = float(model.predict(X)[0])
	return k_pred


def _simulate_with_profile(params: PBPKParams, t_end: float, dt: float, u_fn) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	y0 = np.zeros(8)
	res = simulate(params, y0=y0, t_end=t_end, dt=dt, u_fn=u_fn)
	t = res.t
	y = res.y
	u = np.array([u_fn(ti) for ti in t])
	return t, y, u


def _simulate_mpc(params: PBPKParams, mpc_cfg: MPCConfig, t_end: float, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	controller = MPCController(params, mpc_cfg)
	Ts = mpc_cfg.Ts
	steps = int(np.ceil(t_end / dt))
	y = np.zeros((steps, 8))
	t = np.arange(steps) * dt
	u = np.zeros(steps)
	state = np.zeros(8)
	for i in range(steps):
		if i % int(max(Ts / dt, 1)) == 0:
			u_rate = controller.optimize_step(state)
		u[i] = u_rate
		# integrate single step using constant dose in [t, t+dt]
		u_fn = lambda _t: u_rate
		res = simulate(params, y0=state, t_end=dt, dt=dt, u_fn=u_fn)
		state = res.y[-1].copy()
		y[i] = state
	return t, y, u


def run_e2(base_cfg: dict, exp_cfg: dict, artifacts_dir: Path) -> Dict[str, float]:
	paths = base_cfg["paths"]
	fig_dir = Path(paths["figures_dir"]).resolve()
	fig_dir.mkdir(parents=True, exist_ok=True)

	sim_cfg = exp_cfg.get("sim", {})
	t_end = float(sim_cfg.get("t_end", base_cfg["sim"]["t_end"]))
	dt = float(sim_cfg.get("dt", base_cfg["sim"]["dt"]))

	# Real data integration
	long_p, feat_p = _ensure_processed_data()
	models_dir = _train_ai_if_needed(feat_p)
	long_df = pd.read_parquet(long_p)
	features = pd.read_parquet(feat_p)
	# Choose canonical nanoparticle with most observations if not specified
	candidate_np: Optional[str] = exp_cfg.get("nanoparticle_id") if isinstance(exp_cfg, dict) else None
	if not candidate_np:
		counts = long_df.groupby("nanoparticle_id").size().sort_values(ascending=False)
		candidate_np = str(counts.index[0])
	feat_row = features[features["nanoparticle_id"] == candidate_np].iloc[0]
	k_pred = _predict_k_for_np(models_dir, feat_row)
	V = np.ones(8)
	k_vec = np.ones(8) * max(0.01, k_pred)
	params = PBPKParams(k=k_vec, V=V)
	dose_total = float(exp_cfg.get("mpc", {}).get("dose_total", base_cfg["mpc"]["dose_total"]))

    # Baselines
	tu_bolus = lambda t: bolus_profile(t, dose_total=dose_total, dt=dt)
	tu_infusion = lambda t: infusion_profile(t, dose_total=dose_total, t_end=t_end)
	t_bolus, y_bolus, u_b = _simulate_with_profile(params, t_end, dt, tu_bolus)
	t_inf, y_inf, u_i = _simulate_with_profile(params, t_end, dt, tu_infusion)

	# MPC
	mpc_cfg_raw = exp_cfg.get("mpc", {})
	mpc_cfg = MPCConfig(
		Np=int(mpc_cfg_raw.get("Np", base_cfg["mpc"]["Np"])),
		Nc=int(mpc_cfg_raw.get("Nc", base_cfg["mpc"]["Nc"])),
		Ts=float(mpc_cfg_raw.get("Ts", base_cfg["mpc"]["Ts"])),
		w1=float(mpc_cfg_raw.get("w1", base_cfg["mpc"]["w1"])),
		w2=float(mpc_cfg_raw.get("w2", base_cfg["mpc"]["w2"])),
		u_max=float(mpc_cfg_raw.get("u_max", base_cfg["mpc"]["u_max"])),
		dose_total=float(mpc_cfg_raw.get("dose_total", base_cfg["mpc"]["dose_total"]))
	)
	t_mpc, y_mpc, u_m = _simulate_mpc(params, mpc_cfg, t_end, dt)

	# Metrics
	V_tumor = float(params.V[-1])
	pid_b = percent_injected_dose(y_bolus[:, -1], V_tumor, u_b, dt)
	pid_i = percent_injected_dose(y_inf[:, -1], V_tumor, u_i, dt)
	pid_m = percent_injected_dose(y_mpc[:, -1], V_tumor, u_m, dt)
	ratio_b = tumor_plasma_auc_ratio(y_bolus[:, -1], y_bolus[:, 0], dt)
	ratio_i = tumor_plasma_auc_ratio(y_inf[:, -1], y_inf[:, 0], dt)
	ratio_m = tumor_plasma_auc_ratio(y_mpc[:, -1], y_mpc[:, 0], dt)

    # Figures G1/G2
	plot_timeseries(t_mpc, {
		"Plasma - Bolus": y_bolus[:, 0],
		"Plasma - Inf": y_inf[:, 0],
		"Plasma - MPC": y_mpc[:, 0],
		"Tumor - Bolus": y_bolus[:, -1],
		"Tumor - Inf": y_inf[:, -1],
		"Tumor - MPC": y_mpc[:, -1],
	}, Path("reports/figures/G1_e2_timeseries.png"), title="E2 Concentrations")

    plot_dose(t_mpc, {
		"Bolus": u_b,
		"Infusion": u_i,
		"MPC": u_m,
	}, Path("reports/figures/G2_e2_dose.png"), title="E2 Dose Profiles")

    metrics = {
		"percent_id": {"bolus": pid_b, "infusion": pid_i, "mpc": pid_m},
		"tumor_plasma_auc_ratio": {"bolus": ratio_b, "infusion": ratio_i, "mpc": ratio_m},
    }

    # MLflow logging
    try:
        mlflow.set_tracking_uri(base_cfg["logging"]["mlflow_tracking_uri"])
        mlflow.set_experiment("AI-PBPK-E2")
        with mlflow.start_run(run_name="E2"):
            mlflow.log_params({
                "t_end": t_end, "dt": dt, "dose_total": dose_total,
                "Np": mpc_cfg.Np, "Nc": mpc_cfg.Nc, "Ts": mpc_cfg.Ts,
                "w1": mpc_cfg.w1, "w2": mpc_cfg.w2, "u_max": mpc_cfg.u_max
            })
            mlflow.log_metrics({
                "percent_id_mpc": pid_m,
                "percent_id_infusion": pid_i,
                "percent_id_bolus": pid_b,
                "ratio_mpc": ratio_m,
                "ratio_infusion": ratio_i,
                "ratio_bolus": ratio_b,
            })
            mlflow.log_artifact("reports/figures/G1_e2_timeseries.png")
            mlflow.log_artifact("reports/figures/G2_e2_dose.png")
    except Exception as e:
        logger.warning(f"MLflow logging skipped: {e}")

    return metrics


def run_e3(base_cfg: dict, exp_cfg: dict, artifacts_dir: Path) -> Dict[str, float]:
	N = int(exp_cfg.get("cohort", {}).get("N", 50))
	sim_cfg = base_cfg["sim"]
	t_end = float(sim_cfg["t_end"])  # reuse defaults
	dt = float(sim_cfg["dt"])
	dose_total = float(base_cfg["mpc"]["dose_total"])

	metrics_mpc = []
	metrics_inf = []
	for i in range(N):
		params = _default_params(seed=base_cfg.get("seed", 42) + i)
		# infusion baseline
		tu_infusion = lambda t: infusion_profile(t, dose_total=dose_total, t_end=t_end)
		_, y_inf, u_i = _simulate_with_profile(params, t_end, dt, tu_infusion)
		# MPC
		mpc_cfg = MPCConfig(Np=8, Nc=4, Ts=0.5, w1=1.0, w2=0.2, u_max=1.0, dose_total=dose_total)
		_, y_mpc, u_m = _simulate_mpc(params, mpc_cfg, t_end, dt)
		metrics_inf.append(tumor_plasma_auc_ratio(y_inf[:, -1], y_inf[:, 0], dt))
		metrics_mpc.append(tumor_plasma_auc_ratio(y_mpc[:, -1], y_mpc[:, 0], dt))

	from ..viz.cohorts import plot_cohort_box
	from ..eval.stats import paired_test

	plot_cohort_box({"Inf": metrics_inf, "MPC": metrics_mpc}, Path("reports/figures/G3_cohort.png"), title="E3 Cohort AUC Ratios")
	arr_inf = np.asarray(metrics_inf)
	arr_mpc = np.asarray(metrics_mpc)
	test = paired_test(arr_mpc, arr_inf)
	return {"p_value": test.p_value, "method": test.method}


def run_e4(base_cfg: dict, exp_cfg: dict, artifacts_dir: Path) -> Dict[str, float]:
    sim_cfg = base_cfg["sim"]
    t_end = float(sim_cfg["t_end"])
    dt = float(sim_cfg["dt"])
    w1_list = exp_cfg.get("sweep", {}).get("w1", [0.5, 1.0, 2.0])
    w2_list = exp_cfg.get("sweep", {}).get("w2", [0.1, 0.2])
    dose_total = float(base_cfg["mpc"]["dose_total"])

    params = _default_params(seed=base_cfg.get("seed", 42))
    points = []  # (tumor_auc, plasma_auc)
    for w1 in w1_list:
        for w2 in w2_list:
            mpc_cfg = MPCConfig(Np=8, Nc=4, Ts=0.5, w1=float(w1), w2=float(w2), u_max=1.0, dose_total=dose_total)
            t_mpc, y_mpc, u_m = _simulate_mpc(params, mpc_cfg, t_end, dt)
            tumor = auc(y_mpc[:, -1], dt)
            plasma = auc(y_mpc[:, 0], dt)
            points.append([tumor, plasma])

    import numpy as np
    pts = np.asarray(points, dtype=float)
    from ..viz.pareto import plot_pareto
    plot_pareto(pts, Path("reports/figures/G4_pareto.png"), title="E4 Pareto Frontier")
    return {"n_points": int(len(points))}


def run_experiment(exp_id: str, base_cfg: dict, exp_cfg: dict, artifacts_dir: Path) -> Dict[str, float]:
	if exp_id.lower() == "e2":
		return run_e2(base_cfg, exp_cfg, artifacts_dir)
	if exp_id.lower() == "e3":
		return run_e3(base_cfg, exp_cfg, artifacts_dir)
    if exp_id.lower() == "e4":
        return run_e4(base_cfg, exp_cfg, artifacts_dir)
	# Minimal stubs for other experiments
	return {"status": "not_implemented"}


