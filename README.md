# AI-PBPK-MPC

Research-grade, end-to-end pipeline for AI-parameterized PBPK simulation with MPC and reporting.

## Quickstart

1) Install Poetry and dependencies

```bash
poetry install
```

2) Drop your Nano-Tumor Excel/CSV(s) into `data/raw/`.

3) Run an experiment (E2 canonical case):

```bash
python scripts/run_experiment.py --exp e2 --config ai_pbpk/config/experiments/e2.yaml
```

Artifacts are saved under `artifacts/<timestamp>/`. Figures under `reports/figures/` and `reports/summary.pdf` via:

```bash
python -m ai_pbpk.report.build_report
```

## Make targets

```bash
make setup   # poetry install
make test    # pytest -q
make run-e2  # run canonical E2
```

## IPOPT/CasADi install notes

- CasADi Python wheels are installed via Poetry (`casadi>=3.6`).
- IPOPT is used by CasADi as a native solver; on Windows/Linux, install a prebuilt IPOPT binary and ensure it is on PATH. See `https://coin-or.github.io/Ipopt/INSTALL.html` for details. If IPOPT is unavailable, the MPC uses a SciPy SLSQP fallback.

## Project layout

See `ai_pbpk/` for modules: data prep, AI parameter predictor, PBPK ODE simulator, MPC, experiments, evaluation, visualization, and report.

## Reproducibility

- All runs log seeds, config, and git commit to MLflow (local `mlruns/`).
- Configurable via YAML in `ai_pbpk/config/`.


