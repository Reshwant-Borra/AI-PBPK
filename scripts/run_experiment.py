import os
import sys
import json
from pathlib import Path
import typer
import yaml

app = typer.Typer(add_completion=False)


def load_yaml(path: Path):
	with open(path, "r", encoding="utf-8") as f:
		return yaml.safe_load(f)


@app.command()
def main(
	# Support both original and alternative flag names
	exp: str = typer.Option(None, help="Experiment id: e1..e5", param_declarations=["--exp", "--exp-id"]),
	config: Path = typer.Option(Path("ai_pbpk/config/experiments") / "e2.yaml", help="Path to experiment YAML config", param_declarations=["--config", "--config-path"]),
):
	base_cfg = load_yaml(Path("ai_pbpk/config/base.yaml"))
	exp_cfg = load_yaml(config)

	# Lazy import to speed CLI
	from ai_pbpk.experiments.runner import run_experiment

	from datetime import datetime
	stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	artifacts_root = Path(base_cfg["paths"]["artifacts_dir"]) / exp / stamp
	artifacts_root.mkdir(parents=True, exist_ok=True)

	metrics = run_experiment(exp_id=exp, base_cfg=base_cfg, exp_cfg=exp_cfg, artifacts_dir=artifacts_root)
	metrics_path = artifacts_root / "metrics.json"
	with open(metrics_path, "w", encoding="utf-8") as f:
		json.dump(metrics, f, indent=2)
	print(str(metrics_path))


if __name__ == "__main__":
	app()


