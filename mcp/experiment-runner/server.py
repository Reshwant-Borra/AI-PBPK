from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import json
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from ai_pbpk.experiments.runner import run_experiment
from datetime import datetime
import yaml


class RunRequest(BaseModel):
	exp_id: str
	config_path: str


app = FastAPI()


@app.post("/run")
def run(req: RunRequest) -> Dict[str, Any]:
	base_cfg = yaml.safe_load(open("ai_pbpk/config/base.yaml", "r", encoding="utf-8"))
	exp_cfg = yaml.safe_load(open(req.config_path, "r", encoding="utf-8"))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifacts_dir = Path(base_cfg["paths"]["artifacts_dir"]) / req.exp_id / stamp
	artifacts_dir.mkdir(parents=True, exist_ok=True)
	metrics = run_experiment(req.exp_id, base_cfg, exp_cfg, artifacts_dir)
    with open(artifacts_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
	return {"status": "ok", "metrics": metrics}


if __name__ == "__main__":
	uvicorn.run(app, host="127.0.0.1", port=8765)


