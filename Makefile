SHELL := /bin/sh

.PHONY: setup test run-e2 report

setup:
	poetry install

test:
	poetry run pytest -q

run-e2:
	poetry run python scripts/run_experiment.py --exp e2 --config ai_pbpk/config/experiments/e2.yaml

report:
	poetry run python -m ai_pbpk.report.build_report


