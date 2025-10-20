from pathlib import Path


def test_report_directories_exist():
	Path('reports/figures').mkdir(parents=True, exist_ok=True)
	assert Path('reports/figures').exists()


