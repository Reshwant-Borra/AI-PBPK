import numpy as np
import pandas as pd

from ai_pbpk.data_prep.preprocess import stratified_split


def distribution(series):
	counts = series.value_counts(normalize=True)
	return counts.to_dict()


def test_stratified_70_15_15_split():
	rng = np.random.default_rng(0)
	N = 200
	materials = rng.choice(["gold", "liposome", "silica"], size=N, p=[0.4, 0.4, 0.2])
	df = pd.DataFrame({
		"core_material": materials,
		"feature1": rng.normal(size=N),
		"k_params": rng.normal(loc=0.2, scale=0.05, size=N),
	})
	train, val, test = stratified_split(df, target_col="k_params", stratify_by="core_material", test_size=0.15, val_size=0.15, seed=42)

	N_train, N_val, N_test = len(train), len(val), len(test)
	assert abs(N_train / N - 0.70) <= 0.05
	assert abs(N_val / N - 0.15) <= 0.05
	assert abs(N_test / N - 0.15) <= 0.05

	d_all = distribution(df["core_material"]) 
	d_train = distribution(train["core_material"]) 
	d_val = distribution(val["core_material"]) 
	d_test = distribution(test["core_material"]) 
	for k in d_all.keys():
		for dist in (d_train, d_val, d_test):
			assert abs(dist.get(k, 0.0) - d_all[k]) <= 0.02


