import numpy as np
import pandas as pd

from ai_pbpk.data_prep.preprocess import prepare_features


def test_preprocessor_fit_on_train_only(tmp_path):
	# Train has categories A,B; Val/Test have an extra C
	rng = np.random.default_rng(0)
	df = pd.DataFrame({
		"core_material": ["A"] * 40 + ["B"] * 40 + ["C"] * 20,
		"x": rng.normal(size=100),
		"k_params": rng.normal(loc=0.2, scale=0.05, size=100),
	})
	out = prepare_features(df, target_col="k_params", artifacts_dir=tmp_path, seed=1)
	enc = out.preprocessor.named_transformers_["cat"].named_steps["onehot"]
	cats = set(enc.categories_[0])
	# Category C should not be in training categories with high probability
	assert "C" not in cats


