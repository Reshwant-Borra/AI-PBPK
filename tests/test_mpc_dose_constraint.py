import numpy as np

from ai_pbpk.pbpk.model import PBPKParams
from ai_pbpk.mpc.controller import MPCController, MPCConfig


def test_mpc_respects_total_dose():
	params = PBPKParams(k=np.ones(8) * 0.1, V=np.ones(8))
	mpc_cfg = MPCConfig(Np=6, Nc=3, Ts=0.5, w1=1.0, w2=0.1, u_max=2.0, dose_total=1.0)
	controller = MPCController(params, mpc_cfg)
	state = np.zeros(8)
	acc_dose = 0.0
	for _ in range(10):
		u = controller.optimize_step(state)
		acc_dose += u * mpc_cfg.Ts
	assert acc_dose <= mpc_cfg.dose_total + 1e-6


