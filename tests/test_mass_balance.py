import numpy as np

from ai_pbpk.pbpk.model import PBPKParams
from ai_pbpk.pbpk.solvers import simulate
from ai_pbpk.pbpk.sanity import check_mass_balance


def test_mass_balance_simple():
	params = PBPKParams(k=np.ones(8) * 0.1, V=np.ones(8))
	dt = 0.1
	t_end = 2.0
	u_fn = lambda t: 1.0
	res = simulate(params, y0=np.zeros(8), t_end=t_end, dt=dt, u_fn=u_fn)
	u_series = np.array([u_fn(t) for t in res.t])
	assert check_mass_balance(res.y, u_series, dt, tol=1e-4)


