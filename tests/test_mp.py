# coding: utf-8
#
# This code is part of dqmc.
#
# Copyright (c) 2022, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

from dqmc import Parameters
from dqmc.mp import map_params, run_dqmc_parallel


def test_map_params():
    p_default = Parameters(5)

    # Map interactions
    values = [1.0, 2.0, 3.0]
    params = map_params(p_default, u=values)
    for x, p in zip(values, params):
        assert p.u == x

    # Map betas
    values = [1.0, 2.0, 3.0]
    params = map_params(p_default, beta=values)
    for x, p in zip(values, params):
        assert p.beta == x
        assert p.dt == x / p.num_timesteps

    # Map temps
    values = [1.0, 0.5, 0.25]
    params = map_params(p_default, temp=values)
    for x, p in zip(values, params):
        assert p.beta == 1 / x
        assert p.dt == 1 / (x * p.num_timesteps)


def test_run_dqmc_parallel():
    p_default = Parameters(5, num_sampl=512)
    params = map_params(p_default, u=[1.0, 2.0, 3.0])
    results = run_dqmc_parallel(params, progress=False)
    assert len(results) == 3
