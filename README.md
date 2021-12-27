# DQMC

![GitHub tag (latest by date)](https://img.shields.io/github/v/tag/dylanljones/dqmc)
![GitHub license](https://img.shields.io/github/license/dylanljones/dqmc)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Determinant Quantum Monte Carlo simulations of the Hubbard model in Python.

| :warning: **WARNING**: This project is still under development and might contain errors or change significantly in the future! |
| --- |

## Installation

Install via `pip` from github:
```commandline
pip install git+https://github.com/dylanljones/dqmc.git@VERSION
```
or download/clone the package, navigate to the root directory and install via
````commandline
pip install -e <folder path>
````
or the `setup.py` script
````commandline
python setup.py install
````


## Quickstart

To run a simulation, run the `main.py` script with a configuration text file
as parameter, for example:
````commandline
python main.py examples/chain.txt
````

### Parameters

- `shape`
   The shape of the lattice model.
- `U`
   The interaction strength of the Hubbard model.
- `eps`
   The on-site energy of the Hubbard model.
- `t`
   The hopping energy of the Hubbard model.
- `mu`
   The chemical potential of the Hubbard model. Set to `U/2` for half filling.
- `dt` (optional)
   The imaginary time step size.
- `beta` (optional)
   The inverse temperature. Can be set instead of `dt`.
- `temp` (optional)
   The temperature. Can be set instead of `dt`.
- `L`
   The number of imaginary time slices
- `nequil`
   The number of warmup-sweeps
- `nsampl`
   The number of measurement-sweeps
- `nrecomp` (optional)
   The number of time slice wraps after which the Green's functions are recomputed.
   The default is `1`.
- `prodLen` (optional)
   The number of explicit matrix products used for the stabilized matrix product
   via ASvQRD. The default is `1`.

## Usage

### Initializing the Hubbard model

In order to run a simulation, a Hubbard model has to be constructed. This can be
done manually by initializing the included `HubbardModel`:
```python
import numpy as np
from dqmc import HubbardModel

# Initialize a square Hubbard model
vectors = np.eye(2)
model = HubbardModel(vectors, u=4., eps=0., hop=1., mu=0., beta=1/5)
# Add an atom to the unit cell
model.add_atom()
# Set nearest neighbor hoppings
model.add_connections(1)
# Build the lattice with periodic boundary conditions along both axis
model.build((5, 5), relative=True, periodic=(0, 1))
```
or by using the helper function `hubbard_hypercube` to construct a `d`-dimensional
hyper-rectangular Hubbard model with one atom in the unit cell and nearest neighbor
hopping:
```python
from dqmc import hubbard_hypercube

shape = (5, 5)
model = hubbard_hypercube(shape, u=4., eps=0., hop=1., mu=0., beta=1/5, periodic=True)
```
Setting `periodic=True` marks all axis as periodic.

### Running simulations and measuring observables

To run a Determinant Quantum Monte carlo simulation the `DQMC`-object can be used.
This is a wrapper of the main DQMC methods, which are contained in `dqmc/dqmc.py`
and use jit (just in time compilation) to improve performance:
```python
from dqmc import hubbard_hypercube, mfuncs, DQMC

shape = (5, 5)
num_timesteps = 100
warmup, measure = 300, 3000
model = hubbard_hypercube(shape, u=4., eps=0., hop=1., mu=0., beta=1/5, periodic=True)

dqmc = DQMC(model, num_timesteps, num_recomp=1, prod_len=1, seed=0)
results = dqmc.simulate(warmup, measure, callback=mfuncs.occupation)
```
The `simulate`-method measures the observables
- `n_up`
   The spin-up occupation `<n_↑>`.
- `n_dn`
   The spin-down occupation `<n_↓>`.
- `n_double`
   The double occupation `<n_↑ n_↓>`.
- `local_moment`
   The local moment `<n_↑> + <n_↓> - 2 <n_↑ n_↓>`.


Additionally, the `simulate`-method has a `callback` parameter for measuring observables, which
expects a method of the form
```python
def callback(gf_up, gf_dn):
    ...
    return result
```
The returned result must be a `np.ndarray` for ensuring correct averaging after the
measurement sweeps. If no callback is given the local Green's function `G_{ij}` is
measured by default. A collection of methods for measuring observables is contained
in the `mfuncs` module.

The above steps can be simplified by calling the `run_dqmc`-method:
```python
from dqmc import run_dqmc, mfuncs

shape = 10
u, eps, mu, hop = 4.0, 0.0, 0.0, 1.0
beta = 1.0
num_timesteps = 100
warmup, measure = 300, 3000

res = run_dqmc(shape, u, eps, hop, mu, beta, num_timesteps,
               warmup, measure, mfuncs.occupation)
```

### Multiprocessing

To run multiple DQMC simulations in parallel use the `ProcessPool`-object:
```python
from dqmc import ProcessPool, run_dqmc, mfuncs

shape = 10
inters = [2, 3, 4, 5, 6, 7, 8, 9]
eps, mu, hop = 0.0, 0.0, 1.0
beta = 1.0
num_timesteps = 100
warmup, measure = 300, 3000

params = (
  shape, inters, eps, hop, mu, beta, num_timesteps,
  warmup, measure, mfuncs.occupation
)
with ProcessPool(max_workers=4) as executor:
  results = executor.distribute(run_dqmc, params)
```
or the inlcuded `run_parallel`-method, which internally calls the `run_dqmc`-method:
```python
from dqmc import run_dqmc_parallel

results = run_dqmc_parallel(params, max_workers=4)
```
Scalar arguments are expanded to a list and shared by the processes.

## Contributing

Before submitting pull requests, run the lints, tests and optionally the
[black](https://github.com/psf/black) formatter with the following commands
from the root of the repo:
`````commandline
python -m black dqmc/
pre-commit run
`````

## References
1. Z. Bai, W. Chen, R. T. Scalettar and I. Yamazaki
   "Numerical Methods for Quantum Monte Carlo Simulations of the Hubbard Model"
   Series in Contemporary Applied Mathematics, 1-110 (2010) [DOI](https://doi.org/10.1142/9789814273268_0001)
2. S. R. White, D. J. Scalapino, R. L. Sugar, E. Y. Loh, J. E. Gubernatis and R. T. Scalettar
   "Numerical study of the two-dimensional Hubbard model"
   Phys. Rev. B 40, 506-516 (1989) [DOI](https://doi.org/10.1103/PhysRevB.40.506)
3. P. Broecker and S. Trebst
   "Numerical stabilization of entanglement computation in auxiliary field quantum Monte Carlo simulations of interacting many-fermion systems"
   Phys. Rev. E 94, 063306 (2016) [DOI](https://doi.org/10.1103/PhysRevE.94.063306)
4. J. E. Hirsch
   "Two-dimensional Hubbard model: Numerical simulation study"
   Phys. Rev. B 31, 4403 (1985) [DOI](https://doi.org/10.1103/PhysRevB.31.4403)
5. J. E. Hirsch
   "Discrete Hubbard-Stratonovich transformation for fermion lattice models"
   Phys. Rev. B 29, 4159 (1984) [DOI](https://doi.org/10.1103/PhysRevB.28.4059)
