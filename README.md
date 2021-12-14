# DQMC

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![GitHub license](https://badgen.net/github/license/Naereen/Strapdown.js)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE)

Determinant Quantum Monte Carlo simulations of the Hubbard model in Python.

*NOTE*: This project is still under development and might contain errors or change significantly in the future!

## Installation

Install via `pip` from github:
```commandline
pip install git+git://github.com/dylanljones/cmpy.git@VERSION
```
or download/clone the package, navigate to the root directory and install via
````commandline
pip install -e <folder path>
````
or the `setup.py` script
````commandline
python setup.py install
````

## Usage

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
# Build the lattice iwth periodic boundary conditions along both axis
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

To run a Determinant Quantum Monte carlo simulation the `DQMC`-object can be used. 
This is a wrapper of the main DQMC methods, which are contained in `dqmc/dqmc.py` 
and use jit (just in time compilation) to improve performance:
```python
from dqmc import hubbard_hypercube, mfuncs, DQMC

shape = (5, 5)
num_timesteps = 100
warmup, measure = 300, 3000
model = hubbard_hypercube(shape, u=4., eps=0., hop=1., mu=0., beta=1/5, periodic=True)

dqmc = DQMC(model, num_timesteps)
results = dqmc.simulate(warmup, measure, callback=mfuncs.occupation)
```
The `simulate`-method has a `callback` parameter for measuring observables, which 
expects a method of the form
```python
def callback(gf_up, gf_dn):
    ...
    return result
```
The returned result must be a `np.ndarray` for ensuring correct averaging after the 
measurement sweeps. If no callback is given the local Green's function `G_{ij}` is
measured by default.


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
