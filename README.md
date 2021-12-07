# dqmc 0.0.1

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![GitHub license](https://badgen.net/github/license/Naereen/Strapdown.js)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE)

Determinant Quantum Monte Carlo simulations in Python.

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


## Contributing

Before submitting pull requests, run the lints, tests and optionally the
[black](https://github.com/psf/black) formatter with the following commands
from the root of the repo:
`````commandline
python -m black dqmc/
pre-commit run
`````
