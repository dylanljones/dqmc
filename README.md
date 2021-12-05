# dqmc 0.0.1

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![GitHub license](https://badgen.net/github/license/Naereen/Strapdown.js)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE)

Determinant Quantum Monte Carlo simulations in python

NOTE: This project is still under development!

## Installation

Download package and install via pip
````commandline
pip install -e <folder path>
````
or the setup.py script
````commandline
python setup.py install
````

## Usage


## References
1. Z. Bai, W. Chen, R. T. Scalettar and I. Yamazaki
   Numerical Methods for Quantum Monte Carlo Simulations of the Hubbard Model
   Series in Contemporary Applied Mathematics, 1-110 (2010) [DOI](10.1142/9789814273268_0001)
2. S. R. White, D. J. Scalapino, R. L. Sugar, E. Y. Loh, J. E. Gubernatis, and R. T. Scalettar
   Numerical study of the two-dimensional Hubbard model
   Phys. Rev. B 40, 506-516 (1989) [DOI](https://doi.org/10.1103/PhysRevB.40.506)


## Contributing

Before submitting pull requests, run the [black](https://github.com/psf/black)
formatter, lints and tests with the following commands from the root of the repo:
`````commandline
pre-commit run
`````
