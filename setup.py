# coding: utf-8
#
# This code is part of dqmc.
#
# Copyright (c) 2021, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

from setuptools import find_packages
import versioneer
from numpy.distutils.core import setup, Extension


def requirements():
    with open("requirements.txt", "r") as f:
        return f.readlines()


def long_description():
    with open("README.md", "r") as f:
        return f.read()


ext_modules = [
    Extension(name="dqmc.src.timeflow", sources=["dqmc/src/timeflow.f90"],
              libraries=["lapack", "blas"]),
    Extension(name="dqmc.src.greens", sources=["dqmc/src/greens.f90"],
              libraries=["lapack", "blas"])
]


setup(
    name="dqmc",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="Dylan Jones",
    author_email="dylanljones94@gmail.com",
    description="Determinant Quantum Monte Carlo simulations in python",
    long_description=long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/dylanljones/dqmc",
    packages=find_packages(),
    ext_modules=ext_modules,
    license="MIT License",
    install_requires=requirements(),
    python_requires=">=3.6",
)
