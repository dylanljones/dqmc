# coding: utf-8
#
# This code is part of dqmc.
#
# Copyright (c) 2021, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

from setuptools import setup, find_packages


def requirements():
    with open("requirements.txt", "r") as f:
        return f.readlines()


def long_description():
    with open("README.md", "r") as f:
        return f.read()


setup(
    name='dqmc',
    version='0.0.1',
    author='Dylan Jones',
    author_email='dylanljones94@gmail.com',
    description='Python package for Determinant Quantum Monte Carlo simulations',
    long_description=long_description(),
    long_description_content_type="text/markdown",
    # url='https://github.com/dylanljones/lattpy',
    packages=find_packages(),
    license='MIT License',
    install_requires=requirements(),
    python_requires='>=3.6',
)
