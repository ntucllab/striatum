#!/usr/bin/env python

from setuptools import setup

setup(
    name='striatum',
    version='0.0.1',
    description='Contextual bandit in python',
    long_description='Contextual bandit in python',
    author='Y.-Y. Yang, Y.-A. Lin',
    author_email='b01902066@csie.ntu.edu.tw, r02922163@csie.ntu.edu.tw',
    url='https://github.com/ntucllab/straitum',
    setup_requires=[
        'numpy',
        'scipy',
        'scikit-learn',
    ],
    classifiers=[
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
    ],
    test_suite='nose.collector',
    packages=[
        'striatum',
        'striatum.bandit',
        'striatum.storage'
    ],
    package_dir={
        'striatum': 'striatum',
        'striatum.bandit': 'striatum/bandit',
        'striatum.storage': 'striatum/storage'
    },
)
