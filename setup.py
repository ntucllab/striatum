from setuptools import setup

setup(
    name='striatum',
    version='0.0.1',
    description='Contextual bandit in python',
    long_description='Contextual bandit in python',
    author='Y.-Y. Yang, Y.-A. Lin',
    author_email='r02922163@csie.ntu.edu.tw',
    url='https://github.com/ntucllab/straitum',
    setup_requires=[
        'nose>=1.0',
        'coverage',
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
