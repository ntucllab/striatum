from setuptools import setup

setup(
    name='striatum',
    version='0.0.0.dev1',
    description='Contextual bandit in python',
    long_description='Contextual bandit in python',
    author='test',
    author_email='test@example.com',
    url='https://github.com/ntucllab/straitum',
    setup_requires=[
        'nose>=1.0',
    ],
    classifiers=[
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
    ],
    test_suite='striatum',
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
