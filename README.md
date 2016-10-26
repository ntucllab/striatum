# Striatum
Contextual bandit in python

[![Build Status](https://travis-ci.org/ntucllab/striatum.svg?branch=master)](https://travis-ci.org/ntucllab/striatum)
[![Documentation Status](https://readthedocs.org/projects/striatum/badge/?version=latest)](http://striatum.readthedocs.io/en/latest/?badge=latest)

## Installation

- Install `striatum`

  ```bash
  pip install striatum
  ```

- There may be some problem installing `matplotlib`
  - Use `apt-get install python-matplotlib` (for Python 2) or `apt-get install python3-matplotlib` (for Python 3) if you don't want to meet any problem
  - Install `tk-dev` and `tcl-dev` if you want to use pip to install `matplotlib` (`apt-get install tk-dev tcl-dev` for Ubuntu>=14.04)

## Test
```bash
git clone https://github.com/ntucllab/striatum.git
cd striatum
pip install -r requirements.txt
# if you only want to test your current environment
python setup.py test
# if you want to test multiple environments and installation
tox
```
