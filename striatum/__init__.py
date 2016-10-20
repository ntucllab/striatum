"""
striatum
========

Available subpackages
---------------------
bandit
    Implementation of bandit algorithms.
storage
    Interface for storing history.

"""
import pkg_resources


__all__ = ['bandit', 'storage']
__version__ = pkg_resources.get_distribution('striatum').version
