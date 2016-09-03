"""Bandit algorithm classes
"""

from .exp3 import Exp3
from .exp4p import Exp4P
from .linthompsamp import LinThompSamp
from .linucb import LinUCB
from .ucb1 import UCB1


__all__ = ['Exp3', 'Exp4P', 'LinThompSamp', 'LinUCB', 'UCB1']
