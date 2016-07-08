import unittest

import numpy as np

from striatum.bandit import UCB1
from striatum.storage import MemoryHistoryStorage


class TestUCB1(unittest.TestCase):

    def test_ucb1(self):
        ucb1 = UCB1()



if __name__ == '__main__':
    unittest.main()
