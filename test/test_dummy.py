import unittest

class DummyTest1(unittest.TestCase):
    def test_dummy1(self):
        self.assertTrue(True)

    def test_dummmy2(self):
        self.assertFalse(False)


if __name__ == '__main__':
    unittest.main()
