import unittest
import numpy as np
import time
import FrictionQPotSpringBlock

class Test_main(unittest.TestCase):

    def test_basic(self):

        self.assertTrue(True)

        N = 5

        y = np.ones((N, 100))
        y[:, 0] = -10.5
        y = np.cumsum(y, axis=1)

        system = FrictionQPotSpringBlock.Line1d.System(N, y)

        system.advanceRightElastic(0.2)

        x = (0.5 - 0.1) * np.ones([N])

        self.assertTrue(np.allclose(x, system.x()))


if __name__ == '__main__':

    unittest.main()
