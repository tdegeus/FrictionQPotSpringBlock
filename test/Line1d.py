import unittest
import numpy as np
import time
import QPot
import FrictionQPotSpringBlock

class Test_main(unittest.TestCase):

    def test_basic(self):

        N = 3
        uniform = QPot.random.UniformList()
        system = FrictionQPotSpringBlock.Line1d.System(N, uniform)

        system.advanceRightElastic(0.2)

        x = (0.5 - 0.1) * np.ones([N])

        self.assertTrue(np.allclose(x, system.x()))

    def test_reconstruct(self):

        N = 3
        seed = int(time.time())
        random = QPot.random.RandList()

        QPot.random.seed(seed)
        system = FrictionQPotSpringBlock.Line1d.System(N, random)

        n = 20
        x = 100.0 * np.ones((N))
        redraw = []

        for i in range(20):
            r = system.set_x(float(i) * x)
            if r:
                redraw += [system.QPot().currentRedraw()]

        QPot.random.seed(seed)
        other = FrictionQPotSpringBlock.Line1d.System(N, random)

        for i in redraw:
            other.QPot().redraw(i)

        other.set_x(system.x());

        self.assertTrue(np.allclose(system.yieldLeft(), other.yieldLeft()))
        self.assertTrue(np.all(np.equal(system.yieldIndex(), other.yieldIndex())))
