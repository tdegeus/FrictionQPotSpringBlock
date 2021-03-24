import unittest
import numpy as np
import time
import QPot
import FrictionQPotSpringBlock

class Test_main(unittest.TestCase):

    def test_basic(self):

        N = 5
        uniform = QPot.random.UniformList()
        system = FrictionQPotSpringBlock.Line1d.System(N, uniform)

        system.advanceRightElastic(0.2)

        x = (0.5 - 0.1) * np.ones([N])

        self.assertTrue(np.allclose(x, system.x()))

    def test_reconstruct(self):

        N = 5
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
                redraw += [system.getRedrawList().currentRedraw()]

        QPot.random.seed(seed)
        other = FrictionQPotSpringBlock.Line1d.System(N, random)

        for i in redraw:
            other.getRedrawList().redraw(i)

        other.set_x(system.x());

        self.assertTrue(np.allclose(system.yieldLeft(), other.yieldLeft()))
        self.assertTrue(np.all(np.equal(system.yieldIndex(), other.yieldIndex())))

    def test_reconstruct_minimal_data(self):

        N = 5
        seed = int(time.time())
        random = QPot.random.RandList()

        QPot.random.seed(seed)
        system = FrictionQPotSpringBlock.Line1d.System(N, random)

        n = 20
        x = 100.0 * np.ones((N))
        direction = []
        particles = []

        for i in range(20):
            r = system.set_x(float(i) * x)
            if r:
                iredraw = system.getRedrawList().currentRedraw()
                r = np.argwhere(iredraw > 0).ravel()
                l = np.argwhere(iredraw < 0).ravel()
                if r.size > 0:
                    direction += [+1]
                    particles += [r]
                if l.size > 0:
                    direction += [-1]
                    particles += [l]

        QPot.random.seed(seed)
        other = FrictionQPotSpringBlock.Line1d.System(N, random)

        for d, p in zip(direction, particles):
            if d > 0:
                other.getRedrawList().redrawRight(p)
            else:
                other.getRedrawList().redrawLeft(p)

        other.set_x(system.x());

        self.assertTrue(np.allclose(system.yieldLeft(), other.yieldLeft()))
        self.assertTrue(np.all(np.equal(system.yieldIndex(), other.yieldIndex())))

if __name__ == '__main__':

    unittest.main()
