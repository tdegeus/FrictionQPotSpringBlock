import faulthandler
import time
import unittest

import FrictionQPotSpringBlock
import numpy as np
import prrng
import QPot

faulthandler.enable()


class Test_Line1d_System(unittest.TestCase):
    """
    Test Line1d.System
    """

    def test_version_dependencies(self):

        deps = FrictionQPotSpringBlock.Line1d.version_dependencies()
        deps = [i.split("=")[0] for i in deps]

        self.assertTrue("qpot" in deps)
        self.assertTrue("xtensor" in deps)
        self.assertTrue("xtensor-python" in deps)
        self.assertTrue("xtl" in deps)

    def test_forces(self):

        N = 5
        y = np.ones((N, 100))
        y[:, 0] = -48.5
        y = np.cumsum(y, axis=1)

        eta = float(np.random.random(1))
        mu = float(np.random.random(1))
        k_neighbours = float(np.random.random(1))
        k_frame = float(np.random.random(1))

        system = FrictionQPotSpringBlock.Line1d.System(
            m=1.0,
            eta=eta,
            mu=mu,
            k_neighbours=k_neighbours,
            k_frame=k_frame,
            dt=1.0,
            x_yield=y,
        )

        self.assertTrue(system.residual() < 1e-5)
        self.assertTrue(np.allclose(system.f, 0.0))
        self.assertTrue(np.allclose(system.f_potential, 0.0))
        self.assertTrue(np.allclose(system.f_frame, 0.0))
        self.assertTrue(np.allclose(system.f_neighbours, 0.0))
        self.assertTrue(np.allclose(system.f_damping, 0.0))

        dx = np.zeros(N)
        dv = np.zeros(N)
        dx[0] = float(np.random.random(1))
        dv[2] = float(np.random.random(1))
        system.x += dx
        system.v += dv
        xmin = np.floor(dx[0] + 0.5)

        f_potential = mu * np.array([xmin - dx[0], 0, 0, 0, 0])
        f_neighbours = k_neighbours * np.array([-2 * dx[0], dx[0], 0, 0, dx[0]])
        f_frame = k_frame * np.array([-dx[0], 0, 0, 0, 0])
        f_damping = eta * np.array([0, 0, -dv[2], 0, 0])

        self.assertTrue(np.allclose(system.f_potential, f_potential))
        self.assertTrue(np.allclose(system.f_frame, f_frame))
        self.assertTrue(np.allclose(system.f_neighbours, f_neighbours))
        self.assertTrue(np.allclose(system.f_damping, f_damping))
        self.assertTrue(np.allclose(system.f, f_potential + f_frame + f_neighbours + f_damping))

        dx = np.zeros(N)
        dv = np.zeros(N)
        dx[1] = 2.0 * float(np.random.random(1))
        dv[3] = 2.0 * float(np.random.random(1))
        system.x += dx
        system.v += dv
        x = system.x
        v = system.v

        f_potential = mu * np.array(
            [np.floor(x[0] + 0.5) - x[0], np.floor(x[1] + 0.5) - x[1], 0, 0, 0]
        )
        f_neighbours = k_neighbours * np.array(
            [
                x[-1] - 2 * x[0] + x[1],
                x[0] - 2 * x[1] + x[2],
                x[1] - 2 * x[2] + x[3],
                0,
                x[-2] - 2 * x[-1] + x[0],
            ]
        )
        f_frame = k_frame * np.array([-x[0], -x[1], 0, 0, 0])
        f_damping = eta * np.array([0, 0, -v[2], -v[3], 0])

        self.assertTrue(np.allclose(system.f_potential, f_potential))
        self.assertTrue(np.allclose(system.f_frame, f_frame))
        self.assertTrue(np.allclose(system.f_neighbours, f_neighbours))
        self.assertTrue(np.allclose(system.f_damping, f_damping))
        self.assertTrue(np.allclose(system.f, f_potential + f_frame + f_neighbours + f_damping))

    def test_eventDrivenStep(self):

        N = 3
        y = np.ones((N, 100))
        y[:, 0] = -48.5
        y = np.cumsum(y, axis=1)

        system = FrictionQPotSpringBlock.Line1d.System(
            m=1.0,
            eta=1.0,
            mu=1.0,
            k_neighbours=1.0,
            k_frame=0.1,
            dt=1.0,
            x_yield=y,
        )

        self.assertTrue(system.residual() < 1e-5)

        i_n = np.copy(system.i)
        system.eventDrivenStep(0.2, False)
        self.assertTrue(system.residual() < 1e-5)
        self.assertTrue(np.allclose(system.x, (0.5 - 0.1) * np.ones(N)))
        self.assertTrue(np.all(system.i == i_n))
        self.assertTrue(np.isclose(system.x_frame, (0.5 - 0.1) * (1.0 + 0.1) / 0.1))

        i_n = np.copy(system.i)
        system.eventDrivenStep(0.2, True)
        self.assertTrue(np.allclose(system.x, (0.5 + 0.1) * np.ones(N)))
        self.assertTrue(not np.all(system.i == i_n))
        self.assertTrue(np.isclose(system.x_frame, (0.5 + 0.1) * (1.0 + 0.1) / 0.1))

        i_n = np.copy(system.i)
        system.eventDrivenStep(0.2, False)
        self.assertTrue(np.allclose(system.x, (1.5 - 0.1) * np.ones(N)))
        self.assertTrue(np.all(system.i == i_n))
        self.assertTrue(np.isclose(system.x_frame, (1.5 - 0.1) * (1.0 + 0.1) / 0.1))

        i_n = np.copy(system.i)
        system.eventDrivenStep(0.2, True)
        self.assertTrue(np.allclose(system.x, (1.5 + 0.1) * np.ones(N)))
        self.assertTrue(not np.all(system.i == i_n))
        self.assertTrue(np.isclose(system.x_frame, (1.5 + 0.1) * (1.0 + 0.1) / 0.1))

    def test_trigger(self):

        N = 3
        y = np.ones((N, 100))
        y[:, 0] = -48.5
        y = np.cumsum(y, axis=1)

        system = FrictionQPotSpringBlock.Line1d.System(
            m=1.0,
            eta=1.0,
            mu=1.0,
            k_neighbours=1.0,
            k_frame=0.1,
            dt=1.0,
            x_yield=y,
        )

        system.trigger(0, 0.2)

        x = np.zeros(N)
        x[0] = 0.5 + 0.1
        self.assertTrue(np.allclose(system.x, x))

    def test_advanceToFixedForce(self):

        N = 3
        y = np.ones((N, 100))
        y[:, 0] = -48.5
        y = np.cumsum(y, axis=1)

        system = FrictionQPotSpringBlock.Line1d.System(
            m=1.0,
            eta=1.0,
            mu=1.0,
            k_neighbours=1.0,
            k_frame=0.1,
            dt=1.0,
            x_yield=y,
        )

        self.assertTrue(system.residual() < 1e-5)
        system.advanceToFixedForce(0.1)
        self.assertTrue(np.isclose(np.mean(system.f_frame), 0.1))
        self.assertTrue(system.residual() < 1e-5)

        self.assertTrue(system.residual() < 1e-5)
        system.advanceToFixedForce(0.0)
        self.assertTrue(np.isclose(np.mean(system.f_frame), 0.0))
        self.assertTrue(np.allclose(system.x, 0.0))
        self.assertTrue(np.allclose(system.x_frame, 0.0))

    def test_chunked(self):
        """
        Chunked sequence, shift optimally left
        """

        N = 3
        seed = int(time.time())
        initstate = seed + np.arange(N)

        nchunk = 100  # size of chunk of yield positions kept in memory
        nbuffer = 50  # buffer when shifting chunks of yield positions
        nmargin = 30  # boundary region to check of chunk-shifting is needed
        init_offset = 50.0  # initial negative position shift

        # generators
        generators = prrng.pcg32_array(initstate)
        state = generators.state()
        istate = np.zeros(N, dtype=np.int64)
        istart = np.zeros(N, dtype=np.int64)

        # draw reference yield positions
        yref = 2.0 * generators.random([nchunk * 10])
        yref = np.cumsum(yref, 1)
        yref -= init_offset
        generators.restore(state)

        # draw initial chunk from the generators and convert to yield positions
        y = 2.0 * generators.random([nchunk])
        y = np.cumsum(y, 1)
        y -= init_offset
        istate += nchunk

        # initialise system
        system = FrictionQPotSpringBlock.Line1d.System(
            m=1.0,
            eta=1.0,
            mu=1.0,
            k_neighbours=1.0,
            k_frame=0.1,
            dt=1.0,
            x_yield=y,
        )

        x = 10.0 * np.ones(N)
        x[0] = 5.0
        x[1] = 7.0

        for i in range(50):

            system.x = i * x

            if np.any(system.i > system.y.shape[1] - nmargin):

                shift = system.i - nbuffer + 1
                advance = np.where(shift < 0, shift + istart - istate, nchunk + istart - istate)
                generators.advance(advance)
                istate += advance
                n = np.max(np.abs(shift)) + 1
                dy = 2.0 * generators.random([n])
                state = generators.state()
                istart += shift
                istate += n
                system.y = QPot.cumsum_chunk(system.y, dy, shift)

            j = QPot.lower_bound(yref, i * x)
            r = np.arange(N)

            self.assertTrue(np.all(system.i + istart == j))
            self.assertTrue(np.allclose(system.y[r, system.i], yref[r, j]))
            self.assertTrue(np.allclose(system.y[r, system.i], system.y_left()))
            self.assertTrue(np.allclose(system.y[r, system.i + 1], yref[r, j + 1]))
            self.assertTrue(np.allclose(system.y[r, system.i + 1], system.y_right()))


if __name__ == "__main__":

    unittest.main(verbosity=2)
