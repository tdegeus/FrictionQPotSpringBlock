import time
import unittest

import FrictionQPotSpringBlock
import numpy as np
import prrng
import QPot  # noqa:


class Test_main(unittest.TestCase):
    """
    Test Line2d
    """

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
        self.assertTrue(np.allclose(system.f(), 0.0))
        self.assertTrue(np.allclose(system.f_potential(), 0.0))
        self.assertTrue(np.allclose(system.f_frame(), 0.0))
        self.assertTrue(np.allclose(system.f_neighbours(), 0.0))
        self.assertTrue(np.allclose(system.f_damping(), 0.0))

        x = system.x()
        v = system.v()
        dx = float(np.random.random(1))
        dv = float(np.random.random(1))
        x[0] = dx
        v[2] = dv
        system.set_x(x)
        system.set_v(v)
        xmin = np.floor(dx + 0.5)

        f_potential = mu * np.array([xmin - dx, 0, 0, 0, 0])
        f_neighbours = k_neighbours * np.array([-2 * dx, dx, 0, 0, dx])
        f_frame = k_frame * np.array([-dx, 0, 0, 0, 0])
        f_damping = eta * np.array([0, 0, -dv, 0, 0])

        self.assertTrue(np.allclose(system.f_potential(), f_potential))
        self.assertTrue(np.allclose(system.f_frame(), f_frame))
        self.assertTrue(np.allclose(system.f_neighbours(), f_neighbours))
        self.assertTrue(np.allclose(system.f_damping(), f_damping))
        self.assertTrue(np.allclose(system.f(), f_potential + f_frame + f_neighbours + f_damping))

        x = system.x()
        v = system.v()
        dx = 2.0 * float(np.random.random(1))
        dv = 2.0 * float(np.random.random(1))
        x[1] += dx
        v[3] += dv
        system.set_x(x)
        system.set_v(v)

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

        self.assertTrue(np.allclose(system.f_potential(), f_potential))
        self.assertTrue(np.allclose(system.f_frame(), f_frame))
        self.assertTrue(np.allclose(system.f_neighbours(), f_neighbours))
        self.assertTrue(np.allclose(system.f_damping(), f_damping))
        self.assertTrue(np.allclose(system.f(), f_potential + f_frame + f_neighbours + f_damping))

    def test_advanceElastic(self):

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

        system.advanceElastic(0.1, False)
        self.assertTrue(system.residual() < 1e-5)
        self.assertTrue(np.allclose(system.x(), 0.1 * np.ones(N)))
        self.assertTrue(np.isclose(system.x_frame(), 1.1))

        system.advanceElastic(-0.1, False)
        self.assertTrue(system.residual() < 1e-5)
        self.assertTrue(np.allclose(system.x(), 0.0 * np.ones(N)))
        self.assertTrue(np.isclose(system.x_frame(), 0.0))

        system.advanceElastic(1.1, True)
        self.assertTrue(system.residual() < 1e-5)
        self.assertTrue(np.allclose(system.x(), 0.1 * np.ones(N)))
        self.assertTrue(np.isclose(system.x_frame(), 1.1))

        system.advanceElastic(-1.1, True)
        self.assertTrue(system.residual() < 1e-5)
        self.assertTrue(np.allclose(system.x(), 0.0 * np.ones(N)))
        self.assertTrue(np.isclose(system.x_frame(), 0.0))

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

        i_n = system.i()
        system.eventDrivenStep(0.2, False)
        self.assertTrue(system.residual() < 1e-5)
        self.assertTrue(np.allclose(system.x(), (0.5 - 0.1) * np.ones(N)))
        self.assertTrue(np.all(system.i() == i_n))
        self.assertTrue(np.isclose(system.x_frame(), (0.5 - 0.1) * (1.0 + 0.1) / 0.1))

        i_n = system.i()
        system.eventDrivenStep(0.2, True)
        self.assertTrue(np.allclose(system.x(), (0.5 + 0.1) * np.ones(N)))
        self.assertTrue(not np.all(system.i() == i_n))
        self.assertTrue(np.isclose(system.x_frame(), (0.5 + 0.1) * (1.0 + 0.1) / 0.1))

        i_n = system.i()
        system.eventDrivenStep(0.2, False)
        self.assertTrue(np.allclose(system.x(), (1.5 - 0.1) * np.ones(N)))
        self.assertTrue(np.all(system.i() == i_n))
        self.assertTrue(np.isclose(system.x_frame(), (1.5 - 0.1) * (1.0 + 0.1) / 0.1))

        i_n = system.i()
        system.eventDrivenStep(0.2, True)
        self.assertTrue(np.allclose(system.x(), (1.5 + 0.1) * np.ones(N)))
        self.assertTrue(not np.all(system.i() == i_n))
        self.assertTrue(np.isclose(system.x_frame(), (1.5 + 0.1) * (1.0 + 0.1) / 0.1))

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
        self.assertTrue(np.allclose(system.x(), x))

    def test_triggerWeakest(self):

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

        x = np.zeros(N)
        x[0] = 0.5 - 0.1
        system.set_x(x)

        system.triggerWeakest(0.2)

        x[0] = 0.5 + 0.1
        self.assertTrue(np.allclose(system.x(), x))

    def test_chunkedSequence(self):
        """
        Chunked sequence, shift optimally left
        """

        N = 3
        seed = int(time.time())
        initstate = seed + np.arange(N)

        nchunk = 100  # size of chunk of yield positions kept in memory
        nbuffer = 40  # buffer when shifting chunks of yield positions
        ncheck = 15  # boundary region to check of chunk-shifting is needed
        nmax = 20  # maximal boundary region for which chunk-shifting is applied
        init_offset = 50.0  # initial negative position shift

        # generators
        generators = prrng.pcg32_array(initstate)
        regenerators = prrng.pcg32_array(initstate)
        state = generators.state()
        istart = np.zeros(N, dtype=np.int64)

        # draw initial chunk from the generators and convert to yield positions
        y = 2.0 * generators.random([nchunk])
        y = np.cumsum(y, 1)
        y -= init_offset
        ymin = y[:, 0]

        # keep history to the state to facilitate restoring the right chunk for a given position
        # (the buffer can make that the current position is in the last chunk)
        state_n = state.copy()
        istart_n = istart.copy()
        ymin_n = ymin.copy()

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

        restore = FrictionQPotSpringBlock.Line1d.System(
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

            xi = i * x
            self.assertFalse(system.any_redraw(xi))
            system.set_x(xi)

            if not system.all_inbounds(ncheck):

                inbounds_left = system.inbounds_left(nmax)
                inbounds_right = system.inbounds_right(nmax)

                self.assertTrue(np.all(inbounds_left))

                for p in range(N):

                    if not inbounds_right[p]:

                        yp = system.y(p)

                        state_n[p] = state[p]
                        istart_n[p] = istart[p]
                        ymin_n[p] = ymin[p]

                        state[p] = generators[p].state()
                        istart[p] += nchunk

                        yp.shift_dy(istart[p], 2.0 * generators[p].random([nchunk]), nbuffer)
                        ymin[p] = yp.ymin_chunk()

                self.assertFalse(system.any_redraw())

            regenerators.restore(state_n)
            ry = 2.0 * regenerators.random([2 * nchunk])
            ry[:, 0] = ymin_n
            ry = np.cumsum(ry, 1)
            restore.set_y(istart_n, ry)
            restore.set_x(system.x())

            self.assertTrue(np.all(system.i() == restore.i()))
            self.assertTrue(np.allclose(system.yieldDistanceLeft(), restore.yieldDistanceLeft()))
            self.assertTrue(np.allclose(system.yieldDistanceRight(), restore.yieldDistanceRight()))
            self.assertTrue(np.allclose(system.yleft(), restore.yleft()))
            self.assertTrue(np.allclose(system.yright(), restore.yright()))

    def test_chunkedSequence2(self):
        """
        Chunked sequence, shift optimally left.
        """

        N = 3
        seed = int(time.time())
        initstate = seed + np.arange(N)

        nchunk = 100  # size of chunk of yield positions kept in memory
        nbuffer = 10  # buffer to keep left
        init_offset = 5.0  # initial negative position shift

        # generators
        generators = prrng.pcg32_array(initstate)
        regenerators = prrng.pcg32_array(initstate)
        state = generators.state()
        istart = np.zeros(N, dtype=np.int64)

        # draw initial chunk from the generators and convert to yield positions
        y = 2.0 * generators.random([nchunk])
        y = np.cumsum(y, 1)
        y -= init_offset

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

        restore = FrictionQPotSpringBlock.Line1d.System(
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

            xi = i * x
            system.set_x(xi)
            i0 = system.i()

            if np.any(system.i_chunk() > nbuffer):

                for p in range(N):

                    yp = system.y(p)
                    nb = yp.size() - yp.i_chunk() + nbuffer
                    if nb >= nchunk:
                        continue

                    state[p] = generators[p].state()
                    istart[p] = yp.istop()
                    yp.shift_dy(yp.istop(), 2.0 * generators[p].random([nchunk - nb]), nb)

            self.assertTrue(np.all(system.i() == i0))
            self.assertTrue(np.all(system.i() - system.istart() <= nbuffer))
            self.assertTrue(np.all(system.i_chunk() == system.i() - system.istart()))

            # restore state: start with the latests draw that is quite close and reverse back
            # in the sequence until the start of the current chunk held in memory
            regenerators.restore(state)
            regenerators.advance(system.istart() - istart)

            # generate the yield distances, convert to yield positions using the first yield
            # position of the current chunk as memory (was also the state from which the random
            # numbers were generated)
            ry = 2.0 * regenerators.random([nchunk])
            ry[:, 0] = system.ymin()
            ry = np.cumsum(ry, 1)

            restore.set_y(system.istart(), ry)
            restore.set_x(system.x())

            self.assertTrue(np.all(system.i() == restore.i()))
            self.assertTrue(np.allclose(system.yieldDistanceLeft(), restore.yieldDistanceLeft()))
            self.assertTrue(np.allclose(system.yieldDistanceRight(), restore.yieldDistanceRight()))
            self.assertTrue(np.allclose(system.yleft(), restore.yleft()))
            self.assertTrue(np.allclose(system.yright(), restore.yright()))


if __name__ == "__main__":

    unittest.main()
