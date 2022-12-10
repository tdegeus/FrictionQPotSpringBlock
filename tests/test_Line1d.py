import faulthandler
import time
import unittest

import FrictionQPotSpringBlock
import numpy as np
import prrng

faulthandler.enable()


class Test_Line1d_System(unittest.TestCase):
    """
    Test Line1d.System
    """

    def test_version_dependencies(self):

        deps = FrictionQPotSpringBlock.Line1d.version_dependencies()
        deps = [i.split("=")[0] for i in deps]

        self.assertTrue("prrng" in deps)
        self.assertTrue("xtensor" in deps)
        self.assertTrue("xtensor-python" in deps)
        self.assertTrue("xtl" in deps)

    def test_forces(self):

        N = 5
        chunk = prrng.pcg32_tensor_cumsum_1_1(
            shape=[100],
            initstate=np.zeros([N], dtype=int),
            initseq=np.zeros([N], dtype=int),
            distribution=prrng.delta,
            parameters=[1.0],
            align=prrng.alignment(margin=10, buffer=5),
        )
        chunk.data -= 49.5

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
            chunk=chunk,
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
        chunk = prrng.pcg32_tensor_cumsum_1_1(
            shape=[100],
            initstate=np.zeros([N], dtype=int),
            initseq=np.zeros([N], dtype=int),
            distribution=prrng.delta,
            parameters=[1.0],
            align=prrng.alignment(margin=10, buffer=5),
        )
        chunk.data -= 49.5

        system = FrictionQPotSpringBlock.Line1d.System(
            m=1.0,
            eta=1.0,
            mu=1.0,
            k_neighbours=1.0,
            k_frame=0.1,
            dt=1.0,
            chunk=chunk,
        )

        self.assertTrue(system.residual() < 1e-5)

        i_n = system.i
        system.eventDrivenStep(0.2, False)
        self.assertTrue(system.residual() < 1e-5)
        self.assertTrue(np.allclose(system.x, (0.5 - 0.1) * np.ones(N)))
        self.assertTrue(np.all(system.i == i_n))
        self.assertTrue(np.isclose(system.x_frame, (0.5 - 0.1) * (1.0 + 0.1) / 0.1))

        i_n = system.i
        system.eventDrivenStep(0.2, True)
        self.assertTrue(np.allclose(system.x, (0.5 + 0.1) * np.ones(N)))
        self.assertTrue(not np.all(system.i == i_n))
        self.assertTrue(np.isclose(system.x_frame, (0.5 + 0.1) * (1.0 + 0.1) / 0.1))

        i_n = system.i
        system.eventDrivenStep(0.2, False)
        self.assertTrue(np.allclose(system.x, (1.5 - 0.1) * np.ones(N)))
        self.assertTrue(np.all(system.i == i_n))
        self.assertTrue(np.isclose(system.x_frame, (1.5 - 0.1) * (1.0 + 0.1) / 0.1))

        i_n = system.i
        system.eventDrivenStep(0.2, True)
        self.assertTrue(np.allclose(system.x, (1.5 + 0.1) * np.ones(N)))
        self.assertTrue(not np.all(system.i == i_n))
        self.assertTrue(np.isclose(system.x_frame, (1.5 + 0.1) * (1.0 + 0.1) / 0.1))

    def test_trigger(self):

        N = 3
        chunk = prrng.pcg32_tensor_cumsum_1_1(
            shape=[100],
            initstate=np.zeros([N], dtype=int),
            initseq=np.zeros([N], dtype=int),
            distribution=prrng.delta,
            parameters=[1.0],
            align=prrng.alignment(margin=10, buffer=5),
        )
        chunk.data -= 49.5

        system = FrictionQPotSpringBlock.Line1d.System(
            m=1.0,
            eta=1.0,
            mu=1.0,
            k_neighbours=1.0,
            k_frame=0.1,
            dt=1.0,
            chunk=chunk,
        )

        system.trigger(0, 0.2)

        x = np.zeros(N)
        x[0] = 0.5 + 0.1
        self.assertTrue(np.allclose(system.x, x))

    def test_advanceToFixedForce(self):

        N = 3
        chunk = prrng.pcg32_tensor_cumsum_1_1(
            shape=[100],
            initstate=np.zeros([N], dtype=int),
            initseq=np.zeros([N], dtype=int),
            distribution=prrng.delta,
            parameters=[1.0],
            align=prrng.alignment(margin=10, buffer=5),
        )
        chunk.data -= 49.5

        system = FrictionQPotSpringBlock.Line1d.System(
            m=1.0,
            eta=1.0,
            mu=1.0,
            k_neighbours=1.0,
            k_frame=0.1,
            dt=1.0,
            chunk=chunk,
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
        buffer = 20  # redraw within this margin from the edges of the chunk
        margin = 10  # position to place the particle after redraw
        init_offset = 50.0  # initial negative position shift

        # draw reference yield positions
        gen = prrng.pcg32_array(initstate, np.zeros_like(initstate))
        yref = np.cumsum(gen.random([2000]), axis=1) - init_offset

        # chunked storage
        align = prrng.alignment(margin=margin, buffer=buffer)
        chunk = prrng.pcg32_tensor_cumsum_1_1(
            shape=[nchunk],
            initstate=initstate,
            initseq=np.zeros_like(initstate),
            distribution=prrng.random,
            parameters=[],
            align=align,
        )
        chunk -= init_offset

        # initialise system
        system = FrictionQPotSpringBlock.Line1d.System(
            m=1.0,
            eta=1.0,
            mu=1.0,
            k_neighbours=1.0,
            k_frame=0.1,
            dt=1.0,
            chunk=chunk,
        )

        x = 10.0 * np.ones(N)
        x[0] = 5.0
        x[1] = 7.0

        for i in range(50):

            system.x = i * x

            j = prrng.lower_bound(yref, system.x)
            r = np.arange(N)

            self.assertTrue(np.all(system.i == j))
            self.assertTrue(np.allclose(yref[r, system.i], system.y_left()))
            self.assertTrue(np.allclose(yref[r, system.i + 1], system.y_right()))


class Test_Line1d_SystemSemiSmooth(unittest.TestCase):
    """
    Test Line1d.SystemSemiSmooth
    """

    def test_eventDrivenStep(self):

        N = 3
        chunk = prrng.pcg32_tensor_cumsum_1_1(
            shape=[100],
            initstate=np.zeros([N], dtype=int),
            initseq=np.zeros([N], dtype=int),
            distribution=prrng.delta,
            parameters=[1.0],
            align=prrng.alignment(margin=10, buffer=5),
        )
        chunk.data -= 49.5

        mu = 1
        kappa = 0.1

        system = FrictionQPotSpringBlock.Line1d.SystemSemiSmooth(
            m=1.0,
            eta=1.0,
            mu=mu,
            kappa=kappa,
            k_neighbours=1.0,
            k_frame=0.1,
            dt=1.0,
            chunk=chunk,
        )

        self.assertTrue(system.residual() < 1e-5)

        x0 = system.x.copy()
        xf0 = system.x_frame
        left = system.y_left()
        right = system.y_right()
        mid = 0.5 * (left + right)
        upper = (mu * mid + kappa * right) / (mu + kappa)
        lower = (mu * mid + kappa * left) / (mu + kappa)
        eps = 0.001

        self.assertAlmostEqual(system.maxUniformDisplacement(), np.min(upper - system.x))
        system.eventDrivenStep(eps=eps, kick=False)
        self.assertTrue(system.residual() < 1e-5)
        self.assertTrue(np.allclose(system.x, upper - 0.5 * eps))
        self.assertAlmostEqual(system.maxUniformDisplacement(), 0.5 * eps)

        system.eventDrivenStep(eps=eps, kick=True)
        self.assertFalse(system.residual() < 1e-5)
        self.assertTrue(np.allclose(system.x, upper + 0.5 * eps))
        self.assertAlmostEqual(system.maxUniformDisplacement(), 0)

        system.x = x0
        system.x_frame = xf0

        self.assertAlmostEqual(system.maxUniformDisplacement(-1), np.min(system.x - lower))
        system.eventDrivenStep(eps=eps, kick=False, direction=-1)
        self.assertTrue(system.residual() < 1e-5)
        self.assertTrue(np.allclose(system.x, lower + 0.5 * eps))
        self.assertTrue(
            np.isclose(system.maxUniformDisplacement(-1), -0.5 * eps, atol=1e-3, rtol=1e-3)
        )

        system.eventDrivenStep(eps=eps, kick=True, direction=-1)
        self.assertFalse(system.residual() < 1e-5)
        self.assertTrue(np.allclose(system.x, lower - 0.5 * eps))
        self.assertAlmostEqual(system.maxUniformDisplacement(), 0)


class Test_Line1d_System2d(unittest.TestCase):
    """
    Test Line1d.System2d
    """

    def test_interactions(self):

        m = 4
        n = 4
        N = m * n
        chunk = prrng.pcg32_tensor_cumsum_1_1(
            shape=[100],
            initstate=np.zeros([N], dtype=int),
            initseq=np.zeros([N], dtype=int),
            distribution=prrng.delta,
            parameters=[1.0],
            align=prrng.alignment(margin=10, buffer=5),
        )
        chunk.data -= 49.5

        k_neighbours = 0.12
        system = FrictionQPotSpringBlock.Line1d.System2d(
            m=1,
            eta=1,
            mu=1,
            k_neighbours=k_neighbours,
            k_frame=0.1,
            dt=1,
            chunk=chunk,
            width=n,
        )

        self.assertTrue(system.residual() < 1e-5)

        index = np.arange(N).reshape(m, n)
        down = np.roll(index, -1, axis=0)
        up = np.roll(index, 1, axis=0)
        left = np.roll(index, 1, axis=1)
        right = np.roll(index, -1, axis=1)

        self.assertTrue(np.all(system.down == down.ravel()))
        self.assertTrue(np.all(system.up == up.ravel()))
        self.assertTrue(np.all(system.left == left.ravel()))
        self.assertTrue(np.all(system.right == right.ravel()))

        c = -4
        f0 = np.array([[c, 1, 0, 1], [1, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]]) * k_neighbours
        x0 = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

        for i in range(m):
            for j in range(n):
                system.x = np.roll(np.roll(x0, i, axis=0), j, axis=1).ravel()
                f = np.roll(np.roll(f0, i, axis=0), j, axis=1).ravel()
                self.assertTrue(np.allclose(system.f_neighbours, f))


if __name__ == "__main__":

    unittest.main(verbosity=2)
