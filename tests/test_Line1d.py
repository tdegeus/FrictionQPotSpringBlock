import faulthandler
import time
import unittest

import FrictionQPotSpringBlock
import numpy as np
import prrng

faulthandler.enable()


class Test_support(unittest.TestCase):
    def test_version_dependencies(self):
        deps = FrictionQPotSpringBlock.Line1d.version_dependencies()
        deps = [i.split("=")[0] for i in deps]

        self.assertTrue("prrng" in deps)
        self.assertTrue("xtensor" in deps)
        self.assertTrue("xtensor-python" in deps)
        self.assertTrue("xtl" in deps)


class Test_Uniform(unittest.TestCase):
    def test_init(self):
        """
        Initial state in a uniform system aligned with the potential energy.
        """

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
        k_interactions = float(np.random.random(1))
        k_frame = float(np.random.random(1))
        par = dict(
            m=1,
            eta=eta,
            mu=mu,
            k_interactions=k_interactions,
            k_frame=k_frame,
            dt=1.0,
            chunk=chunk,
        )
        rand = dict(
            mean=0,
            stddev=1,
            seed=0,
            dinc_init=np.ones(N, dtype=int),
            dinc=np.ones(N, dtype=int),
        )

        systems = [
            FrictionQPotSpringBlock.Line1d.System_Cuspy_Laplace(**par),
            FrictionQPotSpringBlock.Line1d.System_SemiSmooth_Laplace(kappa=1, **par),
            FrictionQPotSpringBlock.Line1d.System_Smooth_Laplace(**par),
            FrictionQPotSpringBlock.Line1d.System_Cuspy_LongRange(alpha=1, **par),
            FrictionQPotSpringBlock.Line1d.System_Cuspy_Laplace_RandomForcing(**par, **rand),
        ]

        par.pop("m")
        systems += [FrictionQPotSpringBlock.Line1d.System_Cuspy_Laplace_Nopassing(**par)]

        par["m"] = 1
        par.pop("k_interactions")
        systems += [FrictionQPotSpringBlock.Line1d.System_Cuspy_Quartic(a1=1, a2=1, **par)]
        systems += [FrictionQPotSpringBlock.Line1d.System_Cuspy_QuarticGradient(k2=1, k4=1, **par)]

        # by construction u = 0 which is a local minimum in all potentials
        for system in systems:
            self.assertLess(system.residual(), 1e-5)
            self.assertTrue(np.allclose(system.f, 0.0))
            self.assertTrue(np.allclose(system.f_potential, 0.0))
            self.assertTrue(np.allclose(system.f_frame, 0.0))
            self.assertTrue(np.allclose(system.f_interactions, 0.0))
            self.assertTrue(np.allclose(system.f_damping, 0.0))
            self.assertTrue(np.all(chunk.index_at_align + 1 == np.argmax(chunk.data[0, :] > 0)))
            self.assertTrue(np.all(chunk.right_of_align > 0))
            self.assertTrue(np.all(chunk.left_of_align <= 0))


class Test_System_Cuspy_Laplace(unittest.TestCase):
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
        k_interactions = float(np.random.random(1))
        k_frame = float(np.random.random(1))

        system = FrictionQPotSpringBlock.Line1d.System_Cuspy_Laplace(
            m=1.0,
            eta=eta,
            mu=mu,
            k_interactions=k_interactions,
            k_frame=k_frame,
            dt=1.0,
            chunk=chunk,
        )

        # by construction u = 0 which is a local minimum in all potentials
        self.assertLess(system.residual(), 1e-5)
        self.assertTrue(np.allclose(system.f, 0.0))
        self.assertTrue(np.allclose(system.f_potential, 0.0))
        self.assertTrue(np.allclose(system.f_frame, 0.0))
        self.assertTrue(np.allclose(system.f_interactions, 0.0))
        self.assertTrue(np.allclose(system.f_damping, 0.0))
        self.assertTrue(np.all(chunk.index_at_align + 1 == np.argmax(chunk.data[0, :] > 0)))
        self.assertTrue(np.all(chunk.right_of_align > 0))
        self.assertTrue(np.all(chunk.left_of_align <= 0))

        du = np.zeros(N)
        dv = np.zeros(N)
        du[0] = float(np.random.random(1))
        dv[2] = float(np.random.random(1))
        system.u += du
        system.v += dv
        umin = np.floor(du[0] + 0.5)

        f_potential = mu * np.array([umin - du[0], 0, 0, 0, 0])
        f_interactions = k_interactions * np.array([-2 * du[0], du[0], 0, 0, du[0]])
        f_frame = k_frame * np.array([-du[0], 0, 0, 0, 0])
        f_damping = eta * np.array([0, 0, -dv[2], 0, 0])

        self.assertTrue(np.all(chunk.right_of_align > system.u))
        self.assertTrue(np.all(chunk.left_of_align <= system.u))
        self.assertTrue(np.allclose(system.f_potential, f_potential))
        self.assertTrue(np.allclose(system.f_frame, f_frame))
        self.assertTrue(np.allclose(system.f_interactions, f_interactions))
        self.assertTrue(np.allclose(system.f_damping, f_damping))
        self.assertTrue(np.allclose(system.f, f_potential + f_frame + f_interactions + f_damping))

        du = np.zeros(N)
        dv = np.zeros(N)
        du[1] = 2.0 * float(np.random.random(1))
        dv[3] = 2.0 * float(np.random.random(1))
        system.u += du
        system.v += dv
        u = system.u
        v = system.v

        f_potential = mu * np.array(
            [np.floor(u[0] + 0.5) - u[0], np.floor(u[1] + 0.5) - u[1], 0, 0, 0]
        )
        f_interactions = k_interactions * np.array(
            [
                u[-1] - 2 * u[0] + u[1],
                u[0] - 2 * u[1] + u[2],
                u[1] - 2 * u[2] + u[3],
                0,
                u[-2] - 2 * u[-1] + u[0],
            ]
        )
        f_frame = k_frame * np.array([-u[0], -u[1], 0, 0, 0])
        f_damping = eta * np.array([0, 0, -v[2], -v[3], 0])

        self.assertTrue(np.all(chunk.right_of_align > system.u))
        self.assertTrue(np.all(chunk.left_of_align <= system.u))
        self.assertTrue(np.allclose(system.f_potential, f_potential))
        self.assertTrue(np.allclose(system.f_frame, f_frame))
        self.assertTrue(np.allclose(system.f_interactions, f_interactions))
        self.assertTrue(np.allclose(system.f_damping, f_damping))
        self.assertTrue(np.allclose(system.f, f_potential + f_frame + f_interactions + f_damping))

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

        system = FrictionQPotSpringBlock.Line1d.System_Cuspy_Laplace(
            m=1.0,
            eta=1.0,
            mu=1.0,
            k_interactions=1.0,
            k_frame=0.1,
            dt=1.0,
            chunk=chunk,
        )
        self.assertLess(system.residual(), 1e-5)

        i_n = chunk.index_at_align
        system.eventDrivenStep(0.2, False)
        self.assertLess(system.residual(), 1e-5)
        self.assertTrue(np.allclose(system.u, (0.5 - 0.1) * np.ones(N)))
        self.assertTrue(np.all(chunk.index_at_align == i_n))
        self.assertAlmostEqual(system.u_frame, (0.5 - 0.1) * (1.0 + 0.1) / 0.1)

        i_n = chunk.index_at_align
        system.eventDrivenStep(0.2, True)
        self.assertTrue(np.allclose(system.u, (0.5 + 0.1) * np.ones(N)))
        self.assertTrue(not np.all(chunk.index_at_align == i_n))
        self.assertAlmostEqual(system.u_frame, (0.5 + 0.1) * (1.0 + 0.1) / 0.1)

        i_n = chunk.index_at_align
        system.eventDrivenStep(0.2, False)
        self.assertTrue(np.allclose(system.u, (1.5 - 0.1) * np.ones(N)))
        self.assertTrue(np.all(chunk.index_at_align == i_n))
        self.assertAlmostEqual(system.u_frame, (1.5 - 0.1) * (1.0 + 0.1) / 0.1)

        i_n = chunk.index_at_align
        system.eventDrivenStep(0.2, True)
        self.assertTrue(np.allclose(system.u, (1.5 + 0.1) * np.ones(N)))
        self.assertTrue(not np.all(chunk.index_at_align == i_n))
        self.assertAlmostEqual(system.u_frame, (1.5 + 0.1) * (1.0 + 0.1) / 0.1)

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

        system = FrictionQPotSpringBlock.Line1d.System_Cuspy_Laplace(
            m=1.0,
            eta=1.0,
            mu=1.0,
            k_interactions=1.0,
            k_frame=0.1,
            dt=1.0,
            chunk=chunk,
        )

        system.trigger(0, 0.2)

        u = np.zeros(N)
        u[0] = 0.5 + 0.1
        self.assertTrue(np.allclose(system.u, u))

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

        system = FrictionQPotSpringBlock.Line1d.System_Cuspy_Laplace(
            m=1.0,
            eta=1.0,
            mu=1.0,
            k_interactions=1.0,
            k_frame=0.1,
            dt=1.0,
            chunk=chunk,
        )
        self.assertLess(system.residual(), 1e-5)
        system.advanceToFixedForce(0.1)
        self.assertAlmostEqual(np.mean(system.f_frame), 0.1)
        self.assertLess(system.residual(), 1e-5)

        self.assertLess(system.residual(), 1e-5)
        system.advanceToFixedForce(0.0)
        self.assertAlmostEqual(np.mean(system.f_frame), 0.0)
        self.assertTrue(np.allclose(system.u, 0.0))
        self.assertTrue(np.allclose(system.u_frame, 0.0))

    def test_chunked(self):
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

        # chunked storage of "yref" (same seed)
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

        system = FrictionQPotSpringBlock.Line1d.System_Cuspy_Laplace(
            m=1.0,
            eta=1.0,
            mu=1.0,
            k_interactions=1.0,
            k_frame=0.1,
            dt=1.0,
            chunk=chunk,
        )

        du = 10.0 * np.ones(N)
        du[0] = 5.0
        du[1] = 7.0

        for i in range(50):
            system.u = i * du

            j = prrng.lower_bound(yref, system.u)
            r = np.arange(N)

            self.assertTrue(np.all(chunk.index_at_align == j))
            self.assertTrue(np.allclose(yref[r, chunk.index_at_align], chunk.left_of_align))
            self.assertTrue(np.allclose(yref[r, chunk.index_at_align + 1], chunk.right_of_align))


class Test_System_SemiSmooth_Laplace(unittest.TestCase):
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

        system = FrictionQPotSpringBlock.Line1d.System_SemiSmooth_Laplace(
            m=1.0,
            eta=1.0,
            mu=mu,
            kappa=kappa,
            k_interactions=1.0,
            k_frame=0.1,
            dt=1.0,
            chunk=chunk,
        )
        self.assertLess(system.residual(), 1e-5)

        u0 = system.u.copy()
        uf0 = system.u_frame
        left = chunk.left_of_align
        right = chunk.right_of_align
        mid = 0.5 * (left + right)
        upper = (mu * mid + kappa * right) / (mu + kappa)
        lower = (mu * mid + kappa * left) / (mu + kappa)
        eps = 0.001

        self.assertAlmostEqual(system.maxUniformDisplacement(), np.min(upper - system.u))
        system.eventDrivenStep(eps=eps, kick=False)
        self.assertLess(system.residual(), 1e-5)
        self.assertTrue(np.allclose(system.u, upper - 0.5 * eps))
        self.assertAlmostEqual(system.maxUniformDisplacement(), 0.5 * eps)

        system.eventDrivenStep(eps=eps, kick=True)
        self.assertGreater(system.residual(), 1e-5)
        self.assertTrue(np.allclose(system.u, upper + 0.5 * eps))
        self.assertAlmostEqual(system.maxUniformDisplacement(), 0)

        system.u = u0
        system.u_frame = uf0

        self.assertAlmostEqual(system.maxUniformDisplacement(-1), np.min(system.u - lower))
        system.eventDrivenStep(eps=eps, kick=False, direction=-1)
        self.assertLess(system.residual(), 1e-5)
        self.assertTrue(np.allclose(system.u, lower + 0.5 * eps))
        self.assertAlmostEqual(system.maxUniformDisplacement(-1), 0.5 * eps)

        system.eventDrivenStep(eps=eps, kick=True, direction=-1)
        self.assertGreater(system.residual(), 1e-5)
        self.assertTrue(np.allclose(system.u, lower - 0.5 * eps))
        self.assertAlmostEqual(system.maxUniformDisplacement(), 0)


class Test_System_Cuspy_QuarticGradient(unittest.TestCase):
    def test_interactions(self):
        N = 10
        chunk = prrng.pcg32_tensor_cumsum_1_1(
            shape=[100],
            initstate=np.zeros([N], dtype=int),
            initseq=np.zeros([N], dtype=int),
            distribution=prrng.delta,
            parameters=[1.0],
            align=prrng.alignment(margin=10, buffer=5),
        )
        chunk.data -= 49.5

        k2 = 0.12
        k4 = 0.34
        system = FrictionQPotSpringBlock.Line1d.System_Cuspy_QuarticGradient(
            m=1,
            eta=1,
            mu=1,
            k2=k2,
            k4=k4,
            k_frame=0.1,
            dt=1,
            chunk=chunk,
        )

        self.assertLess(system.residual(), 1e-5)

        u0 = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        laplace = np.array([-2, 1, 0, 0, 0, 0, 0, 0, 0, 1])
        gradient = np.array([0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0.5])

        f0 = k2 * laplace + k4 * laplace * gradient**2

        for i in range(N):
            u = np.roll(u0, i)
            system.u = u.ravel()
            f = np.roll(f0, i)
            self.assertTrue(np.allclose(system.f_interactions, f))
            self.assertTrue(np.allclose(system.u, u))


class Test_System_Cuspy_LongRange(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        N = 10
        self.N = N
        self.chunk = prrng.pcg32_tensor_cumsum_1_1(
            shape=[100],
            initstate=np.zeros([N], dtype=int),
            initseq=np.zeros([N], dtype=int),
            distribution=prrng.delta,
            parameters=[1.0],
            align=prrng.alignment(margin=10, buffer=5),
        )
        self.chunk.data -= 49.5

        self.k_interactions = 0.12
        self.system = FrictionQPotSpringBlock.Line1d.System_Cuspy_LongRange(
            m=1,
            eta=1,
            mu=1,
            k_interactions=self.k_interactions,
            k_frame=0.1,
            dt=1,
            chunk=self.chunk,
            alpha=1,
        )

    def test_interactions(self):
        N = self.N
        dp = np.arange(N)
        dn = np.arange(N)[::-1] + 1
        d = np.where(dp < dn, dp, dn)

        x = np.zeros_like(self.system.u)
        x[0] = 1
        self.system.u = x

        f = np.zeros_like(x)
        for j in range(1, N):
            f[j] = self.k_interactions * (x[0] - x[j]) / (d[j] ** 2)
        f[0] = -np.sum(f)

        for i in range(N):
            self.system.u = np.roll(x, i)
            self.assertTrue(np.allclose(np.roll(f, i), self.system.f_interactions))

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

        system = FrictionQPotSpringBlock.Line1d.System_Cuspy_LongRange(
            m=1.0,
            eta=1.0,
            mu=1.0,
            k_interactions=1.0,
            k_frame=0.1,
            dt=1.0,
            chunk=chunk,
            alpha=1,
        )

        self.assertLess(system.residual(), 1e-5)

        i_n = chunk.index_at_align
        system.eventDrivenStep(0.2, False)
        self.assertLess(system.residual(), 1e-5)
        self.assertTrue(np.allclose(system.u, (0.5 - 0.1) * np.ones(N)))
        self.assertTrue(np.all(chunk.index_at_align == i_n))
        self.assertAlmostEqual(system.u_frame, (0.5 - 0.1) * (1.0 + 0.1) / 0.1)

        i_n = chunk.index_at_align
        system.eventDrivenStep(0.2, True)
        self.assertTrue(np.allclose(system.u, (0.5 + 0.1) * np.ones(N)))
        self.assertTrue(not np.all(chunk.index_at_align == i_n))
        self.assertAlmostEqual(system.u_frame, (0.5 + 0.1) * (1.0 + 0.1) / 0.1)

        i_n = chunk.index_at_align
        system.eventDrivenStep(0.2, False)
        self.assertTrue(np.allclose(system.u, (1.5 - 0.1) * np.ones(N)))
        self.assertTrue(np.all(chunk.index_at_align == i_n))
        self.assertAlmostEqual(system.u_frame, (1.5 - 0.1) * (1.0 + 0.1) / 0.1)

        i_n = chunk.index_at_align
        system.eventDrivenStep(0.2, True)
        self.assertTrue(np.allclose(system.u, (1.5 + 0.1) * np.ones(N)))
        self.assertTrue(not np.all(chunk.index_at_align == i_n))
        self.assertAlmostEqual(system.u_frame, (1.5 + 0.1) * (1.0 + 0.1) / 0.1)


class Test_System_Cuspy_Laplace_RandomForcing(unittest.TestCase):
    def test_interactions(self):
        N = 10
        chunk = prrng.pcg32_tensor_cumsum_1_1(
            shape=[100],
            initstate=np.zeros([N], dtype=int),
            initseq=np.zeros([N], dtype=int),
            distribution=prrng.delta,
            parameters=[1.0],
            align=prrng.alignment(margin=10, buffer=5),
        )
        chunk.data -= 49.5

        system = FrictionQPotSpringBlock.Line1d.System_Cuspy_Laplace_RandomForcing(
            m=1,
            eta=1,
            mu=1,
            k_interactions=1,
            k_frame=0.1,
            dt=1,
            chunk=chunk,
            mean=0,
            stddev=1,
            seed=0,
            dinc_init=np.ones(N, dtype=int),
            dinc=np.ones(N, dtype=int),
        )
        self.assertLess(system.residual(), 1e-5)

        gen = prrng.pcg32(0)

        system.inc += 1
        system.refresh()
        self.assertGreater(system.residual(), 1e-5)
        self.assertTrue(np.allclose(system.external.f_thermal, gen.normal([system.size], 0, 1)))

        system.inc += 1
        system.refresh()
        self.assertGreater(system.residual(), 1e-5)
        self.assertTrue(np.allclose(system.external.f_thermal, gen.normal([system.size], 0, 1)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
