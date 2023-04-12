import faulthandler
import time
import unittest

import FrictionQPotSpringBlock
import numpy as np
import prrng

faulthandler.enable()
seed = int(time.time())
np.random.seed(seed)


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
            shape=[N],
            seed=0,
            distribution="delta",
            parameters=[1.0],
            offset=-49.5,
            nchunk=100,
        )
        rand = dict(
            mean=0,
            stddev=1,
            seed_forcing=0,
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
            self.assertTrue(
                np.all(system.chunk.index_at_align + 1 == np.argmax(system.chunk.data[0, :] > 0))
            )
            self.assertTrue(np.all(system.chunk.right_of_align > 0))
            self.assertTrue(np.all(system.chunk.left_of_align <= 0))


class Test_System_Cuspy_Laplace(unittest.TestCase):
    def test_forces(self):
        N = 5
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
            shape=[N],
            seed=0,
            distribution="delta",
            parameters=[1.0],
            offset=-49.5,
            nchunk=100,
        )

        # by construction u = 0 which is a local minimum in all potentials
        self.assertLess(system.residual(), 1e-5)
        self.assertTrue(np.allclose(system.f, 0.0))
        self.assertTrue(np.allclose(system.f_potential, 0.0))
        self.assertTrue(np.allclose(system.f_frame, 0.0))
        self.assertTrue(np.allclose(system.f_interactions, 0.0))
        self.assertTrue(np.allclose(system.f_damping, 0.0))
        self.assertTrue(
            np.all(system.chunk.index_at_align + 1 == np.argmax(system.chunk.data[0, :] > 0))
        )
        self.assertTrue(np.all(system.chunk.right_of_align > 0))
        self.assertTrue(np.all(system.chunk.left_of_align <= 0))

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

        self.assertTrue(np.all(system.chunk.right_of_align > system.u))
        self.assertTrue(np.all(system.chunk.left_of_align <= system.u))
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

        self.assertTrue(np.all(system.chunk.right_of_align > system.u))
        self.assertTrue(np.all(system.chunk.left_of_align <= system.u))
        self.assertTrue(np.allclose(system.f_potential, f_potential))
        self.assertTrue(np.allclose(system.f_frame, f_frame))
        self.assertTrue(np.allclose(system.f_interactions, f_interactions))
        self.assertTrue(np.allclose(system.f_damping, f_damping))
        self.assertTrue(np.allclose(system.f, f_potential + f_frame + f_interactions + f_damping))

    def test_eventDrivenStep(self):
        N = 3
        system = FrictionQPotSpringBlock.Line1d.System_Cuspy_Laplace(
            m=1.0,
            eta=1.0,
            mu=1.0,
            k_interactions=1.0,
            k_frame=0.1,
            dt=1.0,
            shape=[N],
            seed=0,
            distribution="delta",
            parameters=[1.0],
            offset=-49.5,
            nchunk=100,
        )
        self.assertLess(system.residual(), 1e-5)

        i_n = system.chunk.index_at_align
        system.eventDrivenStep(0.2, False)
        self.assertLess(system.residual(), 1e-5)
        self.assertTrue(np.allclose(system.u, (0.5 - 0.1) * np.ones(N)))
        self.assertTrue(np.all(system.chunk.index_at_align == i_n))
        self.assertAlmostEqual(system.u_frame, (0.5 - 0.1) * (1.0 + 0.1) / 0.1)

        i_n = system.chunk.index_at_align
        system.eventDrivenStep(0.2, True)
        self.assertTrue(np.allclose(system.u, (0.5 + 0.1) * np.ones(N)))
        self.assertTrue(not np.all(system.chunk.index_at_align == i_n))
        self.assertAlmostEqual(system.u_frame, (0.5 + 0.1) * (1.0 + 0.1) / 0.1)

        i_n = system.chunk.index_at_align
        system.eventDrivenStep(0.2, False)
        self.assertTrue(np.allclose(system.u, (1.5 - 0.1) * np.ones(N)))
        self.assertTrue(np.all(system.chunk.index_at_align == i_n))
        self.assertAlmostEqual(system.u_frame, (1.5 - 0.1) * (1.0 + 0.1) / 0.1)

        i_n = system.chunk.index_at_align
        system.eventDrivenStep(0.2, True)
        self.assertTrue(np.allclose(system.u, (1.5 + 0.1) * np.ones(N)))
        self.assertTrue(not np.all(system.chunk.index_at_align == i_n))
        self.assertAlmostEqual(system.u_frame, (1.5 + 0.1) * (1.0 + 0.1) / 0.1)

    def test_trigger(self):
        N = 3
        system = FrictionQPotSpringBlock.Line1d.System_Cuspy_Laplace(
            m=1.0,
            eta=1.0,
            mu=1.0,
            k_interactions=1.0,
            k_frame=0.1,
            dt=1.0,
            shape=[N],
            seed=0,
            distribution="delta",
            parameters=[1.0],
            offset=-49.5,
            nchunk=100,
        )

        system.trigger(0, 0.2)

        u = np.zeros(N)
        u[0] = 0.5 + 0.1
        self.assertTrue(np.allclose(system.u, u))

    def test_advanceToFixedForce(self):
        N = 3
        system = FrictionQPotSpringBlock.Line1d.System_Cuspy_Laplace(
            m=1.0,
            eta=1.0,
            mu=1.0,
            k_interactions=1.0,
            k_frame=0.1,
            dt=1.0,
            shape=[N],
            seed=0,
            distribution="delta",
            parameters=[1.0],
            offset=-49.5,
            nchunk=100,
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
        init_offset = 50.0

        # draw reference yield positions
        gen = prrng.pcg32_array(initstate, np.zeros_like(initstate))
        yref = np.cumsum(1e-3 + 1.1 * gen.weibull([20000], 2.0), axis=1) - init_offset

        # chunked storage of "yref" (same seed)
        system = FrictionQPotSpringBlock.Line1d.System_Cuspy_Laplace(
            m=1.0,
            eta=1.0,
            mu=1.0,
            k_interactions=1.0,
            k_frame=0.1,
            dt=1.0,
            shape=[N],
            seed=seed,
            distribution="weibull",
            parameters=[2.0, 1.1, 1e-3],
            offset=-init_offset,
            nchunk=100,
        )

        du = 10.0 * np.ones(N)
        du[0] = 5.0
        du[1] = 7.0

        u = np.copy(system.u)
        start = np.copy(system.chunk.start)
        state = np.copy(system.chunk.state_at(start))
        value = np.copy(system.chunk.data[..., 0])

        for repeat in range(3):
            if repeat >= 1:
                system.chunk.restore(state=state, value=value, index=start)
                system.u = u

            for i in list(range(1500)) + list(range(1500))[::-1]:
                system.u = i * du

                j = prrng.lower_bound(yref, system.u)
                r = np.arange(N)

                self.assertTrue(np.all(system.chunk.index_at_align == j))
                self.assertTrue(np.allclose(yref[r, j], system.chunk.left_of_align))
                self.assertTrue(np.allclose(yref[r, j + 1], system.chunk.right_of_align))

                if repeat == 0 and i == 500:
                    u = np.copy(system.u)
                    start = np.copy(system.chunk.start)
                    state = np.copy(system.chunk.state_at(start))
                    value = np.copy(system.chunk.data[..., 0])

                if repeat == 1 and i == 1000:
                    u = np.copy(system.u)
                    start = np.copy(system.chunk.start)
                    state = np.copy(system.chunk.state_at(start))
                    value = np.copy(system.chunk.data[..., 0])


class Test_System_SemiSmooth_Laplace(unittest.TestCase):
    def test_eventDrivenStep(self):
        N = 3
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
            shape=[N],
            seed=0,
            distribution="delta",
            parameters=[1.0],
            offset=-49.5,
            nchunk=100,
        )
        self.assertLess(system.residual(), 1e-5)

        u0 = system.u.copy()
        uf0 = system.u_frame
        left = system.chunk.left_of_align
        right = system.chunk.right_of_align
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


class Test_System_Cuspy_Quartic(unittest.TestCase):
    def test_interactions(self):
        N = 10
        a1 = float(np.random.random(1))
        a2 = float(np.random.random(1))
        system = FrictionQPotSpringBlock.Line1d.System_Cuspy_Quartic(
            m=1,
            eta=1,
            mu=1,
            a1=a1,
            a2=a2,
            k_frame=0.1,
            dt=1,
            shape=[N],
            seed=0,
            distribution="delta",
            parameters=[1.0],
            offset=-49.5,
            nchunk=100,
        )

        self.assertLess(system.residual(), 1e-5)

        du = float(np.random.random(1))
        u0 = np.array([du, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        laplace = np.array([-2 * du, du, 0, 0, 0, 0, 0, 0, 0, du])
        du_p = np.array([-du, 0, 0, 0, 0, 0, 0, 0, 0, du])
        du_n = np.array([-du, du, 0, 0, 0, 0, 0, 0, 0, 0])

        f0 = a1 * laplace + a2 * (du_p**3 + du_n**3)

        for i in range(N):
            u = np.roll(u0, i)
            system.u = u
            f = np.roll(f0, i)
            self.assertTrue(np.allclose(system.f_interactions, f))
            self.assertTrue(np.allclose(system.u, u))


class Test_System_Cuspy_QuarticGradient(unittest.TestCase):
    def test_interactions(self):
        N = 10
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
            shape=[N],
            seed=0,
            distribution="delta",
            parameters=[1.0],
            offset=-49.5,
            nchunk=100,
        )

        self.assertLess(system.residual(), 1e-5)

        u0 = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        laplace = np.array([-2, 1, 0, 0, 0, 0, 0, 0, 0, 1])
        gradient = np.array([0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0.5])

        f0 = k2 * laplace + k4 * laplace * gradient**2

        for i in range(N):
            u = np.roll(u0, i)
            system.u = u
            f = np.roll(f0, i)
            self.assertTrue(np.allclose(system.f_interactions, f))
            self.assertTrue(np.allclose(system.u, u))


class Test_System_Cuspy_LongRange(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        N = 10
        self.N = N
        self.k_interactions = 0.12
        self.system = FrictionQPotSpringBlock.Line1d.System_Cuspy_LongRange(
            m=1,
            eta=1,
            mu=1,
            k_interactions=self.k_interactions,
            k_frame=0.1,
            dt=1,
            alpha=1,
            shape=[N],
            seed=0,
            distribution="delta",
            parameters=[1.0],
            offset=-49.5,
            nchunk=100,
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
        system = FrictionQPotSpringBlock.Line1d.System_Cuspy_LongRange(
            m=1.0,
            eta=1.0,
            mu=1.0,
            k_interactions=1.0,
            k_frame=0.1,
            dt=1.0,
            alpha=1,
            shape=[N],
            seed=0,
            distribution="delta",
            parameters=[1.0],
            offset=-49.5,
            nchunk=100,
        )

        self.assertLess(system.residual(), 1e-5)

        i_n = system.chunk.index_at_align
        system.eventDrivenStep(0.2, False)
        self.assertLess(system.residual(), 1e-5)
        self.assertTrue(np.allclose(system.u, (0.5 - 0.1) * np.ones(N)))
        self.assertTrue(np.all(system.chunk.index_at_align == i_n))
        self.assertAlmostEqual(system.u_frame, (0.5 - 0.1) * (1.0 + 0.1) / 0.1)

        i_n = system.chunk.index_at_align
        system.eventDrivenStep(0.2, True)
        self.assertTrue(np.allclose(system.u, (0.5 + 0.1) * np.ones(N)))
        self.assertTrue(not np.all(system.chunk.index_at_align == i_n))
        self.assertAlmostEqual(system.u_frame, (0.5 + 0.1) * (1.0 + 0.1) / 0.1)

        i_n = system.chunk.index_at_align
        system.eventDrivenStep(0.2, False)
        self.assertTrue(np.allclose(system.u, (1.5 - 0.1) * np.ones(N)))
        self.assertTrue(np.all(system.chunk.index_at_align == i_n))
        self.assertAlmostEqual(system.u_frame, (1.5 - 0.1) * (1.0 + 0.1) / 0.1)

        i_n = system.chunk.index_at_align
        system.eventDrivenStep(0.2, True)
        self.assertTrue(np.allclose(system.u, (1.5 + 0.1) * np.ones(N)))
        self.assertTrue(not np.all(system.chunk.index_at_align == i_n))
        self.assertAlmostEqual(system.u_frame, (1.5 + 0.1) * (1.0 + 0.1) / 0.1)


class Test_System_Cuspy_Laplace_RandomForcing(unittest.TestCase):
    def test_interactions(self):
        N = 10
        system = FrictionQPotSpringBlock.Line1d.System_Cuspy_Laplace_RandomForcing(
            m=1,
            eta=1,
            mu=1,
            k_interactions=1,
            k_frame=0.1,
            dt=1,
            mean=0,
            stddev=1,
            seed_forcing=0,
            dinc_init=np.ones(N, dtype=int),
            dinc=np.ones(N, dtype=int),
            shape=[N],
            seed=0,
            distribution="delta",
            parameters=[1.0],
            offset=-49.5,
            nchunk=100,
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
