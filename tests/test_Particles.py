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
        deps = FrictionQPotSpringBlock.Particles.version_dependencies()
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
        eta = float(np.random.random(1)[0])
        mu = float(np.random.random(1)[0])
        k_frame = float(np.random.random(1)[0])
        par = dict(
            m=1,
            eta=eta,
            mu=mu,
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
            FrictionQPotSpringBlock.Particles.System_Cuspy(**par),
            FrictionQPotSpringBlock.Particles.System_SemiSmooth(kappa=1, **par),
            FrictionQPotSpringBlock.Particles.System_Smooth(**par),
            FrictionQPotSpringBlock.Particles.System_Cuspy_RandomForcing(**par, **rand),
        ]

        # by construction u = 0 which is a local minimum in all potentials
        for system in systems:
            self.assertLess(system.residual, 1e-5)
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


class Test_System_Cuspy(unittest.TestCase):
    def test_forces(self):
        N = 5
        eta = float(np.random.random(1)[0])
        mu = float(np.random.random(1)[0])
        k_frame = float(np.random.random(1)[0])

        system = FrictionQPotSpringBlock.Particles.System_Cuspy(
            m=1.0,
            eta=eta,
            mu=mu,
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
        self.assertLess(system.residual, 1e-5)
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
        du[0] = float(np.random.random(1)[0])
        dv[2] = float(np.random.random(1)[0])
        system.u += du
        system.v += dv
        umin = np.floor(du[0] + 0.5)

        f_potential = mu * np.array([umin - du[0], 0, 0, 0, 0])
        f_frame = k_frame * np.array([-du[0], 0, 0, 0, 0])
        f_damping = eta * np.array([0, 0, -dv[2], 0, 0])

        self.assertTrue(np.all(system.chunk.right_of_align > system.u))
        self.assertTrue(np.all(system.chunk.left_of_align <= system.u))
        self.assertTrue(np.allclose(system.f_potential, f_potential))
        self.assertTrue(np.allclose(system.f_frame, f_frame))
        self.assertTrue(np.allclose(system.f_interactions, np.zeros_like(system.f_interactions)))
        self.assertTrue(np.allclose(system.f_damping, f_damping))
        self.assertTrue(np.allclose(system.f, f_potential + f_frame + f_damping))

        du = np.zeros(N)
        dv = np.zeros(N)
        du[1] = 2.0 * float(np.random.random(1)[0])
        dv[3] = 2.0 * float(np.random.random(1)[0])
        system.u += du
        system.v += dv
        u = system.u
        v = system.v

        f_potential = mu * np.array(
            [np.floor(u[0] + 0.5) - u[0], np.floor(u[1] + 0.5) - u[1], 0, 0, 0]
        )
        f_frame = k_frame * np.array([-u[0], -u[1], 0, 0, 0])
        f_damping = eta * np.array([0, 0, -v[2], -v[3], 0])

        self.assertTrue(np.all(system.chunk.right_of_align > system.u))
        self.assertTrue(np.all(system.chunk.left_of_align <= system.u))
        self.assertTrue(np.allclose(system.f_potential, f_potential))
        self.assertTrue(np.allclose(system.f_frame, f_frame))
        self.assertTrue(np.allclose(system.f_interactions, np.zeros_like(system.f_interactions)))
        self.assertTrue(np.allclose(system.f_damping, f_damping))
        self.assertTrue(np.allclose(system.f, f_potential + f_frame + f_damping))

    def test_eventDrivenStep(self):
        N = 3
        system = FrictionQPotSpringBlock.Particles.System_Cuspy(
            m=1.0,
            eta=1.0,
            mu=1.0,
            k_frame=0.1,
            dt=1.0,
            shape=[N],
            seed=0,
            distribution="delta",
            parameters=[1.0],
            offset=-49.5,
            nchunk=100,
        )
        self.assertLess(system.residual, 1e-5)

        i_n = system.chunk.index_at_align
        system.eventDrivenStep(0.2, False)
        self.assertLess(system.residual, 1e-5)
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
        system = FrictionQPotSpringBlock.Particles.System_Cuspy(
            m=1.0,
            eta=1.0,
            mu=1.0,
            k_frame=0.1,
            dt=1.0,
            shape=[N],
            seed=0,
            distribution="delta",
            parameters=[1.0],
            offset=-49.5,
            nchunk=100,
        )

        i_n = system.chunk.index_at_align.copy()
        system.trigger(0, 0.2)

        u = np.zeros(N)
        u[0] = 0.5 + 0.1
        self.assertTrue(np.allclose(system.u, u))

        ret = system.minimise_truncate(i_n=i_n, A_truncate=1)
        self.assertTrue(np.sum(system.chunk.index_at_align != i_n) >= 1)
        self.assertGreater(ret, 0)
        self.assertGreater(system.residual, 1e-5)

    def test_advanceToFixedForce(self):
        N = 3
        system = FrictionQPotSpringBlock.Particles.System_Cuspy(
            m=1.0,
            eta=1.0,
            mu=1.0,
            k_frame=0.1,
            dt=1.0,
            shape=[N],
            seed=0,
            distribution="delta",
            parameters=[1.0],
            offset=-49.5,
            nchunk=100,
        )
        self.assertLess(system.residual, 1e-5)
        system.advanceToFixedForce(0.1)
        self.assertAlmostEqual(np.mean(system.f_frame), 0.1)
        self.assertLess(system.residual, 1e-5)

        self.assertLess(system.residual, 1e-5)
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
        mu = float(np.random.random(1)[0])
        system = FrictionQPotSpringBlock.Particles.System_Cuspy(
            m=1.0,
            eta=1.0,
            mu=mu,
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

                umin = 0.5 * (yref[r, j] + yref[r, j + 1])
                self.assertTrue(np.allclose(mu * (umin - system.u), system.f_potential))

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


class Test_System_SemiSmooth(unittest.TestCase):
    def test_eventDrivenStep(self):
        N = 3
        mu = 1
        kappa = 0.1

        system = FrictionQPotSpringBlock.Particles.System_SemiSmooth(
            m=1.0,
            eta=1.0,
            mu=mu,
            kappa=kappa,
            k_frame=0.1,
            dt=1.0,
            shape=[N],
            seed=0,
            distribution="delta",
            parameters=[1.0],
            offset=-49.5,
            nchunk=100,
        )
        self.assertLess(system.residual, 1e-5)

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
        self.assertLess(system.residual, 1e-5)
        self.assertTrue(np.allclose(system.u, upper - 0.5 * eps))
        self.assertAlmostEqual(system.maxUniformDisplacement(), 0.5 * eps)

        system.eventDrivenStep(eps=eps, kick=True)
        self.assertGreater(system.residual, 1e-5)
        self.assertTrue(np.allclose(system.u, upper + 0.5 * eps))
        self.assertAlmostEqual(system.maxUniformDisplacement(), 0)

        system.u = u0
        system.u_frame = uf0

        self.assertAlmostEqual(system.maxUniformDisplacement(-1), np.min(system.u - lower))
        system.eventDrivenStep(eps=eps, kick=False, direction=-1)
        self.assertLess(system.residual, 1e-5)
        self.assertTrue(np.allclose(system.u, lower + 0.5 * eps))
        self.assertAlmostEqual(system.maxUniformDisplacement(-1), 0.5 * eps)

        system.eventDrivenStep(eps=eps, kick=True, direction=-1)
        self.assertGreater(system.residual, 1e-5)
        self.assertTrue(np.allclose(system.u, lower - 0.5 * eps))
        self.assertAlmostEqual(system.maxUniformDisplacement(), 0)


class Test_System_Cuspy_RandomForcing(unittest.TestCase):
    def test_interactions(self):
        N = 10
        system = FrictionQPotSpringBlock.Particles.System_Cuspy_RandomForcing(
            m=1,
            eta=1,
            mu=1,
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
        self.assertLess(system.residual, 1e-5)

        gen = prrng.pcg32(0)

        system.inc += 1
        system.refresh()
        self.assertGreater(system.residual, 1e-5)
        self.assertTrue(np.allclose(system.external.f_thermal, gen.normal([system.size], 0, 1)))

        system.inc += 1
        system.refresh()
        self.assertGreater(system.residual, 1e-5)
        self.assertTrue(np.allclose(system.external.f_thermal, gen.normal([system.size], 0, 1)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
