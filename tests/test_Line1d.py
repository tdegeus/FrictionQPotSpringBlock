import faulthandler
import time
import unittest

import FrictionQPotSpringBlock
import numpy as np
import prrng

faulthandler.enable()


class Test_Line1d_System_Cuspy_Laplace(unittest.TestCase):
    """
    Test Line1d.System_Cuspy_Laplace
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

        self.assertTrue(system.residual() < 1e-5)
        self.assertTrue(np.allclose(system.f, 0.0))
        self.assertTrue(np.allclose(system.f_potential, 0.0))
        self.assertTrue(np.allclose(system.f_frame, 0.0))
        self.assertTrue(np.allclose(system.f_interactions, 0.0))
        self.assertTrue(np.allclose(system.f_damping, 0.0))
        self.assertTrue(np.all(np.equal(chunk.target_index + 1, np.argmax(chunk.data[0, :] > 0))))
        self.assertTrue(np.all(chunk.right_of_target > 0))
        self.assertTrue(np.all(chunk.left_of_target <= 0))

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

        self.assertTrue(np.all(chunk.right_of_target > system.u))
        self.assertTrue(np.all(chunk.left_of_target <= system.u))
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

        self.assertTrue(np.all(chunk.right_of_target > system.u))
        self.assertTrue(np.all(chunk.left_of_target <= system.u))
        self.assertTrue(np.allclose(system.f_potential, f_potential))
        self.assertTrue(np.allclose(system.f_frame, f_frame))
        self.assertTrue(np.allclose(system.f_interactions, f_interactions))
        self.assertTrue(np.allclose(system.f_damping, f_damping))
        self.assertTrue(np.allclose(system.f, f_potential + f_frame + f_interactions + f_damping))

#     def test_eventDrivenStep(self):

#         N = 3
#         chunk = prrng.pcg32_tensor_cumsum_1_1(
#             shape=[100],
#             initstate=np.zeros([N], dtype=int),
#             initseq=np.zeros([N], dtype=int),
#             distribution=prrng.delta,
#             parameters=[1.0],
#             align=prrng.alignment(margin=10, buffer=5),
#         )
#         chunk.data -= 49.5

#         system = FrictionQPotSpringBlock.Line1d.System(
#             m=1.0,
#             eta=1.0,
#             mu=1.0,
#             k_interactions=1.0,
#             k_frame=0.1,
#             dt=1.0,
#             chunk=chunk,
#         )

#         self.assertTrue(system.residual() < 1e-5)

#         i_n = system.i
#         system.eventDrivenStep(0.2, False)
#         self.assertTrue(system.residual() < 1e-5)
#         self.assertTrue(np.allclose(system.u, (0.5 - 0.1) * np.ones(N)))
#         self.assertTrue(np.all(system.i == i_n))
#         self.assertTrue(np.isclose(system.u_frame, (0.5 - 0.1) * (1.0 + 0.1) / 0.1))

#         i_n = system.i
#         system.eventDrivenStep(0.2, True)
#         self.assertTrue(np.allclose(system.u, (0.5 + 0.1) * np.ones(N)))
#         self.assertTrue(not np.all(system.i == i_n))
#         self.assertTrue(np.isclose(system.u_frame, (0.5 + 0.1) * (1.0 + 0.1) / 0.1))

#         i_n = system.i
#         system.eventDrivenStep(0.2, False)
#         self.assertTrue(np.allclose(system.u, (1.5 - 0.1) * np.ones(N)))
#         self.assertTrue(np.all(system.i == i_n))
#         self.assertTrue(np.isclose(system.u_frame, (1.5 - 0.1) * (1.0 + 0.1) / 0.1))

#         i_n = system.i
#         system.eventDrivenStep(0.2, True)
#         self.assertTrue(np.allclose(system.u, (1.5 + 0.1) * np.ones(N)))
#         self.assertTrue(not np.all(system.i == i_n))
#         self.assertTrue(np.isclose(system.u_frame, (1.5 + 0.1) * (1.0 + 0.1) / 0.1))

#     def test_trigger(self):

#         N = 3
#         chunk = prrng.pcg32_tensor_cumsum_1_1(
#             shape=[100],
#             initstate=np.zeros([N], dtype=int),
#             initseq=np.zeros([N], dtype=int),
#             distribution=prrng.delta,
#             parameters=[1.0],
#             align=prrng.alignment(margin=10, buffer=5),
#         )
#         chunk.data -= 49.5

#         system = FrictionQPotSpringBlock.Line1d.System(
#             m=1.0,
#             eta=1.0,
#             mu=1.0,
#             k_interactions=1.0,
#             k_frame=0.1,
#             dt=1.0,
#             chunk=chunk,
#         )

#         system.trigger(0, 0.2)

#         x = np.zeros(N)
#         x[0] = 0.5 + 0.1
#         self.assertTrue(np.allclose(system.u, x))

#     def test_advanceToFixedForce(self):

#         N = 3
#         chunk = prrng.pcg32_tensor_cumsum_1_1(
#             shape=[100],
#             initstate=np.zeros([N], dtype=int),
#             initseq=np.zeros([N], dtype=int),
#             distribution=prrng.delta,
#             parameters=[1.0],
#             align=prrng.alignment(margin=10, buffer=5),
#         )
#         chunk.data -= 49.5

#         system = FrictionQPotSpringBlock.Line1d.System(
#             m=1.0,
#             eta=1.0,
#             mu=1.0,
#             k_interactions=1.0,
#             k_frame=0.1,
#             dt=1.0,
#             chunk=chunk,
#         )

#         self.assertTrue(system.residual() < 1e-5)
#         system.advanceToFixedForce(0.1)
#         self.assertTrue(np.isclose(np.mean(system.f_frame), 0.1))
#         self.assertTrue(system.residual() < 1e-5)

#         self.assertTrue(system.residual() < 1e-5)
#         system.advanceToFixedForce(0.0)
#         self.assertTrue(np.isclose(np.mean(system.f_frame), 0.0))
#         self.assertTrue(np.allclose(system.u, 0.0))
#         self.assertTrue(np.allclose(system.u_frame, 0.0))

#     def test_chunked(self):
#         """
#         Chunked sequence, shift optimally left
#         """

#         N = 3
#         seed = int(time.time())
#         initstate = seed + np.arange(N)

#         nchunk = 100  # size of chunk of yield positions kept in memory
#         buffer = 20  # redraw within this margin from the edges of the chunk
#         margin = 10  # position to place the particle after redraw
#         init_offset = 50.0  # initial negative position shift

#         # draw reference yield positions
#         gen = prrng.pcg32_array(initstate, np.zeros_like(initstate))
#         yref = np.cumsum(gen.random([2000]), axis=1) - init_offset

#         # chunked storage
#         align = prrng.alignment(margin=margin, buffer=buffer)
#         chunk = prrng.pcg32_tensor_cumsum_1_1(
#             shape=[nchunk],
#             initstate=initstate,
#             initseq=np.zeros_like(initstate),
#             distribution=prrng.random,
#             parameters=[],
#             align=align,
#         )
#         chunk -= init_offset

#         # initialise system
#         system = FrictionQPotSpringBlock.Line1d.System(
#             m=1.0,
#             eta=1.0,
#             mu=1.0,
#             k_interactions=1.0,
#             k_frame=0.1,
#             dt=1.0,
#             chunk=chunk,
#         )

#         x = 10.0 * np.ones(N)
#         x[0] = 5.0
#         x[1] = 7.0

#         for i in range(50):

#             system.u = i * x

#             j = prrng.lower_bound(yref, system.u)
#             r = np.arange(N)

#             self.assertTrue(np.all(system.i == j))
#             self.assertTrue(np.allclose(yref[r, system.i], system.y_left()))
#             self.assertTrue(np.allclose(yref[r, system.i + 1], system.y_right()))


# class Test_Line1d_SystemSemiSmooth(unittest.TestCase):
#     """
#     Test Line1d.SystemSemiSmooth
#     """

#     def test_eventDrivenStep(self):

#         N = 3
#         chunk = prrng.pcg32_tensor_cumsum_1_1(
#             shape=[100],
#             initstate=np.zeros([N], dtype=int),
#             initseq=np.zeros([N], dtype=int),
#             distribution=prrng.delta,
#             parameters=[1.0],
#             align=prrng.alignment(margin=10, buffer=5),
#         )
#         chunk.data -= 49.5

#         mu = 1
#         kappa = 0.1

#         system = FrictionQPotSpringBlock.Line1d.SystemSemiSmooth(
#             m=1.0,
#             eta=1.0,
#             mu=mu,
#             kappa=kappa,
#             k_interactions=1.0,
#             k_frame=0.1,
#             dt=1.0,
#             chunk=chunk,
#         )

#         self.assertTrue(system.residual() < 1e-5)

#         x0 = system.u.copy()
#         xf0 = system.u_frame
#         left = system.y_left()
#         right = system.y_right()
#         mid = 0.5 * (left + right)
#         upper = (mu * mid + kappa * right) / (mu + kappa)
#         lower = (mu * mid + kappa * left) / (mu + kappa)
#         eps = 0.001

#         self.assertAlmostEqual(system.maxUniformDisplacement(), np.min(upper - system.u))
#         system.eventDrivenStep(eps=eps, kick=False)
#         self.assertTrue(system.residual() < 1e-5)
#         self.assertTrue(np.allclose(system.u, upper - 0.5 * eps))
#         self.assertAlmostEqual(system.maxUniformDisplacement(), 0.5 * eps)

#         system.eventDrivenStep(eps=eps, kick=True)
#         self.assertFalse(system.residual() < 1e-5)
#         self.assertTrue(np.allclose(system.u, upper + 0.5 * eps))
#         self.assertAlmostEqual(system.maxUniformDisplacement(), 0)

#         system.u = x0
#         system.u_frame = xf0

#         self.assertAlmostEqual(system.maxUniformDisplacement(-1), np.min(system.u - lower))
#         system.eventDrivenStep(eps=eps, kick=False, direction=-1)
#         self.assertTrue(system.residual() < 1e-5)
#         self.assertTrue(np.allclose(system.u, lower + 0.5 * eps))
#         self.assertTrue(
#             np.isclose(system.maxUniformDisplacement(-1), -0.5 * eps, atol=1e-3, rtol=1e-3)
#         )

#         system.eventDrivenStep(eps=eps, kick=True, direction=-1)
#         self.assertFalse(system.residual() < 1e-5)
#         self.assertTrue(np.allclose(system.u, lower - 0.5 * eps))
#         self.assertAlmostEqual(system.maxUniformDisplacement(), 0)


# class Test_Line1d_SystemQuartic(unittest.TestCase):
#     """
#     Test Line1d.SystemQuartic
#     """

#     def test_interactions(self):

#         N = 10
#         chunk = prrng.pcg32_tensor_cumsum_1_1(
#             shape=[100],
#             initstate=np.zeros([N], dtype=int),
#             initseq=np.zeros([N], dtype=int),
#             distribution=prrng.delta,
#             parameters=[1.0],
#             align=prrng.alignment(margin=10, buffer=5),
#         )
#         chunk.data -= 49.5

#         k2 = 0.12
#         k4 = 0.34
#         system = FrictionQPotSpringBlock.Line1d.SystemQuartic(
#             m=1,
#             eta=1,
#             mu=1,
#             k2=k2,
#             k4=k4,
#             k_frame=0.1,
#             dt=1,
#             chunk=chunk,
#         )

#         self.assertTrue(system.residual() < 1e-5)

#         x0 = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
#         laplace = np.array([-2, 1, 0, 0, 0, 0, 0, 0, 0, 1])
#         gradient = np.array([0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0.5])

#         f0 = k2 * laplace + k4 * laplace * gradient**2

#         for i in range(N):
#             x = np.roll(x0, i)
#             system.u = x.ravel()
#             f = np.roll(f0, i)
#             self.assertTrue(np.allclose(system.f_interactions, f))
#             self.assertTrue(np.allclose(system.u, x))


# class Test_Line1d_SystemLongRange(unittest.TestCase):
#     """
#     Test Line1d.SystemLongRange
#     """

#     @classmethod
#     def setUpClass(self):

#         N = 10
#         self.N = N
#         self.chunk = prrng.pcg32_tensor_cumsum_1_1(
#             shape=[100],
#             initstate=np.zeros([N], dtype=int),
#             initseq=np.zeros([N], dtype=int),
#             distribution=prrng.delta,
#             parameters=[1.0],
#             align=prrng.alignment(margin=10, buffer=5),
#         )
#         self.chunk.data -= 49.5

#         self.k_interactions = 0.12
#         self.system = FrictionQPotSpringBlock.Line1d.SystemLongRange(
#             m=1,
#             eta=1,
#             mu=1,
#             k_interactions=self.k_interactions,
#             k_frame=0.1,
#             dt=1,
#             chunk=self.chunk,
#             alpha=1,
#         )

#     def test_periodic(self):

#         N = self.N
#         index = np.tile(np.arange(N).reshape(1, -1), [N, 1]).ravel()
#         for i in range(N):
#             for j in range(N):
#                 self.assertEqual(self.system.periodic(i + j), index[i + j])

#     def test_distance(self):

#         N = self.N
#         dp = np.arange(N)
#         dn = np.arange(N)[::-1] + 1
#         d = np.where(dp < dn, dp, dn)

#         for p in range(N):
#             self.assertTrue(np.all(self.system.distance(p) == np.roll(d, p)))

#     def test_interactions(self):

#         N = self.N
#         dp = np.arange(N)
#         dn = np.arange(N)[::-1] + 1
#         d = np.where(dp < dn, dp, dn)

#         x = np.zeros_like(self.system.u)
#         x[0] = 1
#         self.system.u = x

#         f = np.zeros_like(x)
#         for j in range(1, N):
#             f[j] = self.k_interactions * (x[0] - x[j]) / (d[j] ** 2)
#         f[0] = -np.sum(f)

#         for i in range(N):
#             self.system.u = np.roll(x, i)
#             self.assertTrue(np.allclose(np.roll(f, i), self.system.f_interactions))

#     def test_eventDrivenStep(self):

#         N = 3
#         chunk = prrng.pcg32_tensor_cumsum_1_1(
#             shape=[100],
#             initstate=np.zeros([N], dtype=int),
#             initseq=np.zeros([N], dtype=int),
#             distribution=prrng.delta,
#             parameters=[1.0],
#             align=prrng.alignment(margin=10, buffer=5),
#         )
#         chunk.data -= 49.5

#         system = FrictionQPotSpringBlock.Line1d.SystemLongRange(
#             m=1.0,
#             eta=1.0,
#             mu=1.0,
#             k_interactions=1.0,
#             k_frame=0.1,
#             dt=1.0,
#             chunk=chunk,
#             alpha=1,
#         )

#         self.assertTrue(system.residual() < 1e-5)

#         i_n = system.i
#         system.eventDrivenStep(0.2, False)
#         self.assertTrue(system.residual() < 1e-5)
#         self.assertTrue(np.allclose(system.u, (0.5 - 0.1) * np.ones(N)))
#         self.assertTrue(np.all(system.i == i_n))
#         self.assertTrue(np.isclose(system.u_frame, (0.5 - 0.1) * (1.0 + 0.1) / 0.1))

#         i_n = system.i
#         system.eventDrivenStep(0.2, True)
#         self.assertTrue(np.allclose(system.u, (0.5 + 0.1) * np.ones(N)))
#         self.assertTrue(not np.all(system.i == i_n))
#         self.assertTrue(np.isclose(system.u_frame, (0.5 + 0.1) * (1.0 + 0.1) / 0.1))

#         i_n = system.i
#         system.eventDrivenStep(0.2, False)
#         self.assertTrue(np.allclose(system.u, (1.5 - 0.1) * np.ones(N)))
#         self.assertTrue(np.all(system.i == i_n))
#         self.assertTrue(np.isclose(system.u_frame, (1.5 - 0.1) * (1.0 + 0.1) / 0.1))

#         i_n = system.i
#         system.eventDrivenStep(0.2, True)
#         self.assertTrue(np.allclose(system.u, (1.5 + 0.1) * np.ones(N)))
#         self.assertTrue(not np.all(system.i == i_n))
#         self.assertTrue(np.isclose(system.u_frame, (1.5 + 0.1) * (1.0 + 0.1) / 0.1))


# class Test_Line1d_System2d(unittest.TestCase):
#     """
#     Test Line1d.System2d
#     """

#     def test_interactions(self):

#         m = 4
#         n = 4
#         N = m * n
#         chunk = prrng.pcg32_tensor_cumsum_1_1(
#             shape=[100],
#             initstate=np.zeros([N], dtype=int),
#             initseq=np.zeros([N], dtype=int),
#             distribution=prrng.delta,
#             parameters=[1.0],
#             align=prrng.alignment(margin=10, buffer=5),
#         )
#         chunk.data -= 49.5

#         k_interactions = 0.12
#         system = FrictionQPotSpringBlock.Line1d.System2d(
#             m=1,
#             eta=1,
#             mu=1,
#             k_interactions=k_interactions,
#             k_frame=0.1,
#             dt=1,
#             chunk=chunk,
#             width=n,
#         )

#         self.assertTrue(system.residual() < 1e-5)

#         index = np.arange(N).reshape(m, n)
#         down = np.roll(index, -1, axis=0)
#         up = np.roll(index, 1, axis=0)
#         left = np.roll(index, 1, axis=1)
#         right = np.roll(index, -1, axis=1)

#         self.assertTrue(np.all(system.organisation == index))
#         self.assertTrue(np.all(system.down == down.ravel()))
#         self.assertTrue(np.all(system.up == up.ravel()))
#         self.assertTrue(np.all(system.left == left.ravel()))
#         self.assertTrue(np.all(system.right == right.ravel()))

#         c = -4
#         f0 = np.array([[c, 1, 0, 1], [1, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]]) * k_interactions
#         x0 = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

#         for i in range(m):
#             for j in range(n):
#                 x = np.roll(np.roll(x0, i, axis=0), j, axis=1)
#                 system.u = x.ravel()
#                 f = np.roll(np.roll(f0, i, axis=0), j, axis=1).ravel()
#                 self.assertTrue(np.allclose(system.f_interactions, f))
#                 self.assertTrue(np.allclose(system.u[index], x))


# class Test_Line1d_System2dQuartic(unittest.TestCase):
#     """
#     Test Line1d.System2dQuartic
#     """

#     def test_interactions_basic(self):

#         m = 4
#         n = 4
#         N = m * n
#         chunk = prrng.pcg32_tensor_cumsum_1_1(
#             shape=[100],
#             initstate=np.zeros([N], dtype=int),
#             initseq=np.zeros([N], dtype=int),
#             distribution=prrng.delta,
#             parameters=[1.0],
#             align=prrng.alignment(margin=10, buffer=5),
#         )
#         chunk.data -= 49.5

#         k2 = 0.12
#         k4 = 0.0
#         system = FrictionQPotSpringBlock.Line1d.System2dQuartic(
#             m=1,
#             eta=1,
#             mu=1,
#             k2=k2,
#             k4=k4,
#             k_frame=0.1,
#             dt=1,
#             chunk=chunk,
#             width=n,
#         )

#         self.assertTrue(system.residual() < 1e-5)

#         index = np.arange(N).reshape(m, n)
#         down = np.roll(index, -1, axis=0)
#         up = np.roll(index, 1, axis=0)
#         left = np.roll(index, 1, axis=1)
#         right = np.roll(index, -1, axis=1)

#         self.assertTrue(np.all(system.organisation == index))
#         self.assertTrue(np.all(system.down == down.ravel()))
#         self.assertTrue(np.all(system.up == up.ravel()))
#         self.assertTrue(np.all(system.left == left.ravel()))
#         self.assertTrue(np.all(system.right == right.ravel()))

#         c = -4
#         f0 = np.array([[c, 1, 0, 1], [1, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]]) * k2
#         x0 = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

#         for i in range(m):
#             for j in range(n):
#                 x = np.roll(np.roll(x0, i, axis=0), j, axis=1)
#                 system.u = x.ravel()
#                 f = np.roll(np.roll(f0, i, axis=0), j, axis=1).ravel()
#                 self.assertTrue(np.allclose(system.f_interactions, f))
#                 self.assertTrue(np.allclose(system.u[index], x))

#     def test_interactions(self):

#         m = 5
#         n = 5
#         N = m * n
#         chunk = prrng.pcg32_tensor_cumsum_1_1(
#             shape=[100],
#             initstate=np.zeros([N], dtype=int),
#             initseq=np.zeros([N], dtype=int),
#             distribution=prrng.delta,
#             parameters=[1.0],
#             align=prrng.alignment(margin=10, buffer=5),
#         )
#         chunk.data -= 49.5

#         k2 = 0.12
#         k4 = 0.34
#         system = FrictionQPotSpringBlock.Line1d.System2dQuartic(
#             m=1,
#             eta=1,
#             mu=1,
#             k2=k2,
#             k4=k4,
#             k_frame=0.1,
#             dt=1,
#             chunk=chunk,
#             width=n,
#         )

#         self.assertLess(system.residual(), 1e-5)

#         x0 = np.array(
#             [
#                 [0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0],
#                 [0, 0, 1, 0, 0],
#                 [0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0],
#             ]
#         )

#         c = -2
#         d2udx2 = np.array(
#             [
#                 [0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0],
#                 [0, 1, c, 1, 0],
#                 [0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0],
#             ]
#         )
#         d2udy2 = d2udx2.T

#         p = 0.5
#         c = -0.5
#         dudx = np.array(
#             [
#                 [0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0],
#                 [0, p, 0, c, 0],
#                 [0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0],
#             ]
#         )
#         dudy = dudx.T

#         c = -0.25
#         d2udxdy = np.array(
#             [
#                 [0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0],
#             ]
#         )

#         f0 = (d2udx2 + d2udy2) * (k2 + k4 / 3) + 2 / 3 * k4 * (
#             dudx**2 * d2udx2 + dudy**2 * d2udy2 + 2 * dudx * dudy * d2udxdy
#         )

#         for i in range(m):
#             for j in range(n):
#                 x = np.roll(np.roll(x0, i, axis=0), j, axis=1)
#                 system.u = x.ravel()
#                 f = np.roll(np.roll(f0, i, axis=0), j, axis=1).ravel()
#                 self.assertTrue(np.allclose(system.f_interactions, f))


if __name__ == "__main__":

    unittest.main(verbosity=2)
