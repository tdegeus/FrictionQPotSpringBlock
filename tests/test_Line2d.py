import faulthandler
import unittest

import FrictionQPotSpringBlock
import numpy as np
import prrng

faulthandler.enable()


class Test_support(unittest.TestCase):
    def test_version_dependencies(self):
        deps = FrictionQPotSpringBlock.Line2d.version_dependencies()
        deps = [i.split("=")[0] for i in deps]

        self.assertTrue("prrng" in deps)
        self.assertTrue("xtensor" in deps)
        self.assertTrue("xtensor-python" in deps)
        self.assertTrue("xtl" in deps)


class Test_System_Cuspy_Laplace(unittest.TestCase):
    def test_interactions(self):
        rows = 5
        cols = 4
        chunk = prrng.pcg32_tensor_cumsum_2_1(
            shape=[100],
            initstate=np.zeros([rows, cols], dtype=int),
            initseq=np.zeros([rows, cols], dtype=int),
            distribution=prrng.delta,
            parameters=[1.0],
            align=prrng.alignment(margin=10, buffer=5),
        )
        chunk.data -= 49.5

        k_interactions = float(np.random.random(1))
        system = FrictionQPotSpringBlock.Line2d.System_Cuspy_Laplace(
            m=1,
            eta=1,
            mu=1,
            k_interactions=k_interactions,
            k_frame=0.1,
            dt=1,
            chunk=chunk,
        )
        self.assertLess(system.residual(), 1e-5)
        self.assertEqual(list(system.shape), [rows, cols])
        self.assertEqual(system.size, rows * cols)

        c = -4
        f0 = (
            np.array([[c, 1, 0, 1], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]])
            * k_interactions
        )
        u0 = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

        for i in range(rows):
            for j in range(cols):
                x = np.roll(np.roll(u0, i, axis=0), j, axis=1)
                system.u = x
                f = np.roll(np.roll(f0, i, axis=0), j, axis=1)
                self.assertTrue(np.allclose(system.f_interactions, f))
                self.assertTrue(np.allclose(system.u, x))


class Test_System_Cuspy_QuarticGradient(unittest.TestCase):
    def test_interactions_basic(self):
        rows = 5
        cols = 4
        chunk = prrng.pcg32_tensor_cumsum_2_1(
            shape=[100],
            initstate=np.zeros([rows, cols], dtype=int),
            initseq=np.zeros([rows, cols], dtype=int),
            distribution=prrng.delta,
            parameters=[1.0],
            align=prrng.alignment(margin=10, buffer=5),
        )
        chunk.data -= 49.5

        k2 = 0.12
        k4 = 0.0
        system = FrictionQPotSpringBlock.Line2d.System_Cuspy_QuarticGradient(
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

        c = -4
        f0 = np.array([[c, 1, 0, 1], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]]) * k2
        u0 = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

        for i in range(rows):
            for j in range(cols):
                x = np.roll(np.roll(u0, i, axis=0), j, axis=1)
                system.u = x
                f = np.roll(np.roll(f0, i, axis=0), j, axis=1)
                self.assertTrue(np.allclose(system.f_interactions, f))
                self.assertTrue(np.allclose(system.u, x))

    def test_interactions(self):
        rows = 5
        cols = 5
        chunk = prrng.pcg32_tensor_cumsum_2_1(
            shape=[100],
            initstate=np.zeros([rows, cols], dtype=int),
            initseq=np.zeros([rows, cols], dtype=int),
            distribution=prrng.delta,
            parameters=[1.0],
            align=prrng.alignment(margin=10, buffer=5),
        )
        chunk.data -= 49.5

        k2 = 0.12
        k4 = 0.34
        system = FrictionQPotSpringBlock.Line2d.System_Cuspy_QuarticGradient(
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

        u0 = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        )

        c = -2
        d2udx2 = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 1, c, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        d2udy2 = d2udx2.T

        p = 0.5
        c = -0.5
        dudx = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, p, 0, c, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        dudy = dudx.T

        c = -0.25
        d2udxdy = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        )

        f0 = (d2udx2 + d2udy2) * (k2 + k4 / 3) + 2 / 3 * k4 * (
            dudx**2 * d2udx2 + dudy**2 * d2udy2 + 2 * dudx * dudy * d2udxdy
        )

        for i in range(rows):
            for j in range(cols):
                x = np.roll(np.roll(u0, i, axis=0), j, axis=1)
                system.u = x
                f = np.roll(np.roll(f0, i, axis=0), j, axis=1)
                self.assertTrue(np.allclose(system.f_interactions, f))


if __name__ == "__main__":
    unittest.main(verbosity=2)
