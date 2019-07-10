import numpy as np
import unittest

from konvert.points import Azimuthal
from konvert.points import AzimuthalEquidistant
from konvert.points import Bipolar
from konvert.points import Cartesian1D
from konvert.points import Cartesian2D
from konvert.points import Cartesian3D
from konvert.points import Cylindrical
from konvert.points import degrees
from konvert.points import Equirectangular
from konvert.points import Equitorial
from konvert.points import Mercator
from konvert.points import OnPlane
from konvert.points import OnSphere
from konvert.points import Orthographic
from konvert.points import Points
from konvert.points import Polar
from konvert.points import Quaternion
from konvert.points import Sphere
from konvert.points import Spherical
from konvert.points import Stereographic
from konvert.points import Cartesian2DToBipolar
from konvert.points import Square
from konvert.points import Disc
from konvert.points import SquareToDiscSquircular
from konvert.points import DiscToSquareSquircular

class PointsTestCase(unittest.TestCase):

    def assertArraysEqual(self, left, right):
        self.assertTrue(np.allclose(left, np.array(right), equal_nan=True), msg='Array content different {} != {}.'.format(left, right))


class UtilitiesTest(PointsTestCase):

    def test_degrees(self):
        v = 90 * degrees
        self.assertAlmostEqual(v, np.pi / 2)

    def test_multiplying_degrees_with_iterable(self):
        v = [45, 90, 180] * degrees
        self.assertArraysEqual(v, [np.pi/4, np.pi/2, np.pi])


class PointsTest(PointsTestCase):

    def test_points(self):
        points = Points()

    def test_to(self):
        points = Points()
        with self.assertRaises(IndexError):
            points.to(Cartesian1D)

    def test_project(self):
        points = Points()
        with self.assertRaises(IndexError):
            points.project(Stereographic)


class Cartesian2DTest(PointsTestCase):

    def test_negative(self):
        c = Cartesian2D(3, 4)
        self.assertEqual(-c, Cartesian2D(-3, -4))

    def test_shift(self):
        c = Cartesian2D(3, 4)
        c0 = c.shift(-Cartesian2D(3, 4))
        c1 = Cartesian2D(0, 0)
        self.assertEqual(c0, c1)

    def test_rotate(self):
        c = Cartesian2D([3, 4], [4, 3])
        c_ref = - c.copy()

        # rotate around origo.
        c.rotate(np.pi)
        self.assertEqual(c, c_ref)

        # rotate with point
        c.rotate(np.pi, Cartesian2D(3, 3))
        self.assertEqual(c, Cartesian2D([9, 10], [10, 9]))


class Cartesian3DTest(PointsTestCase):

    def test_norm(self):
        c = Cartesian3D(3, 4, 5)
        self.assertEqual(c.norm, np.sqrt(3**2 + 4**2 + 5**2))

        c = Cartesian3D([1, 2], [1, 2], [1, 2])
        self.assertArraysEqual(c.norm, [np.sqrt(3), np.sqrt(12)])
        self.assertEqual(c.norm.shape, (2,))

    def test_normalize(self):
        c = Cartesian3D(1, 1, 1)
        c = c.normalized()
        self.assertArraysEqual(c.x, [(3) ** -0.5])

        c = Cartesian3D([1, 2], [1, 2], [1, 2])
        c = c.normalized()
        self.assertArraysEqual(c.x, np.array([3, 3]) ** -0.5)

    def test_rotate(self):
        c = Cartesian3D(1, 1, 1)
        c.rotate(np.pi, Cartesian3D(1, 0, 0))
        self.assertEqual(c, Cartesian3D(1, -1, -1))


class QuaternionTest(PointsTestCase):

    def setUp(self):
        self._quaternion = Quaternion([0, np.pi, 2/3 * np.pi], [1, 2, 3], [4, 5, 6], [7, 8, 9])

    def tearDown(self):
        del self._quaternion

    def test_quaternion(self):
        q = self._quaternion
        self.assertArraysEqual(q.w, [0, np.pi, 2/3 * np.pi])
        self.assertArraysEqual(q.x, [1, 2, 3])
        self.assertArraysEqual(q.y, [4, 5, 6])
        self.assertArraysEqual(q.z, [7, 8, 9])

    def test_multiply_cartesian(self):
        # one-eight rotation around x axis.
        q = Quaternion.from_angle_axis(np.pi / 4, Cartesian3D(1, 0, 0))
        # first octant rotates to x-z plane.
        c = Cartesian3D(1, 1, 1)
        result = q * c
        self.assertIsInstance(result, Cartesian3D)
        self.assertArraysEqual(result.x, 1)
        self.assertArraysEqual(result.y, 0)
        self.assertArraysEqual(result.z, np.sqrt(2))

        c = Cartesian3D([1, 2], [1, 2], [1, 2])
        result = q * c
        self.assertIsInstance(result, Cartesian3D)
        self.assertArraysEqual(result.x, [1, 2])
        self.assertArraysEqual(result.y, [0, 0])
        self.assertArraysEqual(result.z, [np.sqrt(2), np.sqrt(8)])

        q = Quaternion.from_angle_axis([np.pi / 4, 0], Cartesian3D([1, 0], [0, 1], [0, 0]))
        result = q * c
        self.assertIsInstance(result, Cartesian3D)
        self.assertArraysEqual(result.x, [1, 2])
        self.assertArraysEqual(result.y, [0, 2])
        self.assertArraysEqual(result.z, [np.sqrt(2), 2])

    def test_multiply_quaternion(self):
        q0 = Quaternion.from_angle_axis(np.pi / 4, Cartesian3D(1, 0, 0))
        q1 = Quaternion.from_angle_axis(np.pi / 4, Cartesian3D(1, 0, 0))

        result = q0 * q1
        self.assertArraysEqual(result.theta, np.pi / 2)
        self.assertEqual(result.axis, Cartesian3D(1, 0, 0))

    def test_mulitple_rotations(self):
        # quarter rotation around x
        qx = Quaternion.from_angle_axis(np.pi / 2, Cartesian3D(1, 0, 0))
        # quarter rotation around y
        qy = Quaternion.from_angle_axis(np.pi / 2, Cartesian3D(0, 1, 0))

        # (0, 0, 1).
        c = Cartesian3D(0, 0, 1)

        # Rotate first y then x: (0, 0, 1) -> (1, 0, 0)
        result = (qx * qy) * c
        self.assertArraysEqual(result.x, 1)
        self.assertArraysEqual(result.y, 0)
        self.assertArraysEqual(result.z, 0)

        # Rotate first x then y: (0, 0, 1) -> (0, -1, 0)
        result = (qy * qx) * c
        self.assertArraysEqual(result.x, 0)
        self.assertArraysEqual(result.y, -1)
        self.assertArraysEqual(result.z, 0)

    def test_conjugated(self):
        q0 = Quaternion(1, 2, 3, 4)
        q1 = Quaternion(1, -2, -3, -4)
        self.assertEqual(q0.conjugated(), q1)


class SphericalTest(PointsTestCase):

    def test_spherical(self):
        points = Spherical([0, np.pi], [0, np.pi], [1, 2])
        self.assertIsInstance(points.theta, np.ndarray)
        self.assertIsInstance(points.phi, np.ndarray)
        self.assertIsInstance(points.r, np.ndarray)

    def test_to(self):
        points = Spherical(np.pi, np.pi, 2)
        sp = points.project(Sphere)
        sp_ref = Sphere(np.pi, np.pi, 1)
        self.assertEqual(sp, sp_ref)


class BipolarTest(PointsTestCase):

    def test_foci(self):
        points = Bipolar([0, 0], [-1e10, 1e10], 2)
        p0 = points.to(Cartesian2D)
        p1 = Cartesian2D([-2, 2], [0, 0])
        self.assertEqual(p0, p1)

        # Note. Inversion is imprecise.
        p2 = p0.project(Cartesian2DToBipolar, a=2)
        self.assertLess(p2.tau[0], -1e10)
        self.assertGreater(p2.tau[1], 1e10)

    def test_center(self):
        points = Cartesian2D(0, 0)
        p0 = points.to(Bipolar)
        p1 = Bipolar(np.pi, 0)
        self.assertEqual(p0, p1)

        p2 = p0.to(Cartesian2D)
        self.assertEqual(p2, points)

    def test_centerline(self):
        points = Bipolar(
            sigma=np.linspace(10, 270, 10) * degrees,
            tau=np.zeros((10,)),
            a=1
        )
        p0 = points.to(Cartesian2D)
        self.assertArraysEqual(p0.x, np.zeros((10,)))

        p1 = p0.to(Bipolar)
        self.assertEqual(p1, points)


class SphereTest(PointsTestCase):

    def test_sphere(self):
        points = Sphere([0, np.pi], [0, np.pi], 1)
        self.assertIsInstance(points.theta, np.ndarray)
        self.assertIsInstance(points.phi, np.ndarray)
        self.assertEqual(points.R, 1)

    def test_to(self):
        points = Sphere([0, np.pi], [0, np.pi], 1)
        sp = points.to(Spherical)
        self.assertArraysEqual(sp.theta, [0, np.pi])
        self.assertArraysEqual(sp.phi, [0, np.pi])
        self.assertArraysEqual(sp.r, [1, 1])

        ca = points.to(Cartesian3D)
        self.assertArraysEqual(ca.x, [0, 0])
        self.assertArraysEqual(ca.y, [0, 0])
        self.assertArraysEqual(ca.z, [1, -1])

    def test_project(self):
        points = Sphere([0, np.pi / 2], [0, np.pi / 2], 1)
        po = points.project(Stereographic)
        self.assertArraysEqual(po.theta, [0, np.pi/2])
        self.assertArraysEqual(po.r, [np.nan, 1])


class SquircularTet(PointsTestCase):

    def test_SquareToDisc(self):
        """ Test squircular square to disc and back """
        s = Square(*[np.random.rand(3) * 2 for i in range(2)], 4)
        m = s.project(SquareToDiscSquircular)
        s1 = s.project(SquareToDiscSquircular).project(DiscToSquareSquircular)
        self.assertEqual(s, s1)

    def test_SquareToDiscSquircular(self):
        """ Test plot """
        s = Square([1, 1, -1, -1], [-1, 1, 1, -1], 2)
        d = s.project(SquareToDiscSquircular)
        d0 = Cartesian2D([1, 1, -1, -1], [-1, 1, 1, -1]).to(Disc).resize(1)
        self.assertEqual(d, d0)

    def notest_plot_SquareToDiscSquircular(self):
        xs, ys = [], []
        for s in [1, -1]:
            for a in np.linspace(0, 2, 10):
                xs.append(s * a * np.ones((64, )))
                ys.append(a * np.linspace(-1, 1, 64))

        s = Square(np.concatenate(xs + ys), np.concatenate(ys + xs), 4)
        s.to(Cartesian2D).plot()

        d = s.project(SquareToDiscSquircular)
        d.to(Polar).scatter()

        s = d.project(DiscToSquareSquircular)
        s.to(Cartesian2D).plot()

    def test_DiscToSquareSquircular(self):
        """ Test plot """
        d = Disc([45, 135, 225, 315] * degrees, [2] * 4, 2)
        s = d.project(DiscToSquareSquircular)
        s0 = Cartesian2D([1, -1, -1, 1], [1, 1, -1, -1]).to(Square).resize(4)
        self.assertEqual(s, s0)

    def notest_plot_DiscToSquareSquircular(self):
        thetas, rs = [], []
        for a in np.linspace(0, 2, 10, endpoint=True):
            rs.append(a * np.ones((64, )))
            thetas.append(np.linspace(1, 360, 64) * degrees)

        s = Disc(np.concatenate(thetas), np.concatenate(rs), 2)
        s.to(Polar).scatter()

        s = s.to(DiscToSquareSquircular)
        s.to(Cartesian2D).plot()

        s.to(SquareToDiscSquircular).to(Polar).scatter()


class ConversionsTest(PointsTestCase):

    def test_CartesianSphericalCylindrical(self):
        """ Test a round trip """
        c = Cartesian3D(*[np.random.rand(3) for i in range(3)])
        c_ref = c.to(Spherical).to(Cylindrical).to(Cartesian3D)
        self.assertEqual(c, c_ref)

    def test_CartesianCylindricalSpherical(self):
        """ Test a round trip """
        c = Cartesian3D(*[np.random.rand(3) for i in range(3)])
        c_ref = c.to(Cylindrical).to(Spherical).to(Cartesian3D)
        self.assertEqual(c, c_ref)

    def test_CartesianPolar(self):
        """ Test a round trip """
        c = Cartesian2D(*[np.random.rand(3) for i in range(2)])
        c_ref = c.to(Polar).to(Cartesian2D)
        self.assertEqual(c, c_ref)

    def test_SphereEquitorial(self):
        """ Test a round trip """
        c = Sphere(*[np.random.rand(3) for i in range(2)], 1)
        c_ref = c.to(Equitorial).to(Sphere)
        self.assertEqual(c, c_ref)

    def test_PolarCartesianBipolar(self):
        c = Polar(*[np.random.rand(3) for i in range(2)])
        c_ref = c.to(Cartesian2D).to(Bipolar).to(Polar)
        self.assertEqual(c, c_ref)


class MapProjectionsTest(PointsTestCase):

    def test_Stereographic(self):
        points = Sphere(np.pi / 2, np.pi / 3, 1)
        p0 = points.project(Stereographic)
        self.assertEqual(p0, Polar(np.pi / 3, 1))

    def test_Orthographic(self):
        points = Sphere(np.random.rand(5) * np.pi, np.random.rand(5) * 2 * np.pi, 1)
        p0 = points.project(Orthographic).to(Cartesian2D)
        p1 = points.project(OnPlane)
        self.assertEqual(p0, p1)

    def test_AzimuthalEquidistant(self):
        points = Sphere(np.pi / 4, np.pi / 4, 1)
        p0 = points.project(AzimuthalEquidistant).to(Cartesian2D)
        p1 = Cartesian2D(1.6660811, 1.6660811)
        self.assertEqual(p0, p1)

    def test_Mercator(self):
        points = Sphere(np.pi / 2, np.pi / 3, 1)
        p0 = points.project(Mercator)
        p1 = Cartesian2D(np.pi / 3, 0)
        self.assertEqual(p0, p1)

    def test_Equirectangular(self):
        points = Sphere(np.pi / 2, np.pi / 4, 1)
        p0 = points.project(Equirectangular)
        p1 = Cartesian2D(0, 0.7853981)
        self.assertEqual(p0, p1)

    def test_Azimuthal(self):
        points = Sphere(np.pi / 2, np.pi / 4, 1)
        p0 = points.project(Azimuthal, d=2).to(Cartesian2D)
        p1 = Cartesian2D(*[2 ** -0.5] * 2)
        self.assertEqual(p0, p1)


if __name__ == '__main__':
    unittest.main()