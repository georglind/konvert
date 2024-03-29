from abc import ABCMeta

import numpy as np
import warnings

from konvert.conversions import Conversion, ConversionGraph
from konvert.projections import Projection, ProjectionCollection

# Import matplotlib or replace plt with error.
try:
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
except ImportError:
    class ImportedError(object):
        def __getattr__(self, get):
            raise ImportError('The library matplotlib is not available. Install matplotlib in order to plot points.')
    plt = ImportedError()


# This holds all our conversions.
converters = ConversionGraph()

# This holds all our projections.
projectors = ProjectionCollection()


def numpify(func):
    """Utilitiy convert all function arguments to numpy arrays."""
    def wrapper(self, *args, **kwargs):
        args = [np.array(arg) for arg in args]
        kwargs = {key: np.array(value) for key, value in kwargs.items()}
        func(self, *args, **kwargs)

    return wrapper


# Utilitiy to convert degrees to radians.
class Degree(object):

    def __mul__(self, other):
        return np.deg2rad(other)

    def __rmul__(self, other):
        return np.deg2rad(other)

    # work around numpy
    __array_priority__ = 10000

degree = Degree()
degrees = degree


class Points(object, metaclass=ABCMeta):
    """Points abstract base class"""

    def copy(self):
        """
        Generic copy based on instantiation signature
        """
        cls = type(self)
        return cls(*[getattr(self, name) for name in cls._sig])

    def __eq__(self, other):
        """
        Generic equality based on signature
        """
        cls = type(self)
        neq = any(not np.allclose(getattr(self, name), getattr(other, name)) for name in cls._sig)
        return not neq

    def print(self):
        """
        Generic print based on signature
        """
        cls = type(self)
        print(self)
        for sig in cls._sig:
            print(sig, getattr(self, sig))

    def to(self, dst):
        """
        Convert to destination type using the conversion graph
        """
        # To tuple.
        if dst is tuple:
            return tuple(getattr(self, name) for name in self._sig)

        return converters.convert(self, dst)

    def project(self, dst, *args, **kwargs):
        """
        Project to destination type using registered projector. Additional arguments are passed to projection.
        """
        # Polymorphic destination.
        # Projection onto Points type. A direct projection must exist.
        if issubclass(dst, Points):
            return projectors.project(self, dst, *args, **kwargs)

        # Project using Projection. First convert to projection source type.
        if issubclass(dst, Projection):
            target = self
            if dst.src is not type(self):
                target = self.to(dst.src)

            return dst.convert(target, *args, **kwargs)

        # Errors.
        else:
            raise ValueError(f'Projection destination must by either Points or Projection, instead got {dst}.')


class Cartesian(Points, metaclass=ABCMeta):
    """ Abstract cartesian type """
    def __neg__(self):
        """
        Generic negative
        """
        args = [-getattr(self, sig) for sig in self._sig]
        return type(self)(*args)

    @property
    def norm(self):
        """
        The norm of the cartisian point.
        """
        return np.linalg.norm(np.vstack([getattr(self, sig) for sig in self._sig]), axis=0)

    def normalized(self):
        """
        Create a normalized version of this instance.
        """
        norm = self.norm
        return type(self)(*(getattr(self, sig) / norm for sig in self._sig))

    def shift(self, offset):
        """
        Shift by offset
        """
        return type(self)(*(getattr(self, sig) + getattr(offset, sig) for sig in self._sig))



class Cartesian1D(Cartesian):
    """Cartesian points in one dimension"""
    _sig = ('x',)

    @numpify
    def __init__(self, x):
        self.x = x


class Cartesian2D(Cartesian):
    """Cartesian points in two dimensions"""
    _sig = ('x', 'y')

    @numpify
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def rotate(self, theta, point=None):
        if point is None:
            point = Cartesian2D(0, 0)
        else:
            point = point.to(Cartesian2D)

        x = self.x - point.x
        y = self.y - point.y

        c = Cartesian2D(x, y).to(Polar)
        c.theta += theta

        c = c.to(Cartesian2D).shift(point)

        self.x = c.x
        self.y = c.y

        return self

    def plot(self):
        plt.plot(self.x, self.y, 'o')
        plt.show()


class Polar(Points):
    """Polar coordinates"""
    _sig = ('theta', 'r')

    @numpify
    def __init__(self, theta, r=None):
        self.theta = theta

        if r is None:
            r = np.ones(theta.shape)

        self.r = r

    def scatter(self, **kwargs):
        """ Plot as scatter using matplotlib """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='polar')
        ax.scatter(self.theta, self.r, **kwargs)
        # update limits
        ax.relim()
        ax.autoscale_view()

        plt.show()


# 2D conversions.
@converters.register()
class Cartesian2DToPolar(Conversion):
    src = Cartesian2D
    dst = Polar

    @staticmethod
    def convert(ca):
        theta = np.arctan2(ca.y, ca.x)
        r = np.sqrt(ca.x**2 + ca.y**2)

        return Polar(theta, r)


@converters.register()
class PolarToCartesian2D(Conversion):
    src = Polar
    dst = Cartesian2D

    @staticmethod
    def convert(po):
        x = po.r * np.cos(po.theta)
        y = po.r * np.sin(po.theta)

        return Cartesian2D(x, y)


# 3D
class Cartesian3D(Cartesian):
    """Cartestian points in three dimensions"""
    _sig = ('x', 'y', 'z')

    @numpify
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def rotate(self, theta, point=None):
        """
        Rotate all points theta angle around a given point.
        """
        if point is None:
            point = Cartesian3D(0, 0, 1)

        q = Quaternion.from_angle_axis(theta, point)
        n = q * self
        self.x = n.x
        self.y = n.y
        self.z = n.z

        return self

    def plot(self, size=None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.x, self.y, self.z, s=size)
        plt.show()


@projectors.register()
class OnPlane(Projection):
    """Project 3D points onto plane"""
    src = Cartesian3D
    dst = Cartesian2D

    @staticmethod
    def convert(ca):
        return Cartesian2D(ca.x, ca.y)

class Quaternion(Points):
    """Quaternion representation"""
    _sig = ('w', 'x', 'y', 'z')

    @classmethod
    def from_angle_axis(cls, theta, axis):
        theta = np.array(theta)
        norm = axis.norm

        w = np.cos(theta / 2.)
        x = axis.x * np.sin(theta / 2.) / norm
        y = axis.y * np.sin(theta / 2.) / norm
        z = axis.z * np.sin(theta / 2.) / norm
        return cls(w, x, y, z)

    @numpify
    def __init__(self, w, x, y, z):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    @property
    def theta(self):
        return 2 * np.arccos(self.w)

    @property
    def axis(self):
        return self.project(Cartesian3D).normalized()

    def __mul__(self, other):
        if isinstance(other, Quaternion):
            return self.multiply_quaternion(other)
        elif isinstance(other, Cartesian3D):
            return self.multiply_vector(other)
        else:
            raise Exception(f'Multiplication with unknown type {type(b)}.')

    def multiply_quaternion(self, q):
        w = self.w * q.w - self.x * q.x - self.y * q.y - self.z * q.z
        x = self.w * q.x + self.x * q.w + self.y * q.z - self.z * q.y
        y = self.w * q.y + self.y * q.w + self.z * q.x - self.x * q.z
        z = self.w * q.z + self.z * q.w + self.x * q.y - self.y * q.x

        result = Quaternion(w, x, y, z)
        return result

    def multiply_vector(self, v):
        q = v.to(Quaternion)
        return (self * q * self.conjugated()).project(Cartesian3D)

    def conjugated(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)


@projectors.register()
class QuaternionToCartestian3D(Projection):
    src = Quaternion
    dst = Cartesian3D

    @staticmethod
    def convert(q):
        return Cartesian3D(q.x, q.y, q.z)


@converters.register()
class Cartesian3DToQuaternion(Conversion):
    src = Cartesian3D
    dst = Quaternion

    @staticmethod
    def convert(ca):
        return Quaternion(0, ca.x, ca.y, ca.z)


class Cylindrical(Points):
    """Point in cylindrical coordinates."""
    _sig = ('theta', 'r', 'z')

    @numpify
    def __init__(self, theta, r, z):
        self.theta = theta
        self.r = r
        self.z = z


class Spherical(Points):
    """Point in spherical coordinates in physicists convention (theta, phi, r)"""
    _sig = ('theta', 'phi', 'r')

    @numpify
    def __init__(self, theta, phi, r=None):
        self.theta = theta
        self.phi = phi

        if r is None:
            r = np.ones(theta.shape)

        self.r = r

    def spin(self, delta):
        """
        Spin the ball through a delta angle.
        """
        self.phi += delta

        return self


# 3D conversions
@converters.register()
class SphericalToCartesian3D(Conversion):
    src = Spherical
    dst = Cartesian3D

    @staticmethod
    def convert(sp):
        cp = np.sin(sp.theta)

        x = sp.r * cp * np.cos(sp.phi)
        y = sp.r * cp * np.sin(sp.phi)
        z = sp.r * np.cos(sp.theta)

        return Cartesian3D(x, y, z)


@converters.register()
class Cartesian3DToSpherical(Conversion):
    src = Cartesian3D
    dst = Spherical

    @staticmethod
    def convert(ca):
        r = np.sqrt(ca.x**2 + ca.y**2 + ca.z**2)
        theta = np.arccos(ca.z / r)
        phi = np.arctan2(ca.y, ca.x)

        return Spherical(theta, phi, r)


@converters.register()
class SphericalToCylindrical(Conversion):
    src = Spherical
    dst = Cylindrical

    @staticmethod
    def convert(sp):
        theta = sp.phi
        r = sp.r * np.sin(sp.theta)
        z = sp.r * np.cos(sp.theta)
        return Cylindrical(theta, r, z)


@converters.register()
class CylindricalToCartesian3D(Conversion):
    src = Cylindrical
    dst = Cartesian3D

    @staticmethod
    def convert(cy):
        x = cy.r * np.cos(cy.theta)
        y = cy.r * np.sin(cy.theta)

        return Cartesian3D(x, y, cy.z)


@converters.register()
class Cartesian3DToCylindrical(Conversion):
    src = Cartesian3D
    dst = Cylindrical

    @staticmethod
    def convert(ca):
        theta = np.arctan2(ca.y, ca.x)
        r = np.sqrt(ca.x ** 2 + ca.y ** 2)

        return Cylindrical(theta, r, ca.z)


# Curved 2D
class Bipolar(Points):
    _sig = ('sigma', 'tau', 'a')

    @numpify
    def __init__(self, sigma, tau, a=1):
        """
        Bipolar coordinates with foci located 2 * a apart.
        """
        self.sigma = np.mod(np.array(sigma), 2 * np.pi)
        self.tau = tau
        self.a = a


@converters.register()
class BipolarToCartesian2D(Conversion):
    src = Bipolar
    dst = Cartesian2D

    @staticmethod
    def convert(bi):
        tau = bi.tau
        # Ignore overflow warnings.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = np.cosh(tau) - np.cos(bi.sigma)
            s = np.sinh(tau)

            # Handle sinh or cosh overflows.
            mask = np.isinf(s)
            if mask.any():
                mask_f = ~mask
                mask_n = np.isneginf(s)

                x = bi.a * np.ones(tau.shape)
                x[mask_f] = bi.a * s[mask_f] / d[mask_f]
                x[mask_n] = - bi.a
            else:
                x = bi.a * np.sinh(tau) / d

            # Only fails for tau == 0 and sigma == n * pi.
            y = bi.a * np.sin(bi.sigma) / d

        return Cartesian2D(x, y)


@converters.register()
class BipolarToCartesian2DConversion(Conversion):
    src = Cartesian2D
    dst = Bipolar

    @staticmethod
    def convert(ca):
        return Cartesian2DToBipolar.convert(ca, a=1)


@projectors.register()
class Cartesian2DToBipolar(Projection):
    src = Cartesian2D
    dst = Bipolar

    @staticmethod
    def convert(ca, a=1):
        # Ignore division by zero.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = (a ** 2 - ca.x ** 2 - ca.y ** 2)
            tau = 0.5 * np.log(((ca.x + a) ** 2 + ca.y ** 2) / ((ca.x - a) ** 2 + ca.y ** 2))
            sigma = np.pi - 2 * np.arctan2(2 * a * ca.y, (d + np.sqrt(d ** 2 + ((2 * a) * ca.y) ** 2)))

        return Bipolar(sigma, tau, a)


class Sphere(Points):
    """
    Point on a sphere in physicists convention (theta, phi, r)

    theta : azimuth (angle with negative z).
    phi   : right ascension (angle with positive x).
    R     : radius (single value scalar)
    """
    _sig = ('theta', 'phi', 'R')

    def __init__(self, theta, phi, R):
        self.theta = np.array(theta)
        self.phi = np.array(phi)

        # R is scalar.
        self.R = R

    def resize(self, R):
        return Sphere(self.theta, self.phi, R)


# 2D to 3D
@converters.register()
class SphereToSpherical(Conversion):
    src = Sphere
    dst = Spherical

    @staticmethod
    def convert(sp):
        return Spherical(sp.theta, sp.phi, sp.R)


@projectors.register()
class OnSphere(Projection):
    """Project points onto sphere"""
    src = Spherical
    dst = Sphere

    @staticmethod
    def convert(sp, R=1):
        return Sphere(sp.theta, sp.phi, R)


class Equitorial(Points):
    """Points on the (celestial) sphere defined by right ascension and declination"""
    _sig = ('dec', 'asc', 'r')

    def __init__(self, dec, asc, R=1):
        self.dec = np.array(dec)
        self.asc = np.array(asc)
        self.R = R

    @property
    def lon(self):
        """Longitude"""
        return self.asc

    @lon.setter
    def set_lon(self, lon):
        self.asc = lon

    @property
    def lat(self):
        """Lattitude"""
        return self.dec

    @lat.setter
    def set_lat(self, lat):
        self.dec = lat


@converters.register()
class EquitorialToSphere(Conversion):
    src = Equitorial
    dst = Sphere

    @staticmethod
    def convert(eq):
        return Sphere(np.pi / 2 - eq.dec, eq.asc, R=eq.R)


@converters.register()
class SphereToEquitorial(Conversion):
    src = Sphere
    dst = Equitorial

    @staticmethod
    def convert(sp):
        return Equitorial(np.pi / 2 - sp.theta, sp.phi, R=sp.R)


@projectors.register()
class Mercator(Projection):
    """Mercator map projection"""
    src = Sphere
    dst = Cartesian2D

    @staticmethod
    def convert(sp, cutoff=np.pi / 5, delta=0):
        theta = np.array(sp.theta)
        mask = (theta < cutoff) & (theta > np.pi - cutoff)
        try:
            theta[mask] = np.nan
        except ValueError:
            raise ValueError('Point outside Mercator cut-off.')

        x = sp.R * (sp.phi - delta)
        y = sp.R * np.log(np.tan(theta / 2))
        return Cartesian2D(x, y)


@projectors.register()
class Equirectangular(Projection):
    """Equirectangular map projection"""
    src = Sphere
    dst = Cartesian2D

    @staticmethod
    def convert(sp):
        x = sp.phi * np.cos(sp.theta)
        y = sp.phi

        return Cartesian2D(x, y)

    @staticmethod
    def invert(ca, R=1):
        return Sphere(np.arccos(ca.x / ca.y), ca.y, R)


@projectors.register()
class Stereographic(Projection):
    """Stereographic projection"""
    src = Sphere
    dst = Polar

    @staticmethod
    def convert(sp):
        # Ignore divide by zero warnings.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = sp.R * np.sin(sp.theta) / (1 - np.cos(sp.theta))

        return Polar(sp.phi, r)

    @staticmethod
    def invert(po, R=1):
        theta = 2 * np.arctan(r / po.r)
        return Sphere(phi=po.theta, theta=theta, R=R)


@projectors.register()
class Orthographic(Projection):
    src = Sphere
    dst = Polar

    @staticmethod
    def convert(sp):
        return Polar(sp.phi, sp.R * np.sin(sp.theta))

    @staticmethod
    def invert(sp, R=1):
        return Sphere(np.arcsin(po.theta / R),  po.phi, R)


@projectors.register()
class AzimuthalEquidistant(Projection):
    src = Sphere
    dst = Polar

    @staticmethod
    def convert(sp):
        return Polar(sp.phi, sp.R * (np.pi - sp.theta))

    @staticmethod
    def invert(po, R=1):
        return Sphere(po.theta / R, po.phi, R)


@projectors.register()
class Azimuthal(Projection):
    """
    General azimuthal projection
    """
    src = Sphere
    dst = Polar

    @staticmethod
    def convert(sp, d=1):
        if d <= 0:
            raise ValueError('Azimuthal projection undefined for d<=0')

        elif d is float('inf'):
            return Orthographic.convert(sp).to(Polar)

        r = sp.R * np.sin(sp.theta) / (1 - np.cos(sp.theta) / d)
        return Polar(sp.phi, r)


class Disc(Points):
    """
    A two-dimensional disc.
    """
    _sig = ['theta', 'r', 'R']

    def __init__(self, theta, r, R=1):
        """
        Radius R
        """
        self.theta = np.array(theta)
        self.r = np.array(r)
        self.R = R

    def resize(self, R):
        scale = R / self.R
        return Disc(self.theta, self.r * scale, R)

    def rotate(self, delta):
        return Disc(self.theta + delta, self.r, self.R)


@converters.register()
class DiscToPolarConversion(Conversion):
    src = Disc
    dst = Polar

    @staticmethod
    def convert(di):
        return Polar(di.theta, di.r)


@converters.register()
class PolarToDiscConversion(Conversion):
    src = Polar
    dst = Disc

    @staticmethod
    def convert(po):
        R = np.max(po.r)
        return Disc(po.theta, po.r, R)


@projectors.register()
class PolarToDisc(Projection):
    src = Polar
    dst = Disc

    @staticmethod
    def convert(po, R=1):
        return Disc(po.theta, po.r, R)


class Square(Points):
    """
    A two-dimensional square.
    """
    _sig = ['x', 'y', 'a', 'alpha']

    def __init__(self, x, y, a, alpha=0):
        """
        Side length a
        """
        self.x = np.array(x)
        self.y = np.array(y)
        self.a = a
        self.alpha = alpha

    def resize(self, a):
        scale = a / self.a
        return Square(self.x * scale, self.y * scale, a, self.alpha)

    def rotate(self, delta):
        return Square(self.x, self.y, self.a, self.alpha + delta)


@converters.register()
class SquareToCartesian2D(Conversion):
    src = Square
    dst = Cartesian2D

    @staticmethod
    def convert(sq):
        return Cartesian2D(np.cos(sq.alpha) * sq.x, np.sin(sq.alpha) * np.sq.y)


@converters.register()
class Cartesian2DToSquareConversion(Conversion):
    src = Cartesian2D
    dst = Square

    @staticmethod
    def convert(ca):
        a = 2 * max(np.max(ca.x), np.max(ca.y))
        return Square(ca.x, ca.y, a)


@projectors.register()
class Cartesian2DToSquare(Projection):
    src = Cartesian2D
    dst = Square

    @staticmethod
    def convert(ca, a=1, alpha=0):
        return Square(ca.x, ca.y, a, alpha)



@projectors.register()
class DiscToSquareSquircular(Projection):
    src = Disc
    dst = Square

    @staticmethod
    def convert(di):
        # See https://arxiv.org/abs/1509.06344
        f = np.tan(di.theta)

        # Fix division by zero error my masking zero values.
        mask = (f != 0)

        # Fill in x, and compute unmasked values.
        x = np.zeros(di.theta.shape)
        x_sign = np.sign(np.cos(di.theta))
        km = f[mask] ** 2
        x[mask] = di.R * x_sign[mask] * np.sqrt((km + 1 - np.sqrt( (km + 1) ** 2 - 4 * km * (di.r[mask] / di.R) ** 2)) / (2 * km))
        y = f * x

        # Fill in masked values.
        x[~mask] = x_sign[~mask] * di.r[~mask]

        return Square(x, y, 2 * di.R)


@projectors.register()
class SquareToDiscSquircular(Projection):
    src = Square
    dst = Disc

    @staticmethod
    def convert(sq):
        # See https://arxiv.org/abs/1509.06344
        alpha = sq.alpha
        if alpha != 0:
            sq = sq.rotate(-alpha)

        R = 0.5 * sq.a
        theta = np.arctan2(sq.y, sq.x)
        x2 = (1 / R * sq.x) ** 2
        y2 = (1 / R * sq.y) ** 2
        r = np.sqrt(x2 + y2 - x2 * y2) * R

        di = Disc(theta, r, R)

        if alpha != 0:
            di = di.rotate(alpha)

        return di
