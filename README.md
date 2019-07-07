# konvert

Konvert implements universal conversion graphs for e.g. coordinate transformations.

The core machinery of konvert is a graph representation of entity types that are connected by two-way conversions. The graph can be expandend incrementally, since it only requires two connections to connect a new entity to the graph, The entity can then be freely converted to any other type in the graph, by automatically chaining conversions between existing types.

In addition konvert also handles a projections, conversions that require additional parameters, by exposing a general projection regsitry.

## Points

The moule konvert.points includes a conversion graph for coordinate transformations. As an example consider a set of points in two-dimensional Cartesian coordinates:

```python
from konvert.points import Cartesian2D

line = Cartesian2D(x=[0, 1, 2], y=[0, 1, 2])
```

This points can be converted to other implented 2D point sets, ``Polar`` and ``Bipolar``, by using the ``to()`` method

```python
from konvert.points import Polar, Bipolar

lp = line.to(Polar)
lb = line.to(Bipolar)
```

It can also be lifted up to its 3D description, ```l3 = line.to(Cartesian3D)```. The conversion graph can automatically chains conversions, so in fact, any point in the plane can be hoisted up to any point in three-dimensional space. That means, that points in e.g. polar coordinates, can be hoisted up to ```Cartesian3D```.

```python
p0 = Polar(theta=30 * degrees, r=1.5).to(Cartesian3D)
```

Points in 3D can be represented using ``Cartesian3D``, ``Cylindrical``, and ``Spherical``. All Cartesian types have some helper methods for easily manipulating points. 

```python
p0 = Cartesian3D(1, 1, 1)
q0 = Cartesian3D(1, 1, 0)

# Create normalized version
p1 = p0.normalized()

# Shift p0 by q0
p0.shift(q0)  

# Rotate around axis thorugh q0.
p0.rotate(theta=45 * degrees, point=q0)
```

### Helpers

Because points work extensively with angles it is convenient to define the ``degrees`` symbol, which magically converts values and arrays to radians,

```python
from konvert.points import degrees

theta = 90 * degrees
# theta is now pi/2

theta = numpy.array([45, 60, 90, 180, 360]) * degrees
# theta is now numpy.array([pi/4, pi/3, pi/2, pi, 2*pi])
```

### Map projections

The points module also contains a set of map projections: ``Azimuthal``, ``AzimuthalEquidistant``, ``Mercator``, ``Orthographic``, ``Stereographic`` and ``Equirectangular``. Points are represented on the ``Sphere``, in spherical coordinates, or in ``Equitorial`` coordinates (lattitude and longitude).

```python
points = Equitorial([40.12, 50.53] * degrees, [33.16, 44.53] * degrees, r=1)

p0 = points.project(Mercator)
p1 = points.project(Stereographic)
...
```

The points can be easily plotted using matplotlib and the plot utility on Cartesian2D, ``p0.to(Cartesian2D).plot()``.

### Extending the conversion graph

It is fairly simple to extend an existing conversion graph and an existing projection registry. Let us extend the points graph with a Skew2D coordinate system. Note the use of numpy to efficiently represent and transform coordinates.

```python
from konvert.points import Points

class Skew2D(Points):
    _sig = ('x', 'y', 'theta')
    
    def __init__(self, x, y, theta):
        """ 
        Skew 2D coordinates. Theta is a scalar denominating the skew angle.
        """
        self.x = numpy.array(x)
        self.y = numpy.array(y)
        self.theta = theta
```

The Skew2D class can be connected to the points conversion graph by using the following conversion.

```python
from konvert.points import converters, Conversion

@converters.register()
class Skew2DToCartesian2D(Conversion):
    src = Skew2D
    dst = points.Cartesian2D
    
    @staticmethod
    def convert(skew):
        return points.Cartesian2D(skew.x + numpy.cos(theta) * skew.y, numpy.sin(theta) * skew.y)
```

The reverse conversion requires the additional theta parameter and must be implemented as a projection.

```python
from konvert.points import projectors, Projection

@projectors.register()
class Cartesian2DToSkew2D(Projection):
    src = points.Cartesian2D
    dst = Skew2D
    
    @staticmethod
    def project(cart, theta=np.pi / 2):
        y = cart.y / numpy.sin(theta)
        x = cart.x - numpy.cos(theta) * y
        return Skew2D(x, y, theta)
```

With these two additions we can convet between Skew2D points and any points in the graph,

```python
p0 = Skew2D(1, 2, theta=45 * degrees)
p1 = p0.to(Cylindrical)
```

or the opposite way,

```python
p1 = Cylindrical(theta=30 * degrees, phi=45 * degrees, r=10)
p0 = p1.project(OnPlane).project(Cartesian2DToSkew2D, theta=45 * degrees)
```

## Extensions 

The conversion graph has been implemented in the ``conversions`` module. It is possible to create additional conversion graphs and register existing or new conversions in those graphs. As an example we may create a colors module, which converts between values in RGB and HSL color space. For this is example we will not bother with vectorizing the entities, and instead just work with single color entries and use the  ``colorsys`` module in the python standard library.

```python
import colorsys
from konvert.conversions import Conversion, ConversionGraph

converters = ConversionGraph()

class Color(metaclass=ABCMeta):     
     def to(self, type):
        return convertes.convert(self, type)


class RGB(Color):
    def __init__(self, r, g, b):
        self.r = r
        self.g = g
        self.b = b

class HSL(Color):
    def __init__(self, h, s, l):
        self.h = h
        self.s = s
        self.l = l
  
converters.register()
class RGBToHSL(Conversion):
    src = RGB
    dst = HSL
      
    @staticmethod
    def convert(rgb):
        hsl = colorsys.rgb_to_hsl(rgb.r, rgb.g, rgb.b)
        return HSL(*hsl)
        

converters.register()
class HSLToRGB(Conversion):
  src = RGB
  dst = HSL
  
  @staticmethod
  def convert(hsl):
      rgb = colorsys.hsl_to_rgb(hsl.h, hsl.s, hsl.l)
      return RGB(*rgb)
```

We can now write ``RGB(0.1, 0.3, 0.3).to(HSL)`` and get the correct result.

If we wanted we could connect these to the points conversion graph, or include part of the points graph in the color conversion graph. 





