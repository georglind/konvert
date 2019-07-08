import numpy as np
import collections


class ProjectionCollection(object):
    """Projection manager"""
    def __init__(self):
        self._projections = collections.defaultdict(lambda : collections.defaultdict(list))

    def add_projection(self, p):
        """
        Add projection
        """
        self._projections[p.src][p.dst].append(p)

    def project(self, value, dst, *args, **kwargs):
        """
        Project a value on a destination type using passing arguments to the conversion.
        """
        src = type(value)
        if src not in self._projections:
            raise TypeError(f'No projections registered for {src}')

        if dst not in self._projections[src]:
            raise TypeError(f'No projections registered for {src} to {dst}')

        ps = self._projections[src][dst]

        if len(ps) > 1:
            raise TypeError(f'More than one projection registered for {src} to {dst} projection: {ps}')

        return ps[0].project(value, *args, **kwargs)

    def register(self):
        """
        Registration decorator
        """
        def register_projection(klass):
            self.add_projection(klass)
            return klass
        return register_projection


class Projection(object):
    """Generic projection."""
    src = None
    dst = None

    @staticmethod
    def project(value):
        return value
