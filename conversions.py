import collections
import copy


class ConnectionError(Exception):
    pass


class Graph(object):
    """Directed graph representation"""

    @classmethod
    def merge(first, second):
        nodes = copy.copy(first.nodes)
        nodes.update(copy.copy(second.nodes))

        instance = cls()
        instance.nodes = nodes

        return instance

    def __init__(self):
        self.nodes = collections.defaultdict(set)

    def validate(self, node):
        if node not in self.nodes:
            raise IndexError('No conversions for {}.'.format(node))

    def add_edge(self, edge):
        self.nodes[edge.src].add(edge)
        self.nodes[edge.dst]

    def shortest_path(self, src, dst):
        """Find shortest path using djikstra."""
        self.validate(src)
        self.validate(dst)

        if src is dst:
            return []

        unvisited = set(node for node in self.nodes)

        dist = {node: float('inf') for node in self.nodes}
        prev = {node: None for node in self.nodes}

        dist[src] = 0

        while unvisited:
            node = min(unvisited, key=(lambda k: dist[k]))

            if node is dst:
                break

            unvisited.discard(node)

            for edge in self.nodes[node]:
                neighbor = edge.dst

                d = dist[node] + edge.weight
                if d < dist[neighbor]:
                    dist[neighbor] = d
                    prev[neighbor] = edge

        if prev[node] is None:
            raise ConnectionError('{} not connected to {}'.format(src, dst))

        path = []
        while node is not src:
            edge = prev[node]
            path.append(edge)
            node = edge.src

        return list(reversed(path))


class ConversionGraph(Graph):
    """Conversion manager"""
    def __init__(self):
        super(ConversionGraph, self).__init__()
        self._cache = {}

    def add_conversion(self, conversion):
        self.add_edge(conversion)

    def shortest_path_cached(self, src, dst):
        key = (src, dst)
        if key not in self._cache:
            self._cache[key] = self.shortest_path(src, dst)

        return self._cache[key]

    def convert(self, value, dst):
        path = self.shortest_path_cached(type(value), dst)

        for edge in path:
            value = edge.convert(value)

        return value

    def register(self):
        def register_conversion(klass):
            self.add_edge(klass)
            return klass
        return register_conversion


class Conversion(object):

    src = None
    dst = None
    weight = 1

    @staticmethod
    def convert(src):
        return src

