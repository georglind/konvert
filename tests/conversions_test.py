import unittest
import numpy as np

from konvert.conversions import ConversionGraph
from konvert.conversions import Graph
from konvert.conversions import ConnectionError


class Edge(object):
    """Mock edge object"""
    def __init__(self, src, dst, weight=1):
        self.src = src
        self.dst = dst
        self.weight = weight


class GraphTestCase(unittest.TestCase):
    """Helper for testing graphs"""
    def assertPathFollowed(self, path, nodes):
        """Check list of edges against list of nodes"""
        if len(path) != len(nodes) - 1:
            raise AssertionError('Path length "{}" incompatible with node length "{}"'.format(len(path), len(nodes)))
        elif len(path) == 0:
            self.assertEqual(len(nodes), 1)
        else:
            ref_nodes = [path[0].src] + [e.dst for e in path]
            self.assertEqual(nodes, ref_nodes)


class GraphTest(GraphTestCase):

    def setUp(self):
        g = Graph()
        for i in range(5):
            g.add_edge(Edge(i, (i + 1) % 5))

        g.add_edge(Edge(2, 4))
        g.add_edge(Edge(2, 1))
        g.add_edge(Edge(5, 2))
        g.add_edge(Edge(2, 5))
        g.add_edge(Edge(6, 0))

        self.graph = g

    def tearDown(self):
        del self.graph

    def test_shortest_path(self):
        graph = self.graph

        with self.assertRaisesRegex(IndexError, '7'):
            graph.shortest_path(3, 7)

        path = graph.shortest_path(0, 1)
        self.assertPathFollowed(path, [0, 1])

        path = graph.shortest_path(0, 4)
        self.assertPathFollowed(path, [0, 1, 2, 4])

        path = graph.shortest_path(2, 4)
        self.assertPathFollowed(path, [2, 4])

        path = graph.shortest_path(2, 3)
        self.assertPathFollowed(path, [2, 3])

        path = graph.shortest_path(3, 5)
        self.assertPathFollowed(path, [3, 4, 0, 1, 2, 5])

        with self.assertRaisesRegex(ConnectionError, '6'):
            graph.shortest_path(3, 6)

        # add long shortcut
        graph.add_edge(Edge(2, 7, 0.1))
        graph.add_edge(Edge(7, 0, 0.1))
        graph.add_edge(Edge(0, 3, 0.1))

        path = graph.shortest_path(2, 3)
        self.assertPathFollowed(path, [2, 7, 0, 3])


if __name__ == '__main__':
    unittest.main()
