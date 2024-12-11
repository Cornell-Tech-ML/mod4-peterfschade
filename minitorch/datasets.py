import math
import random
from dataclasses import dataclass
from typing import List, Tuple


def make_pts(N: int) -> List:
    """_summary_
    generate a list of random points (2D)

    Args:
    ----
        N (int): number of points to be generated

    Returns:
    -------
        List: List of tuples of x,y points

    """
    X = []
    for i in range(N):
        x_1 = random.random()
        x_2 = random.random()
        X.append((x_1, x_2))
    return X


@dataclass
class Graph:
    """Graph holds the data for a simple graph dataset.

    Attributes
    ----------
    N : int
        Number of points in the graph.
    X : List[Tuple[float, float]]
        List of coordinates for the points.
    y : List[int]
        List of labels for the points.

    """

    N: int
    X: List[Tuple[float, float]]
    y: List[int]


def simple(N: int) -> Graph:
    """_summary_
    Generate a simple graph class dataset with N points that are labeled with vertical
    dividing line that classifies points

    Args:
    ----
        N (int): number of points to generate

    Returns:
    -------
        Graph: Graph class of N points with simple labeling of points
        where label == 1 if x point is less than 0.5

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def diag(N: int) -> Graph:
    """_summary_
    Generate a simple graph class dataset with N points that are labeled with vertical
    dividing line that classifies points

    Args:
    ----
        N (int): number of random points to be generated

    Returns:
    -------
        Graph: Graph class of N points with diagonal labeling of points
        where label == 1 if x + y is less than 0.5

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 + x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def split(N: int) -> Graph:
    """_summary_
    Generate a graph class dataset with N 2D points that are split labeled such that
    label = 1 if x < 0.2 or x > 0.8, otherwise 0

    Args:
    ----
        N (int): number of random points to be generated

    Returns:
    -------
        Graph: labeled graph struct
        labels: 1 if x < 0.2 or x > 0.8 else 0

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.2 or x_1 > 0.8 else 0
        y.append(y1)
    return Graph(N, X, y)


def xor(N: int) -> Graph:
    """_summary_
    Generate a graph class dataset with N 2D points that are split labeled such that
    label = (x < 0.5 and y > 0.5) or (x > 0.5 and y < 0.5) else 0

    Args:
    ----
        N (int): number of random points to be generated

    Returns:
    -------
        Graph: labeled dataset that is labeled by xor operation

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 and x_2 > 0.5 or x_1 > 0.5 and x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def circle(N: int) -> Graph:
    """_summary_
    Generate a graph class dataset that is labeled such that a circle
    of points is labeled 0 and 1 if outside that circle

    Args:
    ----
        N (int): number of random points to be generated

    Returns:
    -------
        Graph: labeled 2D data labeled by circle enclosure

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        x1, x2 = x_1 - 0.5, x_2 - 0.5
        y1 = 1 if x1 * x1 + x2 * x2 > 0.1 else 0
        y.append(y1)
    return Graph(N, X, y)


def spiral(N: int) -> Graph:
    """_summary_
    Generate a 2D labeled dataset of values that spirals in its labeling

    Args:
    ----
        N (int): number of points to be generated

    Returns:
    -------
        Graph: _description_

    """

    def x(t: float) -> float:
        """_summary_
        generate x point of a spiral and scale it down

        Args:
        ----
            t (float): input location

        Returns:
        -------
            float: x position of point

        """
        return t * math.cos(t) / 20.0

    def y(t: float) -> float:
        """_summary_
        generate y point of a spiral and scale it down

        Args:
        ----
            t (float): input location

        Returns:
        -------
            float: y location of a point

        """
        return t * math.sin(t) / 20.0

    X = [
        (x(10.0 * (float(i) / (N // 2))) + 0.5, y(10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    X = X + [
        (y(-10.0 * (float(i) / (N // 2))) + 0.5, x(-10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    y2 = [0] * (N // 2) + [1] * (N // 2)
    return Graph(N, X, y2)


datasets = {
    "Simple": simple,
    "Diag": diag,
    "Split": split,
    "Xor": xor,
    "Circle": circle,
    "Spiral": spiral,
}
