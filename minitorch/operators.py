"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable
#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.


def mul(x: float, y: float) -> float:
    """_summary_
    returns the product of the two input arguments

    Args:
    ----
        x (float): number 1
        y (float): number 2

    Returns:
    -------
        float: product of x and y

    """
    return x * y


def id(x: float) -> float:
    """_summary_
    returns input unchanged

    Args:
    ----
        x (float): input

    Returns:
    -------
        float: returns input unchanged

    """
    return x


def add(x: float, y: float) -> float:
    """_summary_
    returns the sum of the two arguments

    Args:
    ----
        x (float): number 1
        y (float): number 2

    Returns:
    -------
        float: sum of two inputs

    """
    return x + y


def neg(x: float) -> float:
    """_summary_
    negates a number

    Args:
    ----
        x (float): number input

    Returns:
    -------
        float: negative of input

    """
    return -x


def lt(x: float, y: float) -> float:
    """_summary_
    check if first input is less than the second

    Args:
    ----
        x (float): number 1
        y (float): comparator number

    Returns:
    -------
        float: 1 if first input is less than the second
                0 if the first input is greater or equal to the second

    """
    if x < y:
        return 1.0
    else:
        return 0.0


def eq(x: float, y: float) -> float:
    """_summary_
    checks if two inputs are equal

    Args:
    ----
        x (float): number 1
        y (float): number 2

    Returns:
    -------
        float: 1 if equal, 0 otherwise

    """
    if x == y:
        return 1.0
    else:
        return 0.0


def max(x: float, y: float) -> float:
    """_summary_
    return max of two numbers

    Args:
    ----
        x (float): comparator number
        y (float): comparator number

    Returns:
    -------
        float: the max of two number inputs

    """
    if x > y:
        return x
    else:
        return y


def is_close(x: float, y: float) -> float:
    """_summary_
    check if two numbers are close

    Args:
    ----
        x (float): comparator number
        y (float): comparator number

    Returns:
    -------
        float: 1 if less than 1e-2 away

    """
    if abs((x - y) < 1e-2):
        return 1.0
    else:
        return 0.0


def sigmoid(x: float) -> float:
    """_summary_
    compute sigmoid function of input

    Args:
    ----
        x (float): input number

    Returns:
    -------
        float: returns sigmoid of input

    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """_summary_
    return relu activation, max (0,input)

    Args:
    ----
        x (float): comparator number

    Returns:
    -------
        float: returns x if x > 0, else returns 0

    """
    if x > 0:
        return x
    else:
        return 0.0


epsilon = 1e-10


def log(x: float) -> float:
    """_summary_
    return natural log of input

    Args:
    ----
        x (float): input number

    Returns:
    -------
        float: ln(x)
        adds very small epsilon to avoid undefined

    """
    return math.log(x + epsilon)


def exp(x: float) -> float:
    """_summary_
    returns e^(input)

    Args:
    ----
        x (float): input x

    Returns:
    -------
        float: returns e^x

    """
    return math.exp(x)


def log_back(x: float, y: float) -> float:
    """_summary_
    computes derivate of ln(x) * second argument

    Args:
    ----
        x (float): input to compute derivative of ln(x)
        y (float): multiplier

    Returns:
    -------
        float: derivate of ln(x) * second argument

    """
    return (1.0 / (x + epsilon)) * y


def inv(x: float) -> float:
    """_summary_
    returns the reciprocal 1/(input)

    Args:
    ----
        x (float): number

    Returns:
    -------
        float: 1 / (x + epsilon)
        epsilon is very small

    """
    return 1.0 / (x + epsilon)


def inv_back(x: float, y: float) -> float:
    """_summary_
    returns derivative of 1/x times a second number

    Args:
    ----
        x (float): input point to compute derivate
        y (float): multiplier

    Returns:
    -------
        float: returns derivative of 1/x times a second number

    """
    return -1.0 / (x**2) * y


def relu_back(x: float, y: float) -> float:
    """_summary_
    returns derivative of relu times a second argument

    Args:
    ----
        x (float): input point to compute derivate
        y (float): multiplier

    Returns:
    -------
        float: returns derivative of relu times a second argument

    """
    if x < 0:
        return 0.0
    else:
        return y


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """_summary_
    maps a function (input) to each of the elements in a list

    Args:
    ----
        fn (Callable[[float], float]): function to apply to each element,
        function takes in a float and outputs a float

    Returns:
    -------
        Callable[[Iterable[float]], List[float]]: A function that takes an iterable and returns a list.

    """

    def map_(ls: Iterable[float]) -> Iterable[float]:
        """_summary_
        apply a function to each element of a list

        Args:
        ----
            ls (Iterable[float]): list to be applied

        Returns:
        -------
            Iterable[float]: fn(x) for each x in the list of same size as input

        """
        return [fn(x) for x in ls]

    return map_


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """_summary_
    apply function to each paired element in two lists

    Args:
    ----
        fn (Callable[[float, float], float]): choice of function that takes two floats inputs of the same size, and outputs a float

    Returns:
    -------
        Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
        callable that takes in two iterables and returns an iterable

    """

    def zipWith_(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        """_summary_
        take two input lists and apply function to each paired input

        Args:
        ----
            ls1 (Iterable[float]): list 1
            ls2 (Iterable[float]): list 2

        Returns:
        -------
            Iterable[float]: function applied to each list, of size ls1

        """
        return [fn(x1, x2) for x1, x2 in zip(ls1, ls2)]

    return zipWith_


def reduce(
    fn: Callable[[float, float], float], init: float
) -> Callable[[Iterable[float]], float]:
    """_summary_
    apply a function to a sequence of numbers

    Args:
    ----
        fn (Callable[[float, float], float]): function of choice that accepts a single iterable type
        init (float): initialize input of reduce

    Returns:
    -------
        fn (Callable[[Iterable[float]], float]): function that accepts 2 input lists and outputs a float

    """

    def reduce_(ls: Iterable[float]) -> float:
        """_summary_
        starting with an initial value, apply a function to a list

        Args:
        ----
            ls (Iterable[float]): list to be reduced

        Returns:
        -------
            float: single reduced number of the function(list)

        """
        res = init
        for x in ls:
            res = fn(res, x)
        return res

    return reduce_


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """_summary_
    add elements of two lists together element-wise

    Args:
    ----
        ls1 (Iterable[float]): first list
        ls2 (Iterable[float]): second list

    Returns:
    -------
        Iterable[float]: returns the sum of the two lists of size ls1

    """
    zip_add = zipWith(add)
    return zip_add(ls1, ls2)


def negList(ls: Iterable[float]) -> Iterable[float]:
    """_summary_
    negates a list using map(neg)
    applies neg to each element of the list, returning a list of same size

    Args:
    ----
        ls (Iterable[float]): list of numbers

    Returns:
    -------
        Iterable[float]: negate of input list of same size as ls

    """
    mapNeg = map(neg)
    return mapNeg(ls)


def sum(ls: Iterable[float]) -> float:
    """_summary_
    adds up the numbers in a list using reduce

    Args:
    ----
        ls (Iterable[float]): list of numbers

    Returns:
    -------
        float: sum of a list using reduce(add)

    """
    reduce_add = reduce(add, 0)
    return reduce_add(ls)


def prod(ls: Iterable[float]) -> float:
    """_summary_
    takes the product of all the numbers in a list using reduce

    Args:
    ----
        ls (Iterable[float]): list of numbers

    Returns:
    -------
        float: product of all the numbers in list

    """
    reduce_mul = reduce(mul, 1)
    return reduce_mul(ls)
