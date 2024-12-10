from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    # TODO: Implement for Task 1.
    vm = list(vals)
    vp = list(vals)
    vp[arg] += epsilon
    vm[arg] -= epsilon
    # import pdb; pdb.set_trace();
    return (f(*vp) - f(*vm)) / (2 * epsilon)

    # raise NotImplementedError("Need to implement for Task 1.1")


variable_count = 1


class Variable(Protocol):
    """Class variable holder"""

    def accumulate_derivative(self, x: Any) -> None:
        """Add up the derivatives of the inputs"""
        ...

    @property
    def unique_id(self) -> int:
        """Get unique identifier"""
        ...

    def is_leaf(self) -> bool:
        """Check if leaf"""
        ...

    def is_constant(self) -> bool:
        """Check if current variable is a constant"""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Get list of parents"""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Apply chain rule"""
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    # TODO: Implement for Task 1.4.
    orderedList = []
    visited = []

    def look(var: Variable) -> None:
        if var.unique_id in visited or var.is_constant():
            return
        if not var.is_leaf():
            for inp in var.parents:
                if not inp.is_constant():
                    look(inp)
        visited.append(var.unique_id)
        orderedList.insert(0, var)

    look(variable)
    return orderedList

    # raise NotImplementedError("Need to implement for Task 1.4")


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
    ----
        variable: The right-most variable
        deriv: Its derivative that we want to propagate backward to the leaves.

    Returns:
    -------
    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.

    """
    # run top sort
    sortedList = topological_sort(variable)

    derivatives = {}
    derivatives[variable.unique_id] = deriv

    for v in sortedList:
        deriv = derivatives[v.unique_id]

        if v.is_leaf():
            v.accumulate_derivative(deriv)
        else:
            for p, d in v.chain_rule(deriv):
                derivatives.setdefault(p.unique_id, 0.0)
                derivatives[p.unique_id] += d

    # TODO: Implement for Task 1.4.
    # raise NotImplementedError("Need to implement for Task 1.4")


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Return saved tensor values"""
        return self.saved_values
