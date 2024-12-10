from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """_summary_
        Apply a function to a scalar, or set of Scalars
        Returns:
            Scalar: Scalar with saved information

        """
        raw_vals = []
        scalars = []

        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                c = minitorch.scalar.Scalar(v)
                # c.history = None
                scalars.append(c)
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """_summary_
        Add two scalars

        Args:
        ----
            ctx (Context): context
            a (float): scalar
            b (float): scalar

        Returns:
        -------
            float: new scalar a+b

        """
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """_summary_
        Returns derivative of addition

        Args:
        ----
            ctx (Context): context
            d_output (float): back_derivative

        Returns:
        -------
            Tuple[float, ...]: d_output, d_output

        """
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """_summary_
        Take log of input scalar

        Args:
        ----
            ctx (Context): context
            a (float): scalar

        Returns:
        -------
            float: new scalar log(a)

        """
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """_summary_
        derivative of ln(a) * back_derivative

        Args:
        ----
            ctx (Context): context
            d_output (float): back_derivative

        Returns:
        -------
            float: derivative

        """
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


class Mul(ScalarFunction):
    """Addition function $f(x, y) = x * y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """_summary_
        multiply two scalars

        Args:
        ----
            ctx (Context): context
            a (float): scalar
            b (float): scalar

        Returns:
        -------
            float: new scalar a * b

        """
        ctx.save_for_backward(a, b)
        return operators.mul(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """_summary_
        return back derivative of multiplying a and b

        Args:
        ----
            ctx (Context): _description_
            d_output (float): back_derivative

        Returns:
        -------
            Tuple[float, float]: b * d_output, a * d_output

        """
        (a, b) = ctx.saved_values
        return (b * d_output, a * d_output)


class Inv(ScalarFunction):
    """computes 1/x"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """_summary_
        Compute 1/a

        Args:
        ----
            ctx (Context): context
            a (float): Scalar input

        Returns:
        -------
            float: Inv(a): (1/a)

        """
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """_summary_

        Args:
        ----
            ctx (Context): context
            d_output (float): back_derivative

        Returns:
        -------
            float: derivative

        """
        (a,) = ctx.saved_values
        return operators.inv_back(a, d_output)


class Neg(ScalarFunction):
    """Negate a scalar"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """_summary_
        Negates a Scalar

        Args:
        ----
            ctx (Context): context
            a (float): Scalar

        Returns:
        -------
            float: Neg (scalar)

        """
        return operators.neg(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """_summary_
        derivative of -x times a back_derivative

        Args:
        ----
            ctx (Context): context
            d_output (float): back_derivative

        Returns:
        -------
            float: derivative

        """
        return -d_output


class Sigmoid(ScalarFunction):
    """Apply a sigmoid function to input"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """_summary_
        Apply a sigmoid function to input

        Args:
        ----
            ctx (Context): context
            a (float): Scalar

        Returns:
        -------
            float: sigmoid(a)

        """
        f = operators.sigmoid(a)
        ctx.save_for_backward(f)
        return f

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """_summary_

        Args:
        ----
            ctx (Context): context
            d_output (float): back_derivative

        Returns:
        -------
            float: derivative of sigmoid * back_derivative

        """
        (a,) = ctx.saved_values
        return a * (1 - a) * d_output


class ReLU(ScalarFunction):
    """Apply ReLU to input"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """_summary_
        Apply a ReLU to input

        Args:
        ----
            ctx (Context): context
            a (float): input Scalar

        Returns:
        -------
            float: Scalar(0) if a < 0, else Scalar(a)

        """
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """_summary_
        Compute back derivative of ReLU * back_derivative

        Args:
        ----
            ctx (Context): context
            d_output (float): back derivative

        Returns:
        -------
            float: derivative

        """
        (a,) = ctx.saved_values
        return operators.relu_back(a, d_output)


class Exp(ScalarFunction):
    """Compute e^(x) of Scalar"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """_summary_
        returns e^(input)

        Args:
        ----
            ctx (Context): context
            a (float): input Scalar

        Returns:
        -------
            float: Scalar (exp(a))

        """
        f = operators.exp(a)
        ctx.save_for_backward(f)
        return f

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """_summary_
        derivative of exp(a) * back derivative

        Args:
        ----
            ctx (Context): Context with saved values
            d_output (float): back derivative

        Returns:
        -------
            float: derivative of Scalar exp(input) * back derivative

        """
        (a,) = ctx.saved_values
        return a * d_output


class LT(ScalarFunction):
    """check if one Scalar is less than the other"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """_summary_
        check if two scalars a < b

        Args:
        ----
            ctx (Context): context
            a (float): scalar
            b (float): scalar

        Returns:
        -------
            float: 1.0 if a < b, 0.0 otherwise

        """
        return operators.lt(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """_summary_

        Args:
        ----
            ctx (Context): context
            d_output (float): back derivative

        Returns:
        -------
            Tuple[float, float]: [0.0, 0.0] derivative

        """
        return (0.0, 0.0)


class EQ(ScalarFunction):
    """Check if two scalars are equal"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """_summary_
        check if two scalars are equal

        Args:
        ----
            ctx (Context): context
            a (float): scalar
            b (float): scalar

        Returns:
        -------
            float: new scalar, 1.0 if True, else 0.0

        """
        return operators.eq(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """_summary_

        Args:
        ----
            ctx (Context): context
            d_output (float): back derivative

        Returns:
        -------
            Tuple[float, float]: [0.0, 0.0] derivative

        """
        return (0.0, 0.0)


# To implement.


# TODO: Implement for Task 1.2.
