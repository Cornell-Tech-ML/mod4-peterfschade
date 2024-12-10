from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.
    
    nh = height // kh
    nw = width // kw
    
    new_tensor = input.contiguous().view(batch, channel, height, nw, kw).permute(0,1, 3, 2, 4).contiguous().view(batch, channel, nh, nw, kh * kw)
    
    return (new_tensor, nh, nw)
    
    #raise NotImplementedError("Need to implement for Task 4.3")

def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply average pooling to input

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width

    """
    batch, channel, height, width = input.shape
    reshaped_inp, nh, nw = tile(input, kernel)
    pooled = reshaped_inp.mean(dim=4).contiguous().view(batch, channel, nh, nw)
    return pooled

reduce_max = FastOps.reduce(operators.max,0)


def argmax(input: Tensor, dim:int) -> Tensor:
    """Compute the argmax as a 1-hot tensor

    Args:
    ----
        input: batch x channel x height x width
        dim: dimension to compute the argmax

    Returns:
    -------
        Tensor of size batch x channel x height x width with a 1 at the argmax and 0 elsewhere.

    """
    return reduce_max(input,dim) == input
    #raise NotImplementedError("Need to implement for Task 4.4")

class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """Find the max value in the tensor along a given dimension"""
        ctx.save_for_backward(input,dim)
        return reduce_max(input,int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output:Tensor) -> Tuple[Tensor, float]:
        """Argmax is the backward of max"""
        input,dim = ctx.saved_tensors
        
        return argmax(input,int(dim.item())) * grad_output, 0.0
    
def max(input: Tensor, dim: int) -> Tensor:
    """Apply max reduction"""
    return Max.apply(input,input._ensure_tensor(dim))

def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax as a tensor

    Args:
    ----
        input: batch x channel x height x width
        dim: dimension to compute the softmax

    Returns:
    -------
        Tensor of size batch x channel x height x width with the softmax applied.

    """
    sm = input.exp()
    tot = sm.sum(dim=dim)
    return sm / tot
    
    #raise NotImplementedError("Need to implement for Task 4.4")

def logsoftmax(input: Tensor, dim:int) -> Tensor:
    """Compute the log of the softmax as a tensor

    Args:
    ----
        input: batch x channel x height x width
        dim: dimension to compute the log softmax

    Returns:
    -------
        Tensor of size batch x channel x height x width with the log of the softmax applied.

    """
    #sm = input.exp()
    #tot = sm.sum(dim=dim)
    #return (sm / tot).log()
    e = (input - reduce_max(input,dim)).exp()
    return (e / e.sum(dim=dim)).log()

    #raise NotImplementedError("Need to implement for Task 4.4")

def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply max pooling to input

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width

    """
    batch, channel, height, width = input.shape
    reshaped_inp, nh, nw = tile(input, kernel)
    pooled = max(reshaped_inp,dim=4).contiguous().view(batch, channel, nh, nw)
    return pooled

def dropout(input: Tensor, p: float = 0.5, ignore: bool = False) -> Tensor:
    """Dropout positions based on random noise

    Args:
    ----
        input: batch x channel x height x width
        p: probability of dropout
        ignore: ignore dropout

    Returns:
    -------
        Tensor of size batch x channel x height x width with dropout applied.
        
    """
    if ignore:
        return input
    else:
        return input * (rand(input.shape) > p)
    #raise NotImplementedError("Need to implement for Task 4.4")
    
    
# TODO: Implement for Task 4.3.
