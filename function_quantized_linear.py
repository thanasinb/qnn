import numpy
import util

from chainer import cuda
from chainer import function
from chainer.utils import type_check

def _as_mat(x):
    if x.ndim == 2:
        return x
    return x.reshape(len(x), -1)

class QuantizedLinearFunction(function.Function):

    def check_type_forward(self, in_types):
        n_in = in_types.size()
        type_check.expect(2 <= n_in, n_in <= 3)
        x_type, w_type = in_types[:2]

        type_check.expect(
            x_type.dtype == numpy.float32,
            w_type.dtype == numpy.float32,
            x_type.ndim >= 2,
            w_type.ndim == 2,
            type_check.prod(x_type.shape[1:]) == w_type.shape[1],
        )
        # if n_in.eval() == 3:
        if type_check.eval(n_in) == 3:
            b_type = in_types[2]
            type_check.expect(
                b_type.dtype == numpy.float32,
                b_type.ndim == 1,
                b_type.shape[0] == w_type.shape[0],
            )

    def forward_cpu(self, inputs):
        x = _as_mat(inputs[0])
        W = inputs[1]

        # quantized x, w
        xq = util._log_quant_cpu(x * 64, 64)
        Wq = util._log_quant_cpu(W * 64, 64)
        y = xq.dot(Wq.T) / float(64 ** 2)
        if len(inputs) == 3:
            b = inputs[2]
            y += b
        return y,

    def forward_gpu(self, inputs):
        x = _as_mat(inputs[0])
        W = inputs[1]
        xq = util._log_quant_gpu(x * 64, 64)
        Wb = util._log_quant_gpu(W * 64, 64)
        y = x.dot(Wb.T)
        if len(inputs) == 3:
            b = inputs[2]
            y += b
        return y,

    def backward_cpu(self, inputs, grad_outputs):
        x = _as_mat(inputs[0])
        W = inputs[1]

        gy = grad_outputs[0]

        # quantized x, w
        xq = util._log_quant_cpu(x * 64, 64)
        Wq = util._log_quant_cpu(W * 64, 64)
        gyq = util._log_quant_cpu(gy * 64, 64)

        gx = gyq.dot(Wq).reshape(inputs[0].shape) / float(64 ** 2)
        gW = gyq.T.dot(xq) / float(64 ** 2)
        if len(inputs) == 3:
            gb = gy.sum(0)
            return gx, gW, gb
        else:
            return gx, gW

    def backward_gpu(self, inputs, grad_outputs):
        x = _as_mat(inputs[0])
        W = inputs[1]

        gy = grad_outputs[0]

        # quantized x, w
        xq = util._log_quant_gpu(x * 64, 64)
        Wq = util._log_quant_gpu(W * 64, 64)
        gyq = util._log_quant_gpu(gy * 64, 64)

        gx = gyq.dot(Wq).reshape(inputs[0].shape) / float(64 ** 2)
        gW = gyq.T.dot(xq) / float(64 ** 2)
        if len(inputs) == 3:
            gb = gy.sum(0)
            return gx, gW, gb
        else:
            return gx, gW


def quantized_linear(x, W, b=None):
    """Quantized Linear function, or affine transformation.

    It accepts two or three arguments: an input minibatch ``x``, a weight
    matrix ``W``, and optionally a bias vector ``b``. It computes
    :math:`Y = xW^\\top + b`.

    Args:
        x (~chainer.Variable): Input variable. Its first dimension is assumed
            to be the *minibatch dimension*. The other dimensions are treated
            as concatenated one dimension whose size must be ``N``.
        W (~chainer.Variable): Weight variable of shape ``(M, N)``.
        b (~chainer.Variable): Bias variable (optional) of shape ``(M,)``..

    Returns:
        ~chainer.Variable: Output variable.

    .. seealso:: :class:`~chainer.links.Linear`

    """
    if b is None:
        return QuantizedLinearFunction()(x, W)
    else:
        return QuantizedLinearFunction()(x, W, b)
