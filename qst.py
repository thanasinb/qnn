import numpy
import util

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class QST(function.Function):

    """Quantized with Straight Thourgh estimator Unit."""

    def __init__(self):
        pass

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type, = in_types

        type_check.expect(
            x_type.dtype == numpy.float32,
        )

    def forward_cpu(self, x):
        xq = util._log_quant_cpu(x[0] * 64, 64)
        y = xq / float(64 ** 2)
        return y,

    def forward_gpu(self, x):
        xq = util._log_quant_gpu(x[0] * 64, 64)
        y = xq / float(64 ** 2)
        return y,

    def backward_cpu(self, x, gy):
        gx = gy[0].copy()
        zero_indices = numpy.abs(x[0]) > 1
        gx[zero_indices] = 0
        return gx,

    def backward_gpu(self, x, gy):
        gx = cuda.elementwise(
            'T x, T gy', 'T gx',
            'gx = abs(x) > 1 ? 0 : gy', 'bst_bwd')(
                x[0], gy[0])
        return gx,


def qst(x):
    """Quantized with Straight Thourgh estimator Unit function.

    This function is expressed as

    .. math::
        f(x) = \\left \\{ \\begin{array}{ll}
        1 & {\\rm if}~ x \\ge 0 \\\\
        -1 & {\\rm if}~ x < 0,
        \\end{array} \\right.

    See: http://arxiv.org/abs/1511.07289

    Args:
        x (~chainer.Variable): Input variable.

    Returns:
        ~chainer.Variable: Output variable.

    """
    return QST()(x)
