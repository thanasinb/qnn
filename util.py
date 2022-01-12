import numpy
from chainer import cuda

# def _ap2_cpu(x):
def _ap2(x):
    return numpy.sign(x) * (2 ** numpy.round(numpy.log2(numpy.abs(x))))

def _log_quant_cpu(x, fsr):
    y = x.copy()
    idx = numpy.where(y != 0)
    y[idx] = numpy.clip(_ap2(y[idx]), -fsr, fsr)
    return y

def _log_quant_gpu(x, fsr):
    return cuda.elementwise(
        'T x, T f', 'T y',
        '''
            if(x == 0) {
                y = 0;
            } else {
                T sign_x = (x >= 0 ? 1 : -1);
                T px = sign_x * pow((T)2, round(log2(abs(x))));;
                if(px < -f) px = -f;
                else if(px > f) px = f;
                y = px;
            }
        ''',
        'log_quant')(x, fsr)


