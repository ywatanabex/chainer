import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer.functions.connection import convolution_2d
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


def _asfortranarray(x):
    xp = cuda.get_array_module(x)
    if xp is numpy:
        return xp.asfortranarray(x)
    else:
        return xp.ascontiguousarray(x.T).T


@testing.parameterize(
    {'h': 4, 'w': 3, 'pad': 1, 'c_contiguous': True},
    {'h': 4, 'w': 3, 'pad': 1, 'c_contiguous': False},
    {'h': 10, 'w': 8, 'pad': -1, 'c_contiguous': True},
)
class TestConvolution2DFunction(unittest.TestCase):

    def setUp(self, use_cudnn=True):
        in_channels = 3
        out_channels = 2
        kh, kw = (3, 3)
        self.stride = 2
        self.pad = 1
        self.use_cudnn = use_cudnn
        self.W = numpy.random.normal(
            0, numpy.sqrt(1. / (kh * kw * in_channels)),
            (out_channels, in_channels, kh, kw)).astype(numpy.float32)
        self.b = numpy.random.uniform(
            -1, 1, out_channels).astype(numpy.float32)

        oh = (self.h + 2 * self.pad - 3) // 2 + 1
        ow = (self.w + 2 * self.pad - 3) // 2 + 1
        shape = (2, 3, self.h, self.w)
        out_shape = (2, 2, oh, ow)
        self.x = numpy.random.uniform(-1, 1, shape).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, out_shape).astype(numpy.float32)

    @attr.cudnn
    def test_forward_consistency(self, nobias=False):
        x_cpu = chainer.Variable(self.x)
        W_cpu = chainer.Variable(self.W)
        b_cpu = None if nobias else chainer.Variable(self.b)
        y_cpu = functions.convolution_2d(
            x_cpu, W_cpu, b_cpu, stride=self.stride, pad=self.pad,
            use_cudnn=self.use_cudnn)

        x_gpu = chainer.Variable(cuda.to_gpu(self.x))
        W_gpu = chainer.Variable(cuda.to_gpu(self.W))
        b_gpu = None if nobias else chainer.Variable(cuda.to_gpu(self.b))
        y_gpu = functions.convolution_2d(
            x_gpu, W_gpu, b_gpu, stride=self.stride, pad=self.pad,
            use_cudnn=self.use_cudnn)

        gradient_check.assert_allclose(y_cpu.data, y_gpu.data.get())

    @attr.gpu
    def test_forward_consistency_im2col(self):
        self.use_cudnn = False
        self.test_forward_consistency()

    @attr.gpu
    def test_forward_consistency_im2col_nobias(self):
        self.use_cudnn = False
        self.test_forward_consistency(nobias=True)

    def check_backward(self, x_data, W_data, b_data, y_grad):
        if not self.c_contiguous:
            x_data = _asfortranarray(x_data)
            W_data = _asfortranarray(W_data)
            y_grad = _asfortranarray(y_grad)
            self.assertFalse(x_data.flags.c_contiguous)
            self.assertFalse(W_data.flags.c_contiguous)
            self.assertFalse(y_grad.flags.c_contiguous)
            if b_data is not None:
                xp = cuda.get_array_module(b_data)
                b = xp.empty((len(b_data) * 2,), dtype=self.b.dtype)
                b[::2] = b_data
                b_data = b[::2]
                self.assertFalse(b_data.flags.c_contiguous)

        args = (x_data, W_data)
        if b_data is not None:
            args = args + (b_data,)

        gradient_check.check_backward(
            convolution_2d.Convolution2DFunction(
                self.stride, self.pad, self.use_cudnn),
            args, y_grad, eps=1e-2)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.W, self.b, self.gy)

    @condition.retry(3)
    def test_backward_cpu_nobias(self):
        self.check_backward(self.x, self.W, None, self.gy)

    @attr.cudnn
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.W),
                            cuda.to_gpu(self.b), cuda.to_gpu(self.gy))

    @attr.cudnn
    @condition.retry(3)
    def test_backward_gpu_nobias(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.W),
                            None, cuda.to_gpu(self.gy))

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu_im2col(self):
        self.use_cudnn = False
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.W),
                            cuda.to_gpu(self.b), cuda.to_gpu(self.gy))

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu_im2col_nobias(self):
        self.use_cudnn = False
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.W),
                            None, cuda.to_gpu(self.gy))


testing.run_module(__name__, __file__)
