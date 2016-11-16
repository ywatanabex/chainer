"""
Wrap cuDNN Bidrectional GRU
"""
import binascii
import itertools
import os
import time

import numpy
import six

import chainer
from chainer import cuda
from chainer import function
from chainer.functions.array import concat
from chainer.functions.array import reshape
from chainer.functions.array import split_axis
from chainer.functions.array import stack
from chainer.functions.connection import linear
from chainer.functions.noise import dropout
from chainer.utils import type_check
import chainer.functions as F

from chainer.functions.connection.n_step_lstm import get_random_state, _make_tensor_descriptor_array

if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    libcudnn = cuda.cudnn.cudnn
    _cudnn_version = libcudnn.getVersion()

_random_states = {}


def _split(inputs, pos):
    return inputs[:pos], inputs[pos:]


class NStepBiGRU(function.Function):

    def __init__(self, n_layers, states, train=True):
        self.n_layers = n_layers
        self.train = train
        self.states = states

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() > 1 + 12 * self.n_layers * 2)
        (h_type,), in_types = _split(in_types, 1)
        w_types, in_types = _split(in_types, self.n_layers * 6 * 2)
        b_types, in_types = _split(in_types, self.n_layers * 6 * 2)
        x_types = in_types

        type_check.expect(
            h_type.dtype == numpy.float32,
            h_type.ndim == 3,
            h_type.shape[0] == self.n_layers * 2,
        )

        for x_type in x_types:
            type_check.expect(
                x_type.dtype == numpy.float32,
                x_type.ndim == 2,
            )
        for x1_type, x2_type in zip(x_types, x_types[1:]):
            type_check.expect(
                # Check if xs are sorted by descending lengths
                x1_type.shape[0] >= x2_type.shape[0],
                x1_type.shape[1] == x2_type.shape[1])

        in_size = x_types[0].shape[1]
        out_size = h_type.shape[2]
        for layer in six.moves.range(self.n_layers):
            for i in six.moves.range(6):
                for di in [0, 1]:
                    ind = (2 * layer + di) * 6 + i
                    w_type = w_types[ind]
                    b_type = b_types[ind]
                    if layer == 0 and i < 3:
                        w_in = in_size
                    elif layer > 0 and i < 3:
                        w_in = out_size * 2
                    else:
                        w_in = out_size

                    type_check.expect(
                        w_type.dtype == numpy.float32,
                        w_type.ndim == 2,
                        w_type.shape[0] == out_size,
                        w_type.shape[1] == w_in,

                        b_type.dtype == numpy.float32,
                        b_type.ndim == 1,
                        b_type.shape[0] == out_size,
                    )

    def forward(self, inputs):
        (hx,), inputs = _split(inputs, 1)  # shape = (2 * n_layers, batchsize, n_units)
        ws, inputs = _split(inputs, self.n_layers * 6 * 2)  # NOTE: 6 weights in each direction of GRU
        bs, inputs = _split(inputs, self.n_layers * 6 * 2)
        x_list = inputs

        hx = cuda.cupy.ascontiguousarray(hx)

        x_desc = cudnn.create_tensor_nd_descriptor(x_list[0][..., None])  # [..., None] add additional dim

        length = len(x_list)
        n_units = hx.shape[2]

        xs = cuda.cupy.concatenate(x_list, axis=0)
        ys = cuda.cupy.empty((len(xs), n_units * 2), dtype=xs.dtype)

        handle = cudnn.get_handle()
        self.handle = handle

        rnn_desc = cudnn.create_rnn_descriptor(
            n_units, self.n_layers, self.states.desc,
            libcudnn.CUDNN_LINEAR_INPUT,      # x is multiplied by a matrix in the first layer
            libcudnn.CUDNN_BIDIRECTIONAL,
            libcudnn.CUDNN_GRU, libcudnn.CUDNN_DATA_FLOAT)
        self.rnn_desc = rnn_desc

        c_x_descs = _make_tensor_descriptor_array(x_list)
        hx_desc = cudnn.create_tensor_nd_descriptor(hx)

        weights_size = libcudnn.getRNNParamsSize(
            handle, rnn_desc.value, x_desc.value, libcudnn.CUDNN_DATA_FLOAT)  # parameter size (in bytes)
        w = cuda.cupy.empty((weights_size // 4, 1, 1), dtype=numpy.float32)
        w_desc = cudnn.create_filter_descriptor(w)

        for layer in six.moves.range(self.n_layers):
            for di in [0, 1]:  # 0: forward, 1: backward
                for lin_layer_id in six.moves.range(6):
                    mat = cudnn.get_rnn_lin_layer_matrix_params(
                        handle, rnn_desc, (2 * layer + di), x_desc, w_desc, w,
                        lin_layer_id)
                    m = mat.reshape(mat.size)
                    m[...] = ws[(2 * layer + di) * 6 + lin_layer_id].ravel()
                    bias = cudnn.get_rnn_lin_layer_bias_params(
                        handle, rnn_desc, (2 * layer + di), x_desc, w_desc, w,
                        lin_layer_id)
                    b = bias.reshape(bias.size)
                    b[...] = bs[(2 * layer + di) * 6 + lin_layer_id]
        self.w = w
        self.w_desc = w_desc

        sections = numpy.cumsum([len(x) for x in x_list[:-1]])
        y_list = cuda.cupy.split(ys, sections)

        c_y_descs = _make_tensor_descriptor_array(y_list)
        hy = cuda.cupy.empty_like(hx)
        hy_desc = cudnn.create_tensor_nd_descriptor(hy)

        work_size = libcudnn.getRNNWorkspaceSize(
            handle, rnn_desc.value, length, c_x_descs.data)
        workspace = cuda.cupy.empty((work_size,), dtype='b')
        self.workspace = workspace

        if not self.train:
            libcudnn.RNNForwardInference(
                handle, rnn_desc.value, length,
                c_x_descs.data, xs.data.ptr, hx_desc.value, hx.data.ptr,
                #cx_desc.value, cx.data.ptr,
                0, 0,
                w_desc.value, w.data.ptr,
                c_y_descs.data, ys.data.ptr, hy_desc.value, hy.data.ptr,
                #cy_desc.value, cy.data.ptr,
                0, 0,
                workspace.data.ptr, work_size)

        else:
            reserve_size = libcudnn.getRNNTrainingReserveSize(
                handle, rnn_desc.value, length, c_x_descs.data)
            self.reserve_space = cuda.cupy.empty((reserve_size,), dtype='b')
            libcudnn.RNNForwardTraining(
                handle, rnn_desc.value, length,
                c_x_descs.data, xs.data.ptr, hx_desc.value, hx.data.ptr,
                #cx_desc.value, cx.data.ptr,
                0, 0,
                w_desc.value, w.data.ptr,
                c_y_descs.data, ys.data.ptr, hy_desc.value, hy.data.ptr,
                #cy_desc.value, cy.data.ptr,
                0, 0,
                workspace.data.ptr, work_size,
                self.reserve_space.data.ptr, reserve_size)

        self.c_y_descs = c_y_descs
        self.ys = ys
        self.c_x_descs = c_x_descs

        return tuple([hy,] + y_list)

    def backward(self, inputs, grads):
        (hx,), inputs = _split(inputs, 1)
        ws, inputs = _split(inputs, self.n_layers * 6 * 2)
        bs, inputs = _split(inputs, self.n_layers * 6 * 2)
        x_list = inputs
        n_units = hx.shape[2]

        hx = cuda.cupy.ascontiguousarray(hx)

        dhy, = grads[:1]
        dy_list = list(grads[1:])
        if dhy is None:
            dhy = cuda.cupy.zeros_like(hx)
        for i in six.moves.range(len(dy_list)):
            if dy_list[i] is None:
                #dy_list[i] = cuda.cupy.zeros_like(x_list[i])
                dy_list[i] = cuda.cupy.zeros((len(x_list[i]), n_units * 2), dtype=hx.dtype)  # NOTE: here is different from LSTM code!!!

        xs = cuda.cupy.concatenate(x_list, axis=0)
        length = len(x_list)

        dhx = cuda.cupy.empty_like(hx)

        hx_desc = cudnn.create_tensor_nd_descriptor(hx)
        dhy_desc = cudnn.create_tensor_nd_descriptor(dhy)

        c_dy_descs = _make_tensor_descriptor_array(dy_list)
        dys = cuda.cupy.concatenate(dy_list, axis=0)

        rnn_desc = self.rnn_desc
        handle = self.handle
        work_size = libcudnn.getRNNWorkspaceSize(
            handle, rnn_desc.value, length, self.c_x_descs.data)
        workspace = cuda.cupy.empty((work_size,), dtype='b')

        dhx_desc = cudnn.create_tensor_nd_descriptor(dhx)

        dxs = cuda.cupy.empty_like(xs)
        sections = numpy.cumsum([len(x) for x in x_list[:-1]])
        dx_list = cuda.cupy.split(dxs, sections, 0)
        c_dx_descs = _make_tensor_descriptor_array(dx_list)

        libcudnn.RNNBackwardData(
            handle, rnn_desc.value, length,
            self.c_y_descs.data, self.ys.data.ptr,
            c_dy_descs.data, dys.data.ptr,
            dhy_desc.value, dhy.data.ptr,
            #dcy_desc.value, dcy.data.ptr,
            0, 0,
            self.w_desc.value, self.w.data.ptr,
            hx_desc.value, hx.data.ptr,
            #cx_desc.value, cx.data.ptr,
            0, 0,
            c_dx_descs.data, dxs.data.ptr,
            dhx_desc.value, dhx.data.ptr,
            #dcx_desc.value, dcx.data.ptr,
            0, 0,
            workspace.data.ptr, work_size,
            self.reserve_space.data.ptr, self.reserve_space.size)

        dw = cuda.cupy.zeros_like(self.w)
        dw_desc = cudnn.create_tensor_nd_descriptor(dw)
        libcudnn.RNNBackwardWeights(
            handle, rnn_desc.value, length,
            self.c_x_descs.data, xs.data.ptr,
            hx_desc.value, hx.data.ptr, self.c_y_descs.data, self.ys.data.ptr,
            workspace.data.ptr, work_size, dw_desc.value, dw.data.ptr,
            self.reserve_space.data.ptr, self.reserve_space.size)

        dx = dx_list[0]
        dx = dx.reshape(dx.shape + (1,))
        dx_desc = cudnn.create_tensor_nd_descriptor(dx)
        dws = [cuda.cupy.empty_like(w) for w in ws]
        dbs = [cuda.cupy.empty_like(b) for b in bs]
        for layer in six.moves.range(self.n_layers):
            for di in [0, 1]:
                for lin_layer_id in six.moves.range(6):
                    mat = cudnn.get_rnn_lin_layer_matrix_params(
                        handle, rnn_desc, (2 * layer + di), dx_desc, dw_desc, dw,
                        lin_layer_id)
                    v = dws[(2 * layer + di) * 6 + lin_layer_id]
                    v = v.reshape(v.size)
                    v[:] = mat.ravel()
                    bias = cudnn.get_rnn_lin_layer_bias_params(
                        handle, rnn_desc, (2 * layer + di), dx_desc, dw_desc, dw,
                        lin_layer_id)
                    v = dbs[(2 * layer + di) * 6 + lin_layer_id]
                    v = v.reshape(v.size)
                    v[:] = bias.ravel()

        return tuple([dhx,] + dws + dbs + dx_list)


def _stack_weight(ws):
    w = stack.stack(ws, axis=0)  # NOTE: axis is different from LSTM case!!
    shape = w.shape
    return reshape.reshape(w, (shape[0] * shape[1],) + shape[2:])


def n_step_bigru(n_layers, dropout_ratio, hx, ws, bs, xs, train=True, use_cudnn=True):
    """
    """

    xp = cuda.get_array_module(hx, hx.data)

    if use_cudnn and xp is not numpy and cuda.cudnn_enabled and _cudnn_version >= 5000:
        states = get_random_state().create_dropout_states(dropout_ratio)
        # flatten all input variables
        inputs = tuple(itertools.chain(
            (hx,),
            itertools.chain.from_iterable(ws),
            itertools.chain.from_iterable(bs),
            xs))
        rnn = NStepBiGRU(n_layers, states, train=train)
        ret = rnn(*inputs)
        hy, = ret[:1]
        ys = ret[1:]
        return hy, ys

    else:
        hx = split_axis.split_axis(hx, n_layers * 2, axis=0, force_tuple=True)
        hx = [reshape.reshape(h, h.shape[1:]) for h in hx]

        xws = [_stack_weight([w[0], w[1], w[2]]) for w in ws]
        hws = [_stack_weight([w[3], w[4], w[5]]) for w in ws]
        xbs = [_stack_weight([b[0], b[1], b[2]]) for b in bs]
        hbs = [_stack_weight([b[3], b[4], b[5]]) for b in bs]

        batches = [x.shape[0] for x in xs]
        hy = []
        _xs = xs
        for layer in range(n_layers):
            # forward
            di = 0
            h = hx[2 * layer + di]
            hf = []
            for batch, x in zip(batches, _xs):
                if h.shape[0] > batch:
                    h, h_rest = split_axis.split_axis(h, [batch], axis=0)
                else:
                    h_rest = None
                if x.shape[0] > batch:
                    x, _ = split_axis.split_axis(x, [batch], axis=0)

                x = dropout.dropout(x, ratio=dropout_ratio, train=train)
                h = dropout.dropout(h, ratio=dropout_ratio, train=train)
                gru_in_x = linear.linear(x, xws[2 * layer + di], xbs[2 * layer + di])
                gru_in_h = linear.linear(h, hws[2 * layer + di], hbs[2 * layer + di])
                r_x, z_x, k_x = split_axis.split_axis(gru_in_x, 3, axis=1)
                r_h, z_h, k_h = split_axis.split_axis(gru_in_h, 3, axis=1)
                r = F.sigmoid(r_x + r_h)
                z = F.sigmoid(z_x + z_h)
                h_bar = z * h + (1 - z) * F.tanh(k_x + r * k_h)

                if h_rest is not None:
                    h = concat.concat([h_bar, h_rest], axis=0)
                else:
                    h = h_bar
                hf.append(h)
            hy.append(h)

            # backward
            di = 1
            h = hx[2 * layer + di]
            hb = []
            for batch, x in reversed(zip(batches, _xs)):
                if h.shape[0] > batch:
                    h, h_rest = split_axis.split_axis(h, [batch], axis=0)
                else:
                    h_rest = None
                if x.shape[0] > batch:
                    x, _ = split_axis.split_axis(x, [batch], axis=0)

                x = dropout.dropout(x, ratio=dropout_ratio, train=train)
                h = dropout.dropout(h, ratio=dropout_ratio, train=train)
                gru_in_x = linear.linear(x, xws[2 * layer + di], xbs[2 * layer + di])
                gru_in_h = linear.linear(h, hws[2 * layer + di], hbs[2 * layer + di])
                r_x, z_x, k_x = split_axis.split_axis(gru_in_x, 3, axis=1)
                r_h, z_h, k_h = split_axis.split_axis(gru_in_h, 3, axis=1)
                r = F.sigmoid(r_x + r_h)
                z = F.sigmoid(z_x + z_h)
                h_bar = z * h + (1 - z) * F.tanh(k_x + r * k_h)

                if h_rest is not None:
                    h = concat.concat([h_bar, h_rest], axis=0)
                else:
                    h = h_bar
                hb.append(h)
            hy.append(h)
            hb.reverse()
            _xs = [F.concat([hfi, hbi], axis=1) for (hfi, hbi) in zip(hf, hb)]

        hy = F.stack(hy)
        ys = [x[:batch, :] for (batch, x) in zip(batches, _xs)]
        return hy, tuple(ys)
