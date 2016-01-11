import numpy
import six

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class NegativeSamplingFunction(function.Function):

    def __init__(self, sampler, sample_size):
        self.sampler = sampler
        self.sample_size = sample_size

    def _make_samples(self, t):
        if hasattr(self, 'samples'):
            return self.samples  # for testing

        size = int(t.shape[0])
        # first one is the positive, and others are sampled negatives
        samples = self.sampler((size, self.sample_size + 1))
        samples[:, 0] = t
        self.samples = samples

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 3)
        x_type, t_type, w_type = in_types

        type_check.expect(
            x_type.dtype == numpy.float32,
            x_type.ndim == 2,
            t_type.dtype == numpy.int32,
            t_type.ndim == 1,
            x_type.shape[0] == t_type.shape[0],
            w_type.dtype == numpy.float32,
            w_type.ndim == 2,
        )

    def forward_cpu(self, inputs):
        x, t, W = inputs
        self._make_samples(t)

        loss = numpy.float32(0.0)
        for ix, k in six.moves.zip(x, self.samples):
            w = W[k]
            f = w.dot(ix)
            f[0] *= -1  # positive sample
            loss += numpy.sum(numpy.logaddexp(f, 0))
        return numpy.array(loss, numpy.float32),

    def forward_gpu(self, inputs):
        x, t, W = inputs
        n_in = x.shape[1]
        self._make_samples(t)

        self.wx = cuda.elementwise(
            'raw T W, raw T x, S s, int32 n_units, int32 n_samples',
            'T wx',
            '''
            T f = 0;
            int b = i / n_samples;
            for (int j = 0; j < n_units; ++j) {
              int x_ind[] = {b, j};
              int w_ind[] = {s, j};
              f += x[x_ind] * W[w_ind];
            }
            wx = f;
            ''',
            'negative_sampling_forward_wx'
        )(W, x, self.samples, n_in, self.sample_size + 1)
        y = cuda.elementwise(
            'T wx, int32 n_samples', 'T y',
            '''
            T f = wx;
            if (i % n_samples == 0) {
              f = -f;
            }
            T loss;
            if (f < 0) {
              loss = log1pf(__expf(f));
            } else {
              loss = f + log1pf(__expf(-f));
            }
            y = loss;
            ''',
            'negative_sampling_forward_y'
        )(self.wx, self.sample_size + 1)
        loss = cuda.cupy.sum(y)
        return loss,

    def backward_cpu(self, inputs, grads):
        x, t, W = inputs
        gloss, = grads

        gx = numpy.zeros_like(x)
        gW = numpy.zeros_like(W)
        for i, (ix, k) in enumerate(six.moves.zip(x, self.samples)):
            w = W[k]
            f = w.dot(ix)

            # g == -y * gloss / (1 + exp(yf))
            f[0] *= -1
            g = gloss / (1 + numpy.exp(-f))
            g[0] *= -1

            gx[i] = g.dot(w)
            for ik, ig in six.moves.zip(k, g):
                gW[ik] += ig * ix
        return gx, None, gW

    def backward_gpu(self, inputs, grads):
        cupy = cuda.cupy
        x, t, W = inputs
        gloss, = grads

        n_in = x.shape[1]
        g = cupy.empty_like(self.wx)
        cuda.elementwise(
            'T wx, raw T gloss, int32 n_samples',
            'T g',
            '''
            T y;
            if (i % n_samples == 0) {
              y = 1;
            } else {
              y = -1;
            }
            g = -y * gloss[0] / (1.0f + __expf(wx * y));
            ''',
            'negative_sampling_calculate_g'
        )(self.wx, gloss, self.sample_size + 1, g)
        gW = cupy.zeros_like(W)
        cuda.elementwise(
            'T g, T x, S s, int32 n_samples, int32 n_units',
            'raw T gW',
            '''
            int w_ind[] = {s, i % n_units};
            atomicAdd(&gW[w_ind], g * x);
            ''',
            'negative_sampling_calculate_gw'
        )(cuda.cupy.expand_dims(g, 2),
          cuda.cupy.expand_dims(x, 1),
          cuda.cupy.expand_dims(self.samples, 2),
          self.sample_size + 1, n_in, gW)

        gx = cupy.empty_like(x)
        cuda.elementwise(
            'raw T g, raw T W, raw S s, int32 n_units, int32 n_samples', 'T gx',
            '''
            int b = i / n_units;
            int j = i - b * n_units;
            T w = 0;
            for (int k = 0; k < n_samples; ++k) {
              int s_ind[] = {b, k};
              int w_ind[] = {s[s_ind], j};
              w += g[s_ind] * W[w_ind];
            }
            gx = w;
            ''',
            'negative_sampling_calculate_gx'
        )(g, W, self.samples, n_in, self.sample_size + 1, gx)
        return gx, None, gW


def negative_sampling(x, t, W, sampler, sample_size):
    """Negative sampling loss function.

    In natural language processing, especially language modeling, the number of
    vocabulary is very large.
    Therefore, you need to spend a lot of time to calculate the gradient of the
    embedding matrix.

    Instead, in negative sampling trick, you only need to calculate the
    gradient for a few sampled negative examples.

    The objective function is below:

    .. math::

       f(x, p) = \\log \\sigma(x^\\top w_p) + \\
       k E_{i \\sim P(i)}[\\log \\sigma(- x^\\top w_i)],

    where :math:`\sigma(\cdot)` is a sigmoid function, :math:`w_i` is the
    weight vector for the word :math:`i`, and :math:`p` is a positive example.
    It is approximeted with :math:`k` examples :math:`N` sampled from
    probability :math:`P(i)`, like this:

    .. math::

       f(x, p) \\approx \\log \\sigma(x^\\top w_p) + \\
       \\sum_{n \\in N} \\log \\sigma(-x^\\top w_n).

    Each sample of :math:`N` is drawn from the word distribution :math:`P(w)`.
    This is calculated as :math:`P(w) = \\frac{1}{Z} c(w)^\\alpha`, where
    :math:`c(w)` is the unigram count of the word :math:`w`, :math:`\\alpha` is
    a hyper-parameter, and :math:`Z` is the normalization constant.

    Args:
        x (~chainer.Variable): Batch of input vectors.
        t (~chainer.Variable): Vector of groundtruth labels.
        W (~chainer.Variable): Weight matrix.
        sampler (function): Sampling function. It takes a shape and returns an
            integer array of the shape. Each element of this array is a sample
            from the word distribution. A :class:`~chainer.utils.WalkerAlias`
            object built with the power distribution of word frequency is
            recommended.
        sample_size (int): Number of samples.

    See: `Distributed Representations of Words and Phrases and their\
         Compositionality <http://arxiv.org/abs/1310.4546>`_

    .. seealso:: :class:`~chainer.links.NegativeSampling`.

    """
    return NegativeSamplingFunction(sampler, sample_size)(x, t, W)
