from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
import tensorflow as tf


class EveOptimizer(optimizer.Optimizer):
    """Optimizer that implements the Eve algorithm.

    See [Koushik et. al., 2016](https://arxiv.org/abs/1611.01505)
    ([pdf](https://arxiv.org/abs/1611.01505.pdf)).

    @@__init__
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, beta3=0.999, k=0.1, K=10, use_locking=False, name="Eve"):
        super(EveOptimizer, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._beta3 = beta3
        self._k = k
        self._K = K

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._beta1_t = None
        self._beta2_t = None
        self._beta3_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._beta1_t = ops.convert_to_tensor(self._beta1, name="beta1")
        self._beta2_t = ops.convert_to_tensor(self._beta2, name="beta2")
        self._beta3_t = ops.convert_to_tensor(self._beta3, name="beta3")
        self.t = tf.Variable(1., trainable=False)
        self.f_hat = tf.Variable(1., 'self.f_hat')

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)
            self._zeros_slot(v, "d", self._name)

    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        beta3_t = math_ops.cast(self._beta3_t, var.dtype.base_dtype)
        eps = 1e-8
        t = self.t
        m = self.get_slot(var, "m")
        m_t = m.assign(beta1_t * m + (1. - beta1_t) * grad)
        m_hat = m_t / (1. - tf.pow(beta1_t, t))
        v = self.get_slot(var, "v")
        v_t = v.assign(beta2_t * v + (1. - beta2_t) * grad * grad)
        v_hat = v_t / (1. - tf.pow(beta2_t, t))

        loss = tf.get_collection('losses')[-1]

        def _smaller_than_previous():
            result = tf.constant(self._k + 1, dtype = tf.float32), tf.constant(self._K + 1, dtype = tf.float32)
            return result
        def _larger_than_previous():
            result = tf.constant(1. / (self._K + 1) ), tf.constant(1. / (self._k + 1))
            return result

        # Calculate deltas
        g_e = tf.greater_equal(loss, self.f_hat)
        delta1_t, delta2_t = tf.cond(g_e, _smaller_than_previous, _larger_than_previous)

        # Clip parameter
        c_t = tf.minimum(tf.maximum(delta1_t, tf.div(loss, self.f_hat)), delta2_t)

        # Smoothly tracked objection function
        # self.f_hat * (c_t - 1) = self.f_hat_1 - self.f_hat_2 where self.f_hat_1 = self.f_hat_2 * c_t
        #  r_t = tf.div(tf.abs(self.f_hat * (c_t - 1)), tf.minimum(c_t * self.f_hat, self.f_hat))
        r_t = tf.div(tf.abs((c_t - 1)), tf.minimum(c_t, 1))

        self.f_hat = (c_t * self.f_hat)
        d = self.get_slot(var, "d")

        d_t = tf.cond(t > 1, lambda: beta3_t * d + (1. - beta3_t) * r_t, lambda: 1.0)
        g_t = m_hat / (tf.sqrt(v_hat + eps))
        g_t_eve = g_t / d_t

        d.assign(d_t)
        self.t.assign_add(1.0)

        var_update = state_ops.assign_sub(var, lr_t * g_t_eve)
        return control_flow_ops.group(*[var_update, m_t, v_t])

     def _apply_sparse(self, grad, var):
        return self._apply_dense(grad, var)
