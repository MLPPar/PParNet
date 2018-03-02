from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
import tensorflow as tf


class EveOptimizer(optimizer.Optimizer):
    """Optimizer that implements the Adamax algorithm.

    See [Kingma et. al., 2014](http://arxiv.org/abs/1412.6980)
    ([pdf](http://arxiv.org/pdf/1412.6980.pdf)).

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
        self.t = tf.Variable(0.0, trainable=False)

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)
            self._zeros_slot(v, "d", self._name)
            self._zeros_slot(v, "f_hat", self._name)

    def _smaller_than_previous():
        return tf.constant(self._k + 1, dtype = tf.float32), tf.constant(self._K + 1, dtype = tf.float32)

    def _larger_than_previous():
        return tf.constant(1. / (self._K + 1) ), tf.constant(1. / (self._k + 1))

    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        beta3_t = math_ops.cast(self._beta3_t, var.dtype.base_dtype)
        eps = 1e-8

        t = self.t.assign_add(1.0)
        m = self.get_slot(var, "m")
        m_t = m.assign(beta1_t * m + (1. - beta1_t) * grad)
        m_hat = m_t / (1. - tf.pow(beta1_t, t))
        v = self.get_slot(var, "v")
        v_t = v.assign(beta2_t * v + (1. - beta2_t) * grad * grad)
        v_hat = v_t / (1. - tf.pow(beta2_t, t))
        f_hat = self.get_slot(var, "f_hat")

        loss = tf.get_collection('current_loss', scope="optimizer_variables")
        loss = tf.Print(loss, [loss], "LOSS IN OPTIMIZER")

        #TODO Figure out how to get 'loss' into apply_dense, either from outside or inside the optimizer

        # Calculate deltas
        delta1_t, delta2_t = tf.cond(tf.greater_equal(loss, f_hat), _smaller_than_previous, _larger_than_previous)

        # Clip parameter
        c_t = tf.minimum(tf.maximum(delta1_t, tf.div(loss, f_hat)), delta2_t)

        # Smoothly tracked objection function
        # f_hat * (c_t - 1) = f_hat_1 - f_hat_2 where f_hat_1 = f_hat_2 * c_t
        r_t = tf.div(tf.abs(f_hat * (c_t - 1)), tf.minimum(c_t * f_hat, f_hat))

        d = self.get_slot(var, "d")
        d_t = beta3_t + (1. - beta3_t) * r_t

        g_t = m_t / (d_t * tf.sqrt(v_t + eps))

        var_update = state_ops.assign_sub(var, lr_t * g_t)
        return control_flow_ops.group(*[var_update, m_t, v_t])

    #  def _apply_sparse(self, grad, var):
        #  lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        #  beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        #  beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        #  beta3_t = math_ops.cast(self._beta3_t, var.dtype.base_dtype)
        #  eps = 1e-8

        #  #  grad_dense = tf.sparse_to_dense(grad, grad.dense_shape, 0.0)

        #  t = self.t.assign_add(1.0)
        #  m = self.get_slot(var, "m")
        #  m_t = m.assign(beta1_t * m + (1. - beta1_t) * grad)
        #  m_hat = m_t / (1. - tf.pow(beta1_t, t))
        #  v = self.get_slot(var, "v")
        #  v_t = v.assign(beta2_t * v + (1. - beta2_t) * grad * grad)
        #  v_hat = v_t / (1. - tf.pow(beta2_t, t))
        #  g_t = m_t / tf.sqrt(v_t + eps)

        #  #TODO Figure out how to get 'loss' into apply_sparse, either from outside or inside the optimizer

        #  var_update = state_ops.assign_sub(var, lr_t * g_t)
        #  return control_flow_ops.group(*[var_update, m_t, v_t])

    def _apply_sparse(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        beta3_t = math_ops.cast(self._beta3_t, var.dtype.base_dtype)
        eps = 1e-8

        t = self.t.assign_add(1.0)
        m = self.get_slot(var, "m")
        m_t = m.assign(beta1_t * m + (1. - beta1_t) * grad)
        m_hat = m_t / (1. - tf.pow(beta1_t, t))
        v = self.get_slot(var, "v")
        v_t = v.assign(beta2_t * v + (1. - beta2_t) * grad * grad)
        v_hat = v_t / (1. - tf.pow(beta2_t, t))

        f_hat = self.get_slot(var, "f_hat")

        loss = tf.get_collection('current_loss', scope="optimizer_variables")
        loss = tf.Print(loss, [loss], "LOSS IN OPTIMIZER")

        #TODO Figure out how to get 'loss' into apply_dense, either from outside or inside the optimizer

        # Calculate deltas
        delta1_t, delta2_t = tf.cond(tf.greater_equal(loss, f_hat), _smaller_than_previous, _larger_than_previous)

        # Clip parameter
        c_t = tf.minimum(tf.maximum(delta1_t, tf.div(loss, f_hat)), delta2_t)

        # Smoothly tracked objection function
        # f_hat * (c_t - 1) = f_hat_1 - f_hat_2 where f_hat_1 = f_hat_2 * c_t
        r_t = tf.div(tf.abs(f_hat * (c_t - 1)), tf.minimum(c_t * f_hat, f_hat))

        d = self.get_slot(var, "d")
        d_t = beta3_t + (1. - beta3_t) * r_t

        g_t = m_t / (d_t * tf.sqrt(v_t + eps))

        var_update = state_ops.assign_sub(var, lr_t * g_t)
        return control_flow_ops.group(*[var_update, m_t, v_t])
