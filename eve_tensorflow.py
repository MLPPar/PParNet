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
        super(EveOptimizer, self).__init__(use_locking, name, loss=None)
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

    def _create_slots(self, var_list ):
        # Create slots for the first and second moments.
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)
            self._zeros_slot(v, "d", self._name)

    def _apply_dense(self, grad, var, loss):
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
        v_t = v.assign(beta2_t * v + (1. - beta2_t) * grad ** 2)
        v_hat = v_t / (1. - tf.pow(beta2_t, t))
        g_t = m_t / tf.sqrt(v_t + eps)

        #TODO Figure out how to get 'loss' into apply_dense, either from outside or inside the optimizer

        var_update = state_ops.assign_sub(var, lr_t * g_t)
        return control_flow_ops.group(*[var_update, m_t, v_t])

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")
