import tensorflow as tf
from custom_tf.utils import tf_heaviside


class DiscontinuityDense(tf.keras.layers.Dense):
    """
    New subclass of tf.keras.layers.Dense s.t. it has a trainable discontinuity in 0.
    The map of this layer is:

    L(x) = f(W'x + b) + eps * H(W'x + b),

    where:
    f: activation function;
    W: layer's weight matrix
    b: layer's bias vector
    eps: layer's vector of trainable discontinuities
    """
    def __init__(self,
                 units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 discontinuity_initializer='zeros',
                 **options):
        """
        Initialization method
        :param units: same of class Dense
        :param activation: same of class Dense
        :param use_bias: same of class Dense
        :param kernel_initializer: same of class Dense
        :param bias_initializer: same of class Dense
        :param kernel_regularizer: same of class Dense
        :param bias_regularizer: same of class Dense
        :param activity_regularizer: same of class Dense
        :param kernel_constraint: same of class Dense
        :param bias_constraint: same of class Dense
        :param discontinuity_initializer: NEW PARAMETER, keras initializer for trainable discontinuities (def. Zeros())
        :param options: same of class Dense
        """
        super(DiscontinuityDense, self).__init__(units, activation, use_bias,
                                                 kernel_initializer, bias_initializer,
                                                 kernel_regularizer, bias_regularizer, activity_regularizer,
                                                 kernel_constraint, bias_constraint,
                                                 **options)

        if type(discontinuity_initializer) is str:
            discontinuity_initializer = getattr(tf.keras.initializers, discontinuity_initializer)

        if type(discontinuity_initializer) == type:
            discontinuity_initializer = discontinuity_initializer()

        self.discontinuity_initializer = discontinuity_initializer

    def get_config(self):

        config = super(DiscontinuityDense, self).get_config()
        config['discontinuity_initializer'] = self.discontinuity_initializer

        return config

    def build(self, input_shape):

        # Kernel of classical Dense layer
        # self.kernel = self.add_variable("kernel",
        #                                 shape=[int(input_shape[-1]),
        #                                        self.num_outputs])

        # Kernel creation
        self.kernel = self.add_weight('kernel', shape=[int(input_shape[-1]), self.units],
                                      initializer=self.kernel_initializer)
        self.bias = self.add_weight('bias', shape=[self.units],
                                    initializer=self.bias_initializer)
        self.discontinuity = self.add_weight('discontinuity', shape=[self.units],
                                             initializer=self.discontinuity_initializer)

    def call(self, input):
        matmul_tensor = tf.matmul(input, self.kernel)

        matmul_bias_tensor = tf.add(matmul_tensor, self.bias)

        add_discontinuty_tensor = tf.multiply(tf_heaviside(matmul_bias_tensor, dtype=self.dtype), self.discontinuity)

        out_tensor = tf.add(self.activation(matmul_bias_tensor), add_discontinuty_tensor)

        return out_tensor


class DenseResidualLayer(tf.keras.layers.Dense):
    def __init__(self,
                 units,
                 activation=None,
                 activation_after_sum=True,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs
                 ):

        super().__init__(units=units,
                         activation=activation,
                         use_bias=use_bias,
                         kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                         kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                         activity_regularizer=activity_regularizer,
                         kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
                         **kwargs
                         )

        self._activation_after_sum = activation_after_sum

    def get_config(self):
        config = super().get_config()

        config['activation_after_sum'] = self._activation_after_sum

        return config

    def call(self, inputs):

        x = tf.matmul(inputs, self.kernel)
        if self.bias is not None:
            x = x + self.bias
        if self.activation is not None:
            if self._activation_after_sum:
                x = self.activation(x + inputs)
            else:
                x = self.activation(x) + inputs
        else:
            x = x + inputs

        return x







