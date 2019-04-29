import tensorflow as tf
from keras.layers import Layer
from keras import regularizers


@tf.RegisterGradient("SignGrad")
def sign_grad(op, grad):
    input = op.inputs[0]
    cond  = (input >= -1) & (input <= 1)
    zeros = tf.zeros_like(grad)
    return tf.where(cond, grad, zeros)


class BinaryConv2D(Layer):
    def __init__(self, out_channels, kernel_size, binarize_input, dropout=0, paddings=1, **kwargs):
        self.out_channels   = out_channels
        self.kernel_size    = kernel_size
        self.paddings       = tf.constant([[0, 0], [paddings, paddings], [paddings, paddings], [0, 0]])
        self.binarize_input = binarize_input
        self.dropout        = dropout

        super(BinaryConv2D, self).__init__(**kwargs)


    def build(self, input_shape):
        self.kernel   = self.add_weight(name='kernel', shape=(self.kernel_size[0], self.kernel_size[1], input_shape[3], self.out_channels),
                                      initializer='glorot_uniform', regularizer=regularizers.l2(1e-4), trainable=True)
        self.bias     = self.add_weight(name='bias', shape=(self.out_channels,), initializer='zeros', trainable=True)

        super(BinaryConv2D, self).build(input_shape)

    @staticmethod
    def binarize(input):
        with tf.get_default_graph().gradient_override_map({"Sign":'SignGrad'}):
            sgn  = tf.sign(input)
            cond = tf.equal(sgn, tf.zeros_like(sgn))
            sgn  = tf.where(cond, tf.fill(tf.shape(sgn), -1.0), sgn)

        if input.shape.as_list()[0] is None:
            avg = tf.reduce_mean(tf.abs(input), axis=3, keepdims=True)
        else:
            avg = tf.reduce_mean(tf.abs(input), axis=[0, 1, 2])

        return sgn, avg

    def call(self, input):
        padded_input      = tf.pad(input, self.paddings, "CONSTANT")
        bin_kernel, alpha = BinaryConv2D.binarize(self.kernel)

        if self.binarize_input:
            bin_input, beta = BinaryConv2D.binarize(padded_input)
            K               = tf.nn.avg_pool(beta, (1, self.kernel_size[0],
                              self.kernel_size[1], 1), (1, 1, 1, 1), "VALID")
            Ka              = tf.multiply(K, alpha)

            if self.dropout > 0:
                bin_input = tf.nn.dropout(bin_input, self.dropout)

            bin_conv = tf.nn.convolution(bin_input, bin_kernel, "VALID")
            return tf.multiply(bin_conv, Ka) + self.bias

        bin_conv = tf.nn.convolution(padded_input, bin_kernel, "VALID")
        return tf.multiply(bin_conv, alpha) + self.bias


    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], self.out_channels)