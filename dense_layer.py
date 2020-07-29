import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

'''
fully connected dense layer
'''
class DenseLayer():
    def __init__(self, layer_name, features, activation=tf.nn.leaky_relu, use_bias=True):
        self.is_built = False
        self.layer_name = layer_name
        self.use_bias = use_bias
        self.features = features
        self.activation = activation

    def __call__(self, in_tensor):
        with tf.variable_scope(self.layer_name) as scope:

            in_shape_g = tf.shape(in_tensor)
            in_shape_l = in_tensor.get_shape().as_list()
            if not self.is_built:
                self.weights = tf.get_variable('weights', shape=[in_shape_l[-1], self.features], dtype=tf.float32)
                self.biases = None
                if self.use_bias:
                    #getting better results with default glorot initializer
                    self.biases = tf.get_variable('biases', shape=[self.features], dtype=tf.float32)
                
                self.is_built = True

            # if we get a tensor of shape [ batch, sequence, features ]
            # reshape so it's shape [ batch * sequence, features ]
            needs_reshape = len(in_shape_l) != 2
            if needs_reshape:
                in_tensor = tf.reshape(in_tensor, [ -1, in_shape_l[-1] ], name="flatten")
                
            # matrix multiplication
            out_t = tf.matmul(in_tensor, self.weights, name="mx")

            # maybe add bias
            if self.biases is not None:
                out_t = tf.add(out_t, self.biases, name="add_bias")
            
            # maybe activate
            if self.activation is not None:
                out_t = self.activation(out_t, name="activated")
            
            # reshape back to [ batch, sequence, self.features ]
            # after our matrix multiplication
            if needs_reshape:
                out_t = tf.reshape(out_t, tf.concat([in_shape_g[:-1], tf.constant([self.features])], -1))
            
            return out_t

        