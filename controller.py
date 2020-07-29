import tensorflow.compat.v1 as tf
import dense_layer

tf.disable_v2_behavior()

'''
a non-recurrent 'cell' to use as the controller

using this makes sure that any recurrent functionality
is learned using the DNC's read / write heads.
'''

class StatelessCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, layer_name, features):
        self.linear = dense_layer.DenseLayer(layer_name, features, activation=tf.nn.leaky_relu)
    
    @property
    def state_size(self):
        return 1
    
    @property
    def output_size(self):
        return self.linear.features
    
    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            out = self.linear(inputs)
        return out, state

    def zero_state(self, batch_size, dtype):
        return tf.zeros([batch_size, 1])

    def update_state(self, final_state):
        return tf.no_op()
    
    def reset_state(self):
        return tf.no_op()