import tensorflow.compat.v1 as tf
import collections
tf.disable_v2_behavior()
from memory_access import MemoryAccess

"""
constructs Differentiable Neural Computer architecture

use as an rnn cell
"""

DNCStateTuple = collections.namedtuple("DNCStateTuple", (
    "controller_state", 
    "access_state", 
    "read_vectors", 
))

class DNCCell(tf.nn.rnn_cell.RNNCell):

    def __init__(self, controller_cell, memory_size = 256, word_size = 64, num_reads = 4, num_writes = 1, clip_value=None):
        """
        controller_cell: 
            Tensorflow RNN Cell
        """     
        self.memory = MemoryAccess(memory_size, word_size, num_reads, num_writes)
        self.controller = controller_cell
        self._clip_value = clip_value or 0

    @property
    def state_size(self):
        return DNCStateTuple(
            controller_state=self.controller.state_size, 
            access_state=self.memory.state_size, 
            read_vectors=self.memory.output_size
        )
        
    @property
    def output_size(self):
        return self.controller.output_size + self.memory.output_size
        
    def zero_state(self, batch_size, dtype):
        return DNCStateTuple(
            controller_state=self.controller.zero_state(batch_size, dtype), 
            access_state=self.memory.zero_state(batch_size, dtype),
            read_vectors=tf.zeros([batch_size,] + [self.memory.output_size,], tf.float32)
        )
        
    def _clip_if_enabled(self, x):
        if self._clip_value <= 0:
            return x
        return tf.clip_by_value(x, -self._clip_value, self._clip_value)
        
    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__): 
            
            controller_state, access_state, read_vectors = state

            #concatenate last read vectors
            complete_input = tf.concat([inputs, read_vectors], -1)
            #processes input data through the controller network
            controller_output, controller_state = self.controller(complete_input, controller_state)
            
            controller_output = self._clip_if_enabled(controller_output)
            
            #processes input data through the memory module
            read_vectors, access_state = self.memory(controller_output, access_state)
            read_vectors = self._clip_if_enabled(read_vectors)

            #the final output by taking rececnt memory changes into account
            step_out = tf.concat([controller_output, read_vectors], -1)
            
            #return output and teh new DNC state
            return step_out, DNCStateTuple(controller_state=controller_state, access_state=access_state, read_vectors=read_vectors)
            
    