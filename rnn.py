'''
RNN WRAPPER

loop_state_fn arguments: (time, previous loop state, previous cell state)
should return the new loop state (any sort of tuple)

'''
import tensorflow.compat.v1 as tf
from tensorflow.python.ops.rnn import _transpose_batch_time
tf.disable_v2_behavior()

def dynamic_rnn(input_data, cell, loop_state_fn=None, initial_loop_state=None):
    inputs_shape_g = tf.shape(input_data)
    input_shape_l = input_data.get_shape().as_list()
    
    print ("\n\ninput shape length > 2 ?")
    print (len(input_shape_l) > 2)
    print ("\n\n")
    
    pad_input = tf.zeros([inputs_shape_g[0],] + (input_shape_l[2:] if len(input_shape_l) > 2 else []))
    
    seq_lengths = inputs_shape_g[1]

    # raw_rnn uses TensorArray for the input and outputs, in which Tensor must be in [time, batch_size, input_depth] shape. 
    inputs_ta = tf.TensorArray(size=inputs_shape_g[1], dtype=tf.float32).unstack(_transpose_batch_time(input_data), 'TBD_Input')
    
    initial_state = cell.zero_state(inputs_shape_g[0], None)
    
    def loop_fn(time, previous_output, previous_state, previous_loop_state):
        # this operation produces boolean tensor of [batch_size] defining if corresponding sequence has ended
        # all False at the initial step (time == 0)
        finished = time >= seq_lengths
        if previous_state is None:    # time == 0
            return (finished, inputs_ta.read(time), initial_state, previous_output, initial_loop_state)
        else:
            step_input = tf.cond(tf.reduce_all(finished), lambda: pad_input, lambda: inputs_ta.read(time))
            if loop_state_fn is not None:
                previous_loop_state = loop_state_fn(time, previous_loop_state, previous_state)
            return (finished, step_input, previous_state, previous_output, previous_loop_state)

    outputs_ta, final_state, final_loop_state = tf.nn.raw_rnn(cell, loop_fn)

    output = _transpose_batch_time(outputs_ta.stack())

    return output, final_state, final_loop_state

