import tensorflow.compat.v1 as tf
from tensorflow.python.ops.rnn import _transpose_batch_time
import collections
tf.disable_v2_behavior()

'''initial loop state to pass into dynamic rnns to track dnc cells'''
def initial_loop_state():
    return tuple(tf.TensorArray(tf.float32, dynamic_size=True, size=0) for _ in range(3))

'''loop_state_fn to pass into dynamic rnns to track dnc cells'''
def loop_state_fn(time, previous_loop_state, previous_state):
    assert len(previous_loop_state) == 3
    t = time - 1
    return (
        previous_loop_state[0].write(t, previous_state.access_state.read_weights),
        previous_loop_state[1].write(t, previous_state.access_state.write_weights),
        previous_loop_state[2].write(t, previous_state.access_state.usage)
    )

'''
outputs an image tensor [batch, h, w, 3] 
from the final loop state and series that were inputs, outputs, or targets
'''
def assemble_mem_view(final_loop_state, series_list, vocab_size):
    filters = [ 
        [ [ [ [ 1.0, 0.0, 0.0 ] ] ] ], #R
        [ [ [ [ 0.0, 1.0, 0.0 ] ] ] ], #G
        [ [ [ [ 0.0, 0.0, 1.0 ] ] ] ]  #B
    ] 
    
    memory_view = tuple (_transpose_batch_time(view.stack()) for view in final_loop_state)
    read_weightings = memory_view[0]
    write_weightings = memory_view[1]
    
    #inputs outputs and targets
    series_list = [tf.one_hot(s, depth=vocab_size) for s in series_list if s is not None]
    series_imgs_ = []
    for i, s in enumerate(series_list):
        series_imgs_.append( tf.tile(tf.expand_dims(s, -1), [1,1,1,3]) * filters[i%3])
    series_imgs = tf.concat(series_imgs_, 2)    

    #memory views
    read_weightings = tf.expand_dims(memory_view[0][:, :, 0, :], -1)
    write_weightings = tf.expand_dims(memory_view[1][:, :, 0, :], -1)
    mem_imgs = tf.concat([read_weightings, write_weightings, tf.zeros_like(write_weightings)], -1)
    usage_vectors = tf.tile(tf.expand_dims(memory_view[2], -1), [1, 1, 1, 3])

    #return concatenated memory views and series views
    return tf.transpose(tf.concat([series_imgs, usage_vectors, mem_imgs], 2), [0, 2, 1, 3])
    