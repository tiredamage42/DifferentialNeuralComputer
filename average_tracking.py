import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

AVERAGE_TRACK_OPS = "__avg_ops__" 

'''
return the average of the original tensor scalar value
adds the increment op to AVERAGE_TRACK_OPS collection, run these ops
to add the current value to the running average
every time the average tensor is evaluated it resets the count
'''
def average_tracker(orig_val):
    total = tf.Variable(0.0, trainable=False)
    batches = tf.Variable(0.0, trainable=False)
    avg_dummy = tf.Variable(0.0, trainable=False)
    
    inc_op = tf.group(tf.assign(total, total+orig_val), tf.assign(batches, batches + 1))
    tf.add_to_collection(AVERAGE_TRACK_OPS, inc_op)

    with tf.control_dependencies([inc_op]):
        assign_dmmy = tf.assign(avg_dummy, total / batches)
        with tf.control_dependencies([inc_op, assign_dmmy]):
            rs = tf.group(tf.assign(total, 0.0), tf.assign(batches, 0.0))
        
            with tf.control_dependencies([inc_op, assign_dmmy, rs]):
                get_dmmy = tf.identity(avg_dummy)
                average = tf.identity(get_dmmy) #total / batches
            
    return average
   