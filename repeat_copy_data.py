

'''
TOY DATA SET FOR RNN MODELS

for softmax losses sequences and labels are integer lists

inputs and targets are lined up with pads

I: x x x x | - - - - - - 
T: - - - - x x x x * - - (repeat pattern series amount)

'''
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def copy_sequence_toy_data_fn (
    vocab_size, 
    seq_length_range_train, series_length_range_train, 
    seq_length_range_val, series_length_range_val, 
    num_validation_samples=500, error_pad_steps=2
):
    EOS = 2
    #Here we are doing a toy copy task, so we allow model some room to make mistakes over 2 additional steps:            
    def data_fn (batch_size):
        shape_end = []

        def map_fn(seq_length_range, series_length_range):
            def fn(data_in):
                seq_length = tf.random_uniform(shape=[], minval=seq_length_range[0], maxval=seq_length_range[1]+1, dtype=tf.int32)
                dtype = tf.int32
                
                '''
                x x x x | - - - - - - 
                - - - - x x x x * - -
                seq * 2 + 1 + pad_error
                '''

                series_length = tf.random_uniform(shape=[], minval=series_length_range[0], maxval=series_length_range[1]+1, dtype=tf.int32)
                
                '''
                [[1],
                [1]]
                '''                    
                sos_slice = tf.ones([series_length, 1,] + shape_end, dtype=dtype)
                                    
                '''[6, 3, 4, 5]'''                                    
                sequence = tf.random_uniform(shape=[series_length, seq_length], minval=3, maxval=vocab_size, dtype=dtype)
                
                '''[2]'''                    
                eos_slice = tf.fill([series_length, 1], EOS)
                    
                '''[6, 3, 4, 5, 1]'''                    
                sequence_with_sos = tf.concat([sequence, sos_slice], 1)
                                        
                input_end_pad = tf.zeros([series_length, seq_length + error_pad_steps,] + shape_end, dtype=dtype)
                                
                '''IN: [6, 3, 4, 5, 1, 0, 0, 0, 0, 0, 0]'''                    
                input_data = tf.concat([sequence_with_sos, input_end_pad], 1)
                    
                '''IN: [6, 3, 4, 5, 1, 0, 0, 0, 0, 0, 0]'''                    
                '''OUT:[-------------------------, 0, 0]'''                    
                error_pad_slice = tf.zeros([series_length, error_pad_steps,] + shape_end, dtype=dtype)
                
                '''IN: [6, 3, 4, 5, 1, 0, 0, 0, 0, 0, 0]'''                    
                '''OUT:[0, 0, 0, 0, -------------------]'''                    
                target_start_pad = tf.zeros([series_length, seq_length,] + shape_end, dtype=dtype)

                '''IN: [6, 3, 4, 5, 1, 0, 0, 0, 0, 0, 0]'''                    
                '''OUT:[0, 0, 0, 0, 6, 3, 4, 5, 1, 0, 0]'''                    
                target_output = tf.concat([target_start_pad, sequence, eos_slice, error_pad_slice], 1)
                    
                #'batch' to series
                input_sequence = tf.reshape(input_data, [((seq_length * 2) + 1 + error_pad_steps) * series_length,] + shape_end)    
                target_sequence = tf.reshape(target_output, [((seq_length * 2) + 1 + error_pad_steps) * series_length,] + shape_end)   

                data_in['sequence'] = input_sequence
                data_in['target'] = target_sequence
                data_in['seq_length'] = tf.shape(data_in['sequence'])[0]
                return data_in
            return fn
            
        d = [0]
        template = { 'sequence':d, 'seq_length':d, 'target':d, 'target_seq_length':d }
        padded_shapes = { 'sequence':[None,]+shape_end, 'seq_length':[], 'target':[None,]+shape_end, 'target_seq_length':[] }
        padding_values = { 'sequence': 0, 'seq_length': 0, 'target': 0, 'target_seq_length':0 }

        ds = tf.data.Dataset.from_tensor_slices(template).map(map_fn(seq_length_range_train, series_length_range_train)).repeat()
        ds = ds.shuffle(buffer_size=1000+3 * batch_size)

        ds = ds.padded_batch(batch_size, padded_shapes=padded_shapes, padding_values=padding_values)
        
        ds_v = tf.data.Dataset.from_tensor_slices(template).map(map_fn(seq_length_range_val, series_length_range_val)).repeat(num_validation_samples)
        ds_v = ds_v.padded_batch(batch_size, padded_shapes=padded_shapes, padding_values=padding_values)

        ds_d = tf.data.Dataset.from_tensor_slices(template).map(map_fn(seq_length_range_val, series_length_range_val)).batch(1)
     
        return initialize_dataset_args(ds, ds_v, num_validation_samples, batch_size, ds_d, 1, 1)
   
    return data_fn


'''Prepare tf datasets args dictionary for use with default training and testing loops'''
def initialize_dataset_args(train_set, validation_set, validation_samples, batch_size, debug_set, debug_samples, debug_batch):
    data_args = {}
    data_args['handle_ph'] = tf.placeholder(tf.string, shape=[])
    
    # A feedable iterator is defined by a handle placeholder and its structure. 
    iterator = tf.data.Iterator.from_string_handle(data_args['handle_ph'], train_set.output_types, train_set.output_shapes)
    data_args['Training'] = {}
    data_args['Training']['Iterator'] = train_set.make_one_shot_iterator() #repeats infinite anyways
    data_args['Training']["BatchSize"] = batch_size
    
    data_args['Validation'] = {}
    data_args['Validation']['Iterator'] = validation_set.make_initializable_iterator()
    data_args['Validation']['SampleCount'] = validation_samples
    data_args['Validation']["BatchSize"] = batch_size

    data_args['Debug'] = {}
    data_args['Debug']['Iterator'] = debug_set.make_initializable_iterator()
    data_args['Debug']['SampleCount'] = debug_samples
    data_args['Debug']["BatchSize"] = debug_batch
            
    nxt = iterator.get_next()
    return (nxt, data_args)
            



