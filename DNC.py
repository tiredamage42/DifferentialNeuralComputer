'''
see if the DNC implementation 
can at least repeat back a random series in order
visualizes the read / write weighting and usage vectors during debugging
'''
import os
# suppress info logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import sys
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
from dnc_cell import DNCCell
from controller import StatelessCell
from repeat_copy_data import copy_sequence_toy_data_fn     
from rnn import dynamic_rnn
from dense_layer import DenseLayer
import visualization
import average_tracking

tf.disable_v2_behavior()

def log(msg):
    sys.stdout.write(msg)
    sys.stdout.flush()

vocab_size = 8
norm_clip = 2.0
learn_rate = 1e-3
momentum = 0.9

'''DATASET'''
# copy repeat task (binary sequences)
inputs_fn = copy_sequence_toy_data_fn(
    vocab_size=vocab_size, 
    seq_length_range_train=[4,4], series_length_range_train=[2,4], 
    seq_length_range_val=[4,4], series_length_range_val=[4,4], 
    num_validation_samples=100, error_pad_steps = 2
)

'''MODEL'''
def model_fn (data_in):
    seq_in = data_in['sequence']
    targets = data_in['target']

    seq_in_one_hot = tf.one_hot(seq_in, depth=vocab_size)

    #build the dnc cell
    cell = DNCCell(StatelessCell("linear", features=8), memory_size=8, word_size=8, num_reads=1, num_writes=1)

    output, _, final_loop_state = dynamic_rnn(
        seq_in_one_hot, cell, visualization.loop_state_fn, visualization.initial_loop_state()
    )
    
    logits = DenseLayer("logits", vocab_size, activation=None)(output)
    preds = tf.argmax(tf.nn.softmax(logits), -1)

    #return dictionary of tensors to keep track of
    args_dict = {}

    with tf.variable_scope('Loss'):
        cross_entorpy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)
        
        #no loss at end of sequences in batch (Padding)
        # loss_weights_t = tf.clip_by_value(tf.cast(targets, tf.float32), 0.0, 1.0)
        # cross_entorpy = cross_entorpy * loss_weights_t
        loss = tf.reduce_mean(cross_entorpy)
        args_dict['loss'] = loss

    
    with tf.variable_scope('train'):
        o = tf.train.RMSPropOptimizer(learn_rate, momentum)
        
        gvs = o.compute_gradients(loss, var_list=tf.trainable_variables())
        '''clip gradients'''
        gradients, variables = zip(*gvs)
        gradients, _ = tf.clip_by_global_norm(gradients, norm_clip)
        capped_gvs = zip(gradients, variables)
        
        args_dict['optimizer'] = o.apply_gradients(capped_gvs)

    #track loss average every 100 steps
    args_dict['avg_loss'] = average_tracking.average_tracker(loss)

    #track loop state in tensorboard
    args_dict['mem_view'] = visualization.assemble_mem_view(final_loop_state, [seq_in, preds, targets], vocab_size)
    
    return args_dict

'''TRAINING'''

def plot_mem_view (image, name):
    if not os.path.exists('images'):
        os.makedirs('images')
    
    plt.imshow(image, interpolation='nearest', cmap='binary')
    plt.ylabel("Mem(R:Read G:Write)---Usage---Targets---Outputs---Inputs")
    # Remove ticks
    plt.axes().set_xticks([])
    plt.axes().set_yticks([])
    
    plt.savefig(os.path.join('images', name) + '.png')
    # plt.close()
    plt.clf()

def plot_losses(iterations, train_losses, val_losses, name):
    if not os.path.exists('images'):
        os.makedirs('images')
    
    plt.plot(iterations, train_losses, label='Training')
    plt.plot(iterations, val_losses, label='Validation')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join('images', name) + '.png')
    plt.clf()
    
def _train_iteration_dummy(session, model_args, data_args, train_ops):
    
    run_return = session.run(
        {
            '_': tf.get_collection(average_tracking.AVERAGE_TRACK_OPS),
            'avg_loss': model_args['avg_loss']
        }, 
        feed_dict = {
            #use handle in feed dict when using that data set
            data_args['handle_ph']: train_ops['handle']
        }
    )

    model_args['avg_train_loss'] = run_return['avg_loss']
    
    return model_args

def _one_shot_loop(session, msg, data_args, model_args, data_args_g, iteration):
    print('\n')#msg + '\n')
    
    session.run(data_args['Iterator'].initializer)
    sample_count = data_args['SampleCount']
    batch_size = data_args['BatchSize']

    max_batches = int(sample_count / batch_size) + (1 if sample_count % batch_size > 0 else 0)
    batch = 0
    while True:
        try:
            at_end = (batch == (max_batches - 1))

            run_tensors = {
                '_': tf.get_collection(average_tracking.AVERAGE_TRACK_OPS)
            }

            if at_end:
                run_tensors['avg_validation_loss'] = model_args['avg_loss']
            
                if msg == 'Debugging':
                    run_tensors['mem_view'] = model_args['mem_view']

            run_return = session.run(run_tensors, feed_dict={ data_args_g['handle_ph']: data_args['handle'] })

            if at_end:
                if msg == 'Validating':
                    model_args['avg_validation_loss'] = run_return['avg_validation_loss']
                
                elif msg == 'Debugging':
                    plot_mem_view (run_return['mem_view'][0], 'mem_view_{}'.format(iteration))
                    
            
            log("\r" + msg + " Batch: {0}/{1}\t".format(batch+1, max_batches))
            batch += 1

        except (tf.errors.OutOfRangeError, StopIteration):
            print('\rDone ' + msg + '                                ')
            break

    return model_args


def run_training(batch_size, iterations, inputs_fn, model_fn):
    
    with tf.Graph().as_default():      
        
        print ('Initializing Inputs...')
        samples_in, data_args = inputs_fn(batch_size=batch_size)
        
        train_ops = data_args.get("Training", None)
        val_ops = data_args.get("Validation", None)
        debug_ops = data_args.get("Debug", None)

        print ('Initializing Model...')
        model_args = model_fn(samples_in)
        
        print ('Trainable Vars:')
        [ print ('\t{}'.format(v)) for v in tf.trainable_variables() ]
        
        print ("Building Session...")
        with tf.Session() as sess:

            print ("Initializing Variables...")
            sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
            
            print ("Getting Dataset Handles...")
            train_ops['handle'] = sess.run(train_ops['Iterator'].string_handle())
            val_ops['handle'] = sess.run(val_ops['Iterator'].string_handle())
            debug_ops['handle'] = sess.run(debug_ops['Iterator'].string_handle())
            
            print ('Running Dummy Train Iteration To Populate Train Loss...')
            model_args = _train_iteration_dummy(sess, model_args, data_args, train_ops)


            # keep track of the losses over time
            # so we can visualze them later
            iterations_list = []
            train_losses = []
            val_losses = []

            def validate_and_debug(iteration, model_args):
                model_args = _one_shot_loop(sess, "Validating", val_ops, model_args, data_args, iteration)

                #print validation results
                msg = "==========================================================="
                msg += "\n|LOSS: T: {0:.3} V: {1:.3}\t| ".format(model_args['avg_train_loss'], model_args['avg_validation_loss']) + "Iteration {0}".format(iteration)       
                msg += "\n==========================================================="
                log(msg)

                model_args = _one_shot_loop(sess, "Debugging", debug_ops, model_args, data_args, iteration)


                iterations_list.append(iteration)
                train_losses.append(model_args['avg_train_loss'])
                val_losses.append(model_args['avg_validation_loss'])
                return model_args


            #validate at 0 iteration
            model_args = validate_and_debug(0, model_args)
                    
            
            print ('Starting Training Session...\n')
            for i in range(iterations):
                
                do_debugs = (i % 100 == 0 or i == iterations - 1) and i != 0

                run_tensors = {
                    '_': tf.get_collection(average_tracking.AVERAGE_TRACK_OPS),
                    'optimizer': model_args['optimizer'],
                    'loss': model_args['loss']
                }
                        
                if do_debugs:
                    run_tensors['avg_loss'] = model_args['avg_loss']

                run_return = sess.run(run_tensors, feed_dict={
                    #use handle in feed dict when using that data set
                    data_args['handle_ph']: train_ops['handle']
                })

                log("\rTraining Iteration: {0}/{1} :: Loss: {2:.4} ===========".format(i, iterations, run_return['loss']))

                if do_debugs:
                    model_args['avg_train_loss'] = run_return['avg_loss']
                    model_args = validate_and_debug(i, model_args)
            
            plot_losses(iterations_list, train_losses, val_losses, 'losses')

            
run_training(
    batch_size=1, iterations=10000, 
    inputs_fn=inputs_fn, 
    model_fn=model_fn
)
