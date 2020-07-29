import tensorflow.compat.v1 as tf
import collections
import numpy as np

import dnc_utils
from dense_layer import DenseLayer
from freeness import Freeness
from temporal_linkage import TemporalLinkage

tf.disable_v2_behavior()

AccessState = collections.namedtuple('AccessState', 
    (
        'memory', 
        'read_weights', 
        'write_weights', 
        'linkage', 
        'usage',
    )
)

class MemoryAccess:
    """
    Access module of the Differentiable Neural Computer.    
    This memory module supports multiple read and write heads. It makes use of:
    *  `addressing.TemporalLinkage` to track the temporal ordering of writes in
        memory for each write head.
    *  `addressing.FreenessAllocator` for keeping track of memory usage, where
        usage increase when a memory location is written to, and decreases when
        memory is read from that the controller says can be freed.
    Write-address selection is done by an interpolation between content-based lookup and using unused memory.
    Read-address selection is done by an interpolation of content-based lookup 
    and following the link graph in the forward or backwards read direction.
    """

    def __init__(self, memory_size=128, word_size=20, num_reads=1, num_writes=1, name='memory_access'):
        """
        constructs a memory matrix with read heads and write heads as described in the DNC paper
        http://www.nature.com/nature/journal/vaop/ncurrent/full/nature20101.html
        
        Args:
            memory_size: The number of memory slots (N in the DNC paper).
            word_size: The width of each memory slot (W in the DNC paper)
            num_reads: The number of read heads (R in the DNC paper).
            num_writes: The number of write heads (fixed at 1 in the paper).
            name: The name of the module.
        """
        self._memory_size = memory_size
        
        self._word_size = word_size
        self._num_reads = num_reads
        self._num_writes = num_writes

        self._linkage = TemporalLinkage(memory_size, num_writes)
        self._freeness = Freeness(memory_size)
        
        num_read_modes = 1 + 2 * num_writes
        
        self.interface_size = ((num_writes * word_size)*3) + (num_reads * num_read_modes) + (num_reads * word_size) + (num_writes * 3) + (num_reads * 2)
            
        # flatten channel should be 1....
        self.interface_linear = DenseLayer('write_vectors', self.interface_size, activation=None, use_bias=True)
        
    @property
    def state_size(self):
        return AccessState(
            memory=self._memory_size * self._word_size,
            read_weights=self._num_reads * self._memory_size,
            write_weights=self._num_writes * self._memory_size,
            linkage=self._linkage.state_size,
            usage=self._freeness.state_size,
        )

    def zero_state(self, batch_size, dtype):
        return AccessState(
            memory=tf.zeros([batch_size, self._memory_size, self._word_size]),
            read_weights=tf.zeros([batch_size, self._num_reads, self._memory_size]),
            write_weights=tf.zeros([batch_size, self._num_writes, self._memory_size]),
            linkage=self._linkage.zero_state(batch_size, dtype),
            usage=self._freeness.zero_state(batch_size, dtype),
        )
                

    @property
    def output_size(self):
        """Returns the output shape."""
        return self._num_reads * self._word_size#tf.TensorShape([self._num_reads * self._word_size])


    def __call__(self, inputs, prev_state):
        inputs = self._read_inputs(inputs)

        # Update usage using inputs['free_gate'] and previous read & write weights.
        usage = self._freeness(write_weights=prev_state.write_weights, free_gate=inputs['free_gate'], read_weights=prev_state.read_weights, prev_usage=prev_state.usage)
                
        # Write to memory.
        write_weights = self._write_weights(inputs, prev_state.memory, usage)
        memory = dnc_utils._erase_and_write(prev_state.memory, address=write_weights, reset_weights=inputs['erase_vectors'], values=inputs['write_vectors'])
        linkage_state = self._linkage(write_weights, prev_state.linkage)

        # Read from memory.
        read_weights = self._read_weights(inputs, memory=memory, prev_read_weights=prev_state.read_weights, link=linkage_state.link)
        read_words = tf.matmul(read_weights, memory)
        read_words = tf.reshape(read_words, [-1, self._num_reads * self._word_size])
        return (read_words, 
            AccessState(
                memory=memory, 
                read_weights=read_weights, 
                write_weights=write_weights, 
                linkage=linkage_state, 
                usage=usage
            ))
            
    def _read_inputs(self, inputs):
        # print("input to memory access read input")
        # print(inputs.get_shape())

        """Applies transformations to `inputs` to get control for this module."""
        interface_vectored = self.interface_linear(inputs)


        q = 0
        '''
        WRITE VECTORS [..., WRITE HEADS, WORD SIZE]
        # v_t^i - The vectors to write to memory, for each write head `i`.
        '''
        write_vectors_size = self._num_writes * self._word_size
        write_vectors = interface_vectored[:, q:q+write_vectors_size]
        write_vectors = tf.reshape(write_vectors, [-1, self._num_writes, self._word_size])
        q += write_vectors_size
        
        '''
        ERASE VECTORS [..., WRITE HEADS, WORD SIZE]
        # e_t^i - Amount to erase the memory by before writing, for each write head.
        '''
        erase_vectors_size = write_vectors_size
        erase_vectors = tf.nn.sigmoid(interface_vectored[:, q:q+erase_vectors_size])
        erase_vectors = tf.reshape(erase_vectors, [-1, self._num_writes, self._word_size])
        q += erase_vectors_size
        
        '''
        FREE GATE [..., READ HEADS]
        # f_t^j - Amount that the memory at the locations read from at the previous
        # time step can be declared unused, for each read head `j`.
        '''
        free_gate = tf.nn.sigmoid(interface_vectored[:, q:q+self._num_reads])
        q += self._num_reads
        
        '''
        ALLOCATION GATE [..., WRITE HEADS]
        # g_t^{a, i} - Interpolation between writing to unallocated memory and
        # content-based lookup, for each write head `i`. Note: `a` is simply used to
        # identify this gate with allocation vs writing (as defined below).
        '''
        allocation_gate = tf.nn.sigmoid(interface_vectored[:, q:q+self._num_writes])
        q += self._num_writes
        
        '''
        WRITE GATE [..., WRITE HEADS]
        # g_t^{w, i} - Overall gating of write amount for each write head.
        '''
        write_gate = tf.nn.sigmoid(interface_vectored[:, q:q+self._num_writes])
        q += self._num_writes
        
        '''
        READ MODES [..., READ HEADS, READ MODES]
        # \pi_t^j - Mixing between "backwards" and "forwards" positions (for each write head), 
        # and content-based lookup, for each read head.
        '''
        num_read_modes = 1 + 2 * self._num_writes
        read_mode_size = self._num_reads * num_read_modes
        read_mode = interface_vectored[:, q:q+read_mode_size]
        read_mode = tf.reshape(read_mode, [-1, self._num_reads, num_read_modes])
        read_mode = dnc_utils.BatchApply(read_mode, tf.nn.softmax)
        q += read_mode_size
           
        # Parameters for the (read / write) "weights by content matching" modules.
        '''WRITE KEYES [..., WRITE HEADS, WORD SIZE]'''
        write_keys_size = self._num_writes * self._word_size
        write_keys = interface_vectored[:, q:q+write_keys_size]
        write_keys = tf.reshape(write_keys, [-1, self._num_writes, self._word_size])
        q += write_keys_size
                
        '''READ KEYS [..., READ HEADS, WORD SIZE]'''
        read_keys_size = self._num_reads * self._word_size
        read_keys = interface_vectored[:, q:q+read_keys_size]
        read_keys = tf.reshape(read_keys, [-1, self._num_reads, self._word_size])
        q += read_keys_size
                
        '''WRITE STRENGTHS [..., WRITE HEADS]'''
        write_strengths = interface_vectored[:, q:q+self._num_writes]
        q += self._num_writes

        '''READ STRENGTHS [..., READ HEADS]'''
        read_strengths = interface_vectored[:, q:q+self._num_reads]
        q += self._num_reads

        
        result = {
            'read_content_keys': read_keys,
            'read_content_strengths': read_strengths,
            'write_content_keys': write_keys,
            'write_content_strengths': write_strengths,
            'write_vectors': write_vectors,
            'erase_vectors': erase_vectors,
            'free_gate': free_gate,
            'allocation_gate': allocation_gate,
            'write_gate': write_gate,
            'read_mode': read_mode
        }
        return result

    def _write_weights(self, inputs, memory, usage):
        """Calculates the memory locations to write to.
        This uses a combination of content-based lookup and finding an unused location in memory, for each write head.
        Args:
            memory: A tensor of shape  `[batch_size, memory_size, word_size]`
                containing the current memory contents.
            usage: Current memory usage, which is a tensor of shape `[batch_size, memory_size]`, used for allocation-based addressing.
        Returns:
            tensor of shape `[batch_size, num_writes, memory_size]` indicating where to write to (if anywhere) for each write head.
        """

        write_keys = inputs['write_content_keys'] # [..., WRITE HEADS, WORD SIZE]
        write_strengths = inputs['write_content_strengths'] #[..., WRITE HEADS]
        write_gate = inputs['write_gate'] # [..., WRITE HEADS]
        allocation_gate = inputs['allocation_gate'] # [..., WRITE HEADS]
        
        with tf.name_scope('write_weights', values=[inputs, memory, usage]):
            # c_t^{w, i} - The content-based weights for each write head.

            write_content_weights = dnc_utils.cosine_weighting(memory, write_keys, write_strengths, strength_op=tf.nn.softplus, name='cosine_weights')
            #write_content_weights = self._write_content_weights_mod(memory, write_keys, write_strengths)

            # a_t^i - The allocation weights for each write head.
            write_allocation_weights = self._freeness.write_allocation_weights(usage=usage, write_gates=(allocation_gate * write_gate), num_writes=self._num_writes)

            # Expands gates over memory locations.
            allocation_gate = tf.expand_dims(allocation_gate, -1) # [..., WRITE HEADS, 1]
            write_gate = tf.expand_dims(write_gate, -1) # [..., WRITE HEADS, 1]

            # w_t^{w, i} - The write weightings for each write head.
            return write_gate * (allocation_gate * write_allocation_weights + (1 - allocation_gate) * write_content_weights)

    def _read_weights(self, inputs, memory, prev_read_weights, link):
        """Calculates read weights for each read head.
        The read weights are a combination of following the link graphs in the
        forward or backward directions from the previous read position, and doing
        content-based lookup. The interpolation between these different modes is
        done by `inputs['read_mode']`.
        Args:
            inputs: Controls for this access module. This contains the content-based
                keys to lookup, and the weightings for the different read modes.
            memory: A tensor of shape `[batch_size, memory_size, word_size]`
                containing the current memory contents to do content-based lookup.
            prev_read_weights: A tensor of shape `[batch_size, num_reads,
                memory_size]` containing the previous read locations.
            link: A tensor of shape `[batch_size, num_writes, memory_size,
                memory_size]` containing the temporal write transition graphs.
        Returns:
            A tensor of shape `[batch_size, num_reads, memory_size]` containing the
            read weights for each read head.
        """
        with tf.name_scope('read_weights', values=[inputs, memory, prev_read_weights, link]):
            # c_t^{r, i} - The content weightings for each read head.
            content_weights = dnc_utils.cosine_weighting(memory, inputs['read_content_keys'], inputs['read_content_strengths'], strength_op=tf.nn.softplus, name='cosine_weights')
            
            # Calculates f_t^i and b_t^i.
            forward_weights = self._linkage.directional_read_weights(link, prev_read_weights, forward=True)
            backward_weights = self._linkage.directional_read_weights(link, prev_read_weights, forward=False)

            backward_mode = inputs['read_mode'][:, :, :self._num_writes]
            forward_mode = (inputs['read_mode'][:, :, self._num_writes:2 * self._num_writes])
            content_mode = inputs['read_mode'][:, :, 2 * self._num_writes]

            read_weights = (
                tf.expand_dims(content_mode, 2) * content_weights + tf.reduce_sum(tf.expand_dims(forward_mode, 3) * forward_weights, 2) +
                tf.reduce_sum(tf.expand_dims(backward_mode, 3) * backward_weights, 2)
            )

            return read_weights
















