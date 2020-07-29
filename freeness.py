import tensorflow.compat.v1 as tf
import dnc_utils
tf.disable_v2_behavior()

_EPSILON = 1e-6

class Freeness:
    """Memory usage that is increased by writing and decreased by reading.
    This module is a pseudo-RNNCore whose state is a tensor with values in
    the range [0, 1] indicating the usage of each of `memory_size` memory slots.
    The usage is:
    *   Increased by writing, where usage is increased towards 1 at the write
        addresses.
    *   Decreased by reading, where usage is decreased after reading from a
        location when free_gate is close to 1.
    The function `write_allocation_weights` can be invoked to get free locations
    to write to for a number of write heads.
    """

    def __init__(self, memory_size, name='freeness'):
        self._memory_size = memory_size
        
    @property
    def state_size(self):
        return self._memory_size

    def zero_state(self, batch_size, dtype):
        return tf.zeros([batch_size, self._memory_size])
        
    def __call__(self, write_weights, free_gate, read_weights, prev_usage):
        """Calculates the new memory usage u_t.
        Memory that was written to in the previous time step will have its usage
        increased; memory that was read from and the controller says can be "freed"
        will have its usage decreased.
        Args:
            write_weights: tensor of shape `[batch_size, num_writes,
                memory_size]` giving write weights at previous time step.
            free_gate: tensor of shape `[batch_size, num_reads]` which indicates
                which read heads read memory that can now be freed.
            read_weights: tensor of shape `[batch_size, num_reads,
                memory_size]` giving read weights at previous time step.
            prev_usage: tensor of shape `[batch_size, memory_size]` giving
                usage u_{t - 1} at the previous time step, with entries in range
                [0, 1].
        Returns:
            tensor of shape `[batch_size, memory_size]` representing updated memory
            usage.
        """
        # Calculation of usage is not differentiable with respect to write weights.
        write_weights = tf.stop_gradient(write_weights)
        
        usage = self._usage_after_write(prev_usage, write_weights)
        usage = self._usage_after_read(usage, free_gate, read_weights)
        return usage

    def write_allocation_weights(self, usage, write_gates, num_writes):
        """Calculates freeness-based locations for writing to.
        This finds unused memory by ranking the memory locations by usage, for each
        write head. (For more than one write head, we use a "simulated new usage"
        which takes into account the fact that the previous write head will increase
        the usage in that area of the memory.)
        Args:
            usage: A tensor of shape `[batch_size, memory_size]` representing current memory usage.
            write_gates: A tensor of shape `[batch_size, num_writes]` 
                with values in the range [0, 1] indicating how much each write head does writing
                based on the address returned here (and hence how much usage increases).
            num_writes: The number of write heads to calculate write weights for.
        Returns:
            tensor of shape `[batch_size, num_writes, memory_size]` 
                containing the freeness-based write locations. 
                Note that this isn't scaled by `write_gate`; this scaling must be applied externally.
        """
        with tf.name_scope('write_allocation_weights'):
            # expand gatings over memory locations
            write_gates = tf.expand_dims(write_gates, -1)
            print (write_gates)

            allocation_weights = []
            for i in range(num_writes):
                allocation_weights.append(self._allocation(usage))
                # update usage to take into account writing to this new allocation
                usage += ((1 - usage) * write_gates[:, i, :] * allocation_weights[i])

            # Pack the allocation weights for the write heads into one tensor.
            return tf.stack(allocation_weights, axis=1)

    def _usage_after_write(self, prev_usage, write_weights):
        """Calcualtes the new usage after writing to memory.
        Args:
            prev_usage: tensor of shape `[batch_size, memory_size]`.
            write_weights: tensor of shape `[batch_size, num_writes, memory_size]`.
        Returns:
            New usage, a tensor of shape `[batch_size, memory_size]`.
        """
        with tf.name_scope('usage_after_write'):
            # Calculate the aggregated effect of all write heads
            write_weights = 1 - dnc_utils.reduce_prod(1 - write_weights, 1)
            return prev_usage + (1 - prev_usage) * write_weights

    def _usage_after_read(self, prev_usage, free_gate, read_weights):
        """Calcualtes the new usage after reading and freeing from memory.
        Args:
            prev_usage: tensor of shape `[batch_size, memory_size]`.
            free_gate: tensor of shape `[batch_size, num_reads]` 
                with entries in the range [0, 1] indicating the amount that locations read from can be freed.
            read_weights: tensor of shape `[batch_size, num_reads, memory_size]`.
        Returns:
            New usage, a tensor of shape `[batch_size, memory_size]`.
        """
        with tf.name_scope('usage_after_read'):
            free_gate = tf.expand_dims(free_gate, -1)
            free_read_weights = free_gate * read_weights
            phi = dnc_utils.reduce_prod(1 - free_read_weights, 1, name='phi')
            return prev_usage * phi

    def _allocation(self, usage):
        """Computes allocation by sorting `usage`.
        This corresponds to the value a = a_t[\phi_t[j]] in the paper.
        Args:
            usage: tensor of shape `[batch_size, memory_size]` 
                indicating current memory usage. 
                This is equal to u_t in the paper when we only have one write head, 
                but for multiple write heads, one should update the usage while iterating through the write heads 
                to take into account the allocation returned by this function.
        Returns:
            Tensor of shape `[batch_size, memory_size]` corresponding to allocation.
        """
        with tf.name_scope('allocation'):
            # Ensure values are not too small prior to cumprod.
            usage = _EPSILON + (1 - _EPSILON) * usage
            nonusage = 1 - usage
            
            sorted_nonusage, indices = tf.nn.top_k(nonusage, k=self._memory_size, name='sort')
            sorted_usage = 1 - sorted_nonusage
            prod_sorted_usage = tf.cumprod(sorted_usage, axis=1, exclusive=True)
            sorted_allocation = sorted_nonusage * prod_sorted_usage
            inverse_indices = dnc_utils.batch_invert_permutation(indices)

            # This final line "unsorts" sorted_allocation, so that the indexing
            # corresponds to the original indexing of `usage`.
            return dnc_utils.batch_gather(sorted_allocation, inverse_indices)
            
    
    