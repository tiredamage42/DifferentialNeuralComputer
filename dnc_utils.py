'''
functions needed for dnc implementation
'''

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def reduce_prod(x, axis, name=None):
    """
    Efficient reduce product over axis.
    Uses tf.cumprod and tf.gather_nd as a workaround to the poor performance of calculating tf.reduce_prod's gradient on CPU.
    """
    with tf.name_scope(name, 'util_reduce_prod', values=[x]):
        cp = tf.cumprod(x, axis, reverse=True)
        size = tf.shape(cp)[0]
        idx1 = tf.range(tf.cast(size, tf.float32), dtype=tf.float32)
        idx2 = tf.zeros([size], tf.float32)
        indices = tf.stack([idx1, idx2], 1)
        return tf.gather_nd(cp, tf.cast(indices, tf.int32))


def batch_invert_permutation(permutations):
    """Returns batched `tf.invert_permutation` for every row in `permutations`."""
    with tf.name_scope('batch_invert_permutation', values=[permutations]):
        perm = tf.cast(permutations, tf.float32)
        dim = int(perm.get_shape()[-1])
        size = tf.cast(tf.shape(perm)[0], tf.float32)
        delta = tf.cast(tf.shape(perm)[-1], tf.float32)
        rg = tf.range(0, size * delta, delta, dtype=tf.float32)
        rg = tf.expand_dims(rg, 1)
        rg = tf.tile(rg, [1, dim])
        perm = tf.add(perm, rg)
        flat = tf.reshape(perm, [-1])
        perm = tf.invert_permutation(tf.cast(flat, tf.int32))
        perm = tf.reshape(perm, [-1, dim])
        return tf.subtract(perm, tf.cast(rg, tf.int32))

def batch_gather(values, indices):
    """Returns batched `tf.gather` for every row in the input."""
    with tf.name_scope('batch_gather', values=[values, indices]):
        idx = tf.expand_dims(indices, -1)
        size = tf.shape(indices)[0]
        rg = tf.range(size, dtype=tf.int32)
        rg = tf.expand_dims(rg, -1)
        rg = tf.tile(rg, [1, int(indices.get_shape()[-1])])
        rg = tf.expand_dims(rg, -1)
        gidx = tf.concat([rg, idx], -1)
        return tf.gather_nd(values, gidx)





def _erase_and_write(memory, address, reset_weights, values):
    """Module to erase and write in the external memory.
    Erase operation:
        M_t'(i) = M_{t-1}(i) * (1 - w_t(i) * e_t)
    Add operation:
        M_t(i) = M_t'(i) + w_t(i) * a_t
    where e are the reset_weights, w the write weights and a the values.
    Args:
        memory: 3-D tensor of shape `[batch_size, memory_size, word_size]`.
        address: 3-D tensor `[batch_size, num_writes, memory_size]`.
        reset_weights: 3-D tensor `[batch_size, num_writes, word_size]`.
        values: 3-D tensor `[batch_size, num_writes, word_size]`.
    Returns:
        3-D tensor of shape `[batch_size, num_writes, word_size]`.
    """
    with tf.name_scope('erase_memory', values=[memory, address, reset_weights]):
        expand_address = tf.expand_dims(address, 3)
        reset_weights = tf.expand_dims(reset_weights, 2)
        weighted_resets = expand_address * reset_weights
        reset_gate = reduce_prod(1 - weighted_resets, 1)
        memory *= reset_gate
    with tf.name_scope('additive_write', values=[memory, address, values]):
        add_matrix = tf.matmul(address, values, adjoint_a=True)
        memory += add_matrix
    return memory

def split_leading_dim(tensor, inputs):
    """
    Args:
        tensor: Tensor to have its first dimension split.
        inputs: Original reference input to look the dimensions of.
    Returns:
        The input tensor, with its first dimension split.
    """
    input_shape_static = inputs.get_shape()
    input_shape_list = input_shape_static.as_list()
    tensor_shape_static = tensor.get_shape()
    tensor_shape_list = tensor_shape_static.as_list()
    if (input_shape_static.is_fully_defined() and tensor_shape_static.is_fully_defined()):
        new_shape = input_shape_list[:2] + tensor_shape_list[1:]
        return tf.reshape(tensor, new_shape)
    # Shape can't be inferred statically.
    dims_after_first = tf.shape(tensor)[1:]
    split_sizes = tf.shape(inputs)[:2]
    known_split_sizes = input_shape_list[:2]
    known_dims_after_first = tensor_shape_list[1:]
    output_size = tf.concat([split_sizes, dims_after_first], 0)
    result = tf.reshape(tensor, output_size)
    result.set_shape(known_split_sizes + known_dims_after_first)
    return result

def merge_leading_dims(array_or_tensor):
    """
    Merge the first dimensions of a tensor.
    Returns:
        Either the input value converted to a Tensor, with the requested dimensions
        merged, or the unmodified input value if the input has less than `2` dimensions.
    Raises:
        ValueError: If the rank of `array_or_tensor` is not well-defined.
    """
    tensor = array_or_tensor#tf.convert_to_tensor(array_or_tensor)
    tensor_shape_static = tensor.get_shape()
    # Check if the rank of the input tensor is well-defined.
    if tensor_shape_static.dims is None:
        raise ValueError("Can't merge leading dimensions of tensor of unknown rank.")
    tensor_shape_list = tensor_shape_static.as_list()
    if tensor_shape_static.is_fully_defined():
        new_shape = ([np.prod(tensor_shape_list[:2])] + tensor_shape_list[2:])
        return tf.reshape(tensor, new_shape)
    # Shape can't be inferred statically.
    tensor_shape = tf.shape(tensor)
    new_first_dim = tf.reduce_prod(tensor_shape[:2], keepdims=True)
    other_dims = tensor_shape[2:]
    new_size = tf.concat([new_first_dim, other_dims], 0)
    result = tf.reshape(tensor, new_size)
    if all(value is not None for value in tensor_shape_list[:2]):
        merged_leading_size = np.prod(tensor_shape_list[:2])
    else:
        merged_leading_size = None
    result.set_shape([merged_leading_size] + tensor_shape_list[2:])
    return result

def BatchApply(inputs, module_or_op):
    return split_leading_dim(module_or_op(merge_leading_dims(inputs)), inputs)

def weighted_softmax(activations, strengths, strengths_op):
    """Returns softmax over activations multiplied by positive strengths.
    Args:
        activations: A tensor of shape `[batch_size, num_heads, memory_size]`, of
            activations to be transformed. Softmax is taken over the last dimension.
        strengths: A tensor of shape `[batch_size, num_heads]` containing strengths to
            multiply by the activations prior to the softmax.
        strengths_op: An operation to transform strengths before softmax.
    Returns:
        A tensor of same shape as `activations` with weighted softmax applied.
    """
    transformed_strengths = tf.expand_dims(strengths_op(strengths), -1)
    sharp_activations = activations * transformed_strengths
    return BatchApply(sharp_activations, tf.nn.softmax)
    
# Ensure values are greater than epsilon to avoid numerical instability.
_EPSILON = 1e-6

def _vector_norms(m):
    squared_norms = tf.reduce_sum(m * m, axis=2, keepdims=True)
    return tf.sqrt(squared_norms + _EPSILON)

def cosine_weighting(memory, keys, strengths, strength_op=tf.nn.softplus, name='cosine_weights'):
    """Cosine-weighted attention.
    
    Calculates the cosine similarity between a query and each word in memory, then
    applies a weighted softmax to return a sharp distribution.
    
    Args:
        memory: A 3-D tensor of shape `[batch_size, memory_size, word_size]`.
        keys: A 3-D tensor of shape `[batch_size, num_heads, word_size]`.
        strengths: A 2-D tensor of shape `[batch_size, num_heads]`.
        strength_op: operation to apply to strengths (default is tf.nn.softplus).
        
    Returns:
        Weights tensor of shape `[batch_size, num_heads, memory_size]`.
    """
    # Calculates the inner product between the query vector and words in memory.
    dot = tf.matmul(keys, memory, adjoint_b=True)

    # Outer product to compute denominator (euclidean norm of query and memory).
    memory_norms = _vector_norms(memory)
    key_norms = _vector_norms(keys)
    norm = tf.matmul(key_norms, memory_norms, adjoint_b=True)
    # Calculates cosine similarity between the query vector and words in memory.
    similarity = dot / (norm + _EPSILON)
    return weighted_softmax(similarity, strengths, strength_op)
