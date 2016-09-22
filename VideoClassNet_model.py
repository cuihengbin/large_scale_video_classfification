"""
Build the Network of VideoClassNet

# Formula for dimensions of the output of a conv layer
# W2=(W1−F+2P)/S+1. Where, F kernel size, P is padding
# S is stride.
"""


import tensorflow as tf

from VideoClassNet_video_processer import batch_inputs



FLAGS = tf.app.flags.FLAGS

NUM_CLASS = 102
FRAME_COUNTS = 1  # Number of frames in a video
BATCH_SIZE = 32

def _variable_on_cpu(name, shape, initializer):
	"""Helper to create a Variable stored on CPU memory.

	Args:
	  name: name of the variable
	  shape: list of ints
	  initializer: initializer for Variable

	Returns:
	  Variable Tensor
	"""
	dtype = tf.float32
	with tf.device('/cpu:0'):
		variable = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
	return variable

def _variable_with_weight_decay(name, shape, stddev=1e-4, wd=None):
	"""Helper to create an initialized Variable with weight decay

	Args:
		name: name of the variable
		shape: list of ints
		stddev: float, standard deviation of a truncated Gaussian for initial value
		wd: add L2loss weight decay multiplied by this float. If None, weight decay 
				is not added to this variable

	Returns:
		Variable: Tensor
	"""
	dtype = tf.float32
	variable = _variable_on_cpu(
							name, 
							shape,
							initializer=tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
	if wd is not None:
		weight_decay = tf.mul(tf.nn.l2_loss(variable), wd, name='weight_loss')
		tf.add_to_collection('losses', weight_decay)
	return variable


def _conv_layer(lastlayer, scope, name, d, f, s, wd=0.0):
	"""Convolution layer

		Args:
			scope: scope of the layer
			name: string
			d: int, number of kernel
			f: int, spatial size of kernel
			s: int, strides
			wd: float, weight decay rate of kernel

		Returns:
			Relu activated conv layer
	"""
	with tf.variable_scope(scope.name+'/'+name) as scope_var:
		# Kernel has shape
		# [Kernal_height, Kernal_width, in_channels, out_channels]
		# wd=0.0 or None will not perform weight decay
		c = lastlayer.get_shape()[-1].value
		kernel = _variable_with_weight_decay(scope_var.name+'/weights',
			shape=[f, f, c, d], wd=wd) 

		#  for the same horizontal and vertices strides, 
		# strides = [1, stride, stride, 1]
		strides = [1, s, s, 1]
		conv = tf.nn.conv2d(lastlayer, kernel, strides, padding='SAME')

		# Add bias 
		biases = _variable_on_cpu(scope_var.name + '/biases', 
			[d], tf.constant_initializer(0.0))
		conv_with_bias = tf.nn.bias_add(conv, biases)
		return tf.nn.relu(conv_with_bias, name=scope_var.name)


def _layer_before_fc(videos, scope):
	"""Netwotk before fully connected layer. i.e. a pooling layer

	Using shorthand notation, the full architec-
	ture is C(96, 11, 3)-N -P -C(256, 5, 1)-N -P -C(384, 3, 1)-
	C(384, 3, 1)-C(256, 3, 1)-P -F C(4096)-F C(4096), where
	C(d, f, s) indicates a convolutional layer with d filters of
	spatial size f × f , applied to the input with stride s. F C(n)
	is a fully connected layer with n nodes. All pooling layers P
	pool spatially in non-overlapping 2 × 2 regions and all nor-
	malization layers N are defined as described in Krizhevsky
	et al. [11] and use the same parameters: k = 2, n = 5, α =
	10 −4 , β = 0.5.
	
	Dimensions for each layer
	(total_frames, 89, 89, 3) Video
	(total_frames, 30, 30, 96) conv1
	(total_frames, 30, 30, 96) norm1
	(total_frames, 15, 15, 96) pool1
	(total_frames, 15, 15, 256) conv2
	(total_frames, 15, 15, 256) norm2
	(total_frames, 8, 8, 256) pool2
	(total_frames, 8, 8, 384) conv3
	(total_frames, 8, 8, 384) conv4
	(total_frames, 8, 8, 256) conv5
	(total_frames, 4, 4, 256) pool3

	Args:
		video: 4D tensor, with [total_frames, height, width, 3]
			where total_frames = FRAME_COUNTS * batch_size
		scope: scope

	return:
		last_layer_before_fc: pool3 layer 
			with (total_frames, 4, 4, 256) pool3
	"""
	print(videos.get_shape(), 'Raw')
	## Declare parameters consistent with the paper
	k, n, alpha, beta = 2, 5, 1e-4, 0.5 #for norm layers

	# Max pooling layers.
	k_pool, s_pool = [1, 2, 2, 1], [1, 2, 2, 1]

	# Conv1 with shape
	# [total_frames, 30, 30, 96] # (89 +1 ) // 3
	conv1 = _conv_layer(video, scope, 'conv1', 96, 11, 3)
	print(conv1.get_shape(), 'conv1')
	# Norm1 layer with shape
	# [total_frames, 30, 30, 96]
	norm1 = tf.nn.lrn(conv1, k, n, alpha, beta, name='norm1')
	print(norm1.get_shape(), 'norm1')

	# Max pooling1 layer, with shape
	# [total_frames, 15, 15, 96] # W2=(W1-F)/S +1
	pool1 = tf.nn.max_pool(norm1, 
		ksize=k_pool, strides=s_pool, padding='SAME', name='pool1')
	print(pool1.get_shape(), 'pool1')
	# Conv2 layer with shape
	# [total_frames, 15, 15, 256]

	conv2 = _conv_layer(pool1, scope, 'conv2', 256, 5, 1)
	print(conv2.get_shape(), 'conv2')
	# Norm2 layer with shape
	norm2 = tf.nn.lrn(conv2, k, n, alpha, beta, name='norm2')
	print(norm2.get_shape(), 'norm2')
	# Max_pooling2 layer
	# [total_frames, 3, 3, 256]
	pool2 = tf.nn.max_pool(norm2, 
		ksize=k_pool, strides=s_pool, padding='SAME', name='pool2')
	print(pool2.get_shape(), 'pool2')
	# Conv3 layer with shape
	# [total_frames, 3, 3, 384]
	conv3 = _conv_layer(pool2, scope, 'conv3', 384, 3, 1)
	print(conv3.get_shape(), 'conv3')
	# Conv4 layer with shape
	# [total_frames, 3, 3, 384]
	conv4 = _conv_layer(conv3, scope, 'conv4', 384, 3, 1)
	print(conv4.get_shape(), 'conv4')
	# Conv5 layer with shape
	# [total_frames, 3, 3, 256]
	conv5 = _conv_layer(conv4, scope, 'conv5', 256, 3, 1)
	print(conv5.get_shape(), 'conv5')
	pool3 = tf.nn.max_pool(conv5,
		ksize=k_pool, strides=s_pool, padding='SAME', name='pool3')
	print(pool3.get_shape(), 'pool3')
	return pool3




def inference(fovea_batch, context_batch, batch_size):
	"""Produce losses across all videos in a batch
		
		Args:
			fovea_batch: 4D tensor [total_frames, 89, 89, 3]
			context_batch: 4D tensor [total_frames, 89, 89, 3]
			batch_size: 1D tensor with length [total_frames]

		Returns:
			logits: 2D tensor with shape[total_frames, NUM_CLASS],
				 unnormalized likelyhood probability
	"""
	total_frames = FRAME_COUNTS * batch_size

	# (total_frames, 4, 4, 256) pool3
	with tf.variable_scope('fovea') as scope:
		fovea_pool3 = _layer_before_fc(fovea_video, scope)
	with tf.variable_scope('context') as scope:
		context_pool3 = _layer_before_fc(context_video, scope)

	# Concatnate two streams along the channel dimension
	# Flattern concated layer except the first dimension
	# so we can perform a single matrix multiply
	with tf.variable_scope('reshaped_concated_pool3') as scope:
		concated_pool3 = tf.concat(concat_dim=3, 
						values=[fovea_pool3, context_pool3])
		reshaped_concated = tf.reshape(concated_pool3, [total_frames, -1])
		dim = reshaped_concated.get_shape()[1].value

	# Fully connected layer 1
	with tf.variable_scope('fc1') as scope:
		weights = _variable_with_weight_decay('weights', [dim, 4096])
		biases = _variable_on_cpu('biases', [4096], 
				tf.constant_initializer(0.1))
		fc1 = tf.nn.relu(tf.matmul(reshaped_concated, weights) + biases,
				name = scope.name)

	# Fully connected layer 2
	with tf.variable_scope('fc2') as scope:
		weights = _variable_with_weight_decay('weights', [4096, NUM_CLASS])
		biases = _variable_on_cpu('biases', [NUM_CLASS], 
				tf.constant_initializer(0.1))
		logits = tf.add(tf.matmul(fc1, weights) + biases,
				name = scope.name)

	# Return unormalised logits
	return logits


def losses(logits, labels):
	"""Compute losses and add L2 regulisations for all trainables
	
	"""
	labels = tf.cast(labels, tf.int32)
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
					logits, labels, name='cross_entropy_per_example')
	cross_entropy_mean = tf.reduce_mean(
					cross_entropy, name='cross_entropy_mean')
	tf.add_to_collection('losses', cross_entropy_mean)
	
 	# The total loss is defined as the cross entropy loss plus 
 	# all of the weight decay terms (L2 loss).
	return tf.add_n(tf.get_collection('losses'), name='total_loss')


if __name__=='__main__':
	# test
	