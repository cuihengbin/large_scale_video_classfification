"""
Build the Network of VideoClassNet

# Formula for dimensions of the output of a conv layer
# W2=(W1−F+2P)/S+1. Where, F kernel size, P is padding
# S is stride.
"""


import tensorflow as tf

from VideoClassNet_video_processer import batch_inputs




FLAGS = tf.app.flags.FLAGS

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

def _singleframe_layer_before_fc(image, scope):
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

	return last_layer_before_fc
	"""




def inference(fovea_batch, context_batch):
	"""Produce losses across all videos in a batch
	

		Returns:
			logits: 1D tensor with length=num_class,
				 unnormalized likelyhood probability averaged 
				 across a abtch
	"""

	# Multi-threading, each thread return a loss from a video


	logit_video = []
	for fovea_video, context_video in zip(fovea_batch, context_batch):
		
		for fovea_image, context_image in zip(fovea_video, context_video):
			# Porcess two streams seperately
			fovea_layer_before_fc = _singleframe_layer_before_fc(fovea_image)
			context_layer_before_fc = _singleframe_layer_before_fc(context_image)
		# Connect two streams to fc

		fc1 = tf...

		weights = _variable_on_cpu( , shape[ ,NUM_CLASS])
		logits = tf.matmul(2Dtensor multiplied) + biases
		
		logit_video.append(soft_max)
	logits = tf.reduce_mean(logit_video)
	return logits

def losses(logits, label_batch):

	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels)
	cross_entropy_mean = tf.reduce_mean(cross_entropy)

	return cross_entropy_mean #or tf.add_n(tf.ge_collection('losses')) for L2 term



def train(total_loss, global_step):


	return train_op