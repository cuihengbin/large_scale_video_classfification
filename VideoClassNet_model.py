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

def _variable_with_weight_decay(name, shape, stddev, wd):
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

def _singleframe_network_before_fc(image, scope):

	return last_layer_before_fc


def inference(fovea_batch, context_batch):
	"""Build the model

	Args:
		fovea_batch: Tensor with size [batch_size, height, weight, 3] 
			batch of fovea stream images
		context_batch: Tensor with size [batch_size, height, weight, 3] 
			batch of context stream images

	Returns:
		logits

	"""
	# 


	return logits


def losses(fovea_batch, context_batch, label_batch):
	"""Produce losses across all videos in a batch

	"""

	# Multi-threading, each thread return a loss from a video


	logits_batch = []
	for fovea_video, context_video in zip(fovea_batch, context_batch):
		logit_video = []
		for fovea_image, context_image in zip(fovea_video, context_video):
			# Porcess two streams seperately
			fovea_layer_before_fc = _singleframe_layer_before_fc(fovea_image)
			context_layer_before_fc = _singleframe_layer_before_fc(context_image)
		# Connect two streams to fc
		



	return loss


def train():

	return cross_entropy


