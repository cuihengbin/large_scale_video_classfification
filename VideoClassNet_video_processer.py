"""
Read and decode TFRecords files with tf.queue type operation.

Differences from the paper:
	Raw frames size
	Color distortion.
	Mean value subtracted.
"""

import os
from datetime import datetime

import tensorflow as tf
import numpy as np

from datasets import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)



#tf.app.flags.DEFINE_integer('batch_size', 32,
#                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('num_threads', 10,
							"""Number of threads to run batching""")



# Images are preprocessed asynchronously using multiple threads specified by
# --num_preprocss_threads and the resulting processed images are stored in a
# random shuffling queue. The shuffling queue dequeues --batch_size images
# for processing on a given Inception tower. A larger shuffling queue guarantees
# better mixing across examples within a batch and results in slightly higher
# predictive performance in a trained model. Empirically,
# --input_queue_memory_factor=16 works well. A value of 16 implies a queue size
# of 1024*16 images. Assuming RGB 299x299 images, this implies a queue size of
# 16GB. If the machine is memory limited, then decrease this factor to
# decrease the CPU memory footprint, accordingly.
tf.app.flags.DEFINE_integer('input_queue_memory_factor', 16,
                            """Size of the queue of preprocessed images. """
                            """Default is ideal but try smaller values, e.g. """
                            """4, 2 or 1, if host memory is constrained. See """
                            """comments in code for more details.""")
tf.app.flags.DEFINE_integer('frame_counts', 1, 
							"""Number of frames in a video."""
							"""Depends on sampling strategy."""
							"""Single-frame: 1"""
							"""Others?""")

FLAGS = tf.app.flags.FLAGS

# UCF101 raw frame resolution
IMAGE_HEIGHT = 240
IMAGE_WIDTH = 320

# Frames will first cropped to a size (178, 178)
# Further resized to (89, 89) after gen fovea, context processes
# Aside, 89 is constrained by max_pool operation with ksize=2, stride=2




def _parse_example_proto(serialized_example):
	"""Read TFRecords files from filename_queue

	Args:
		serialized_example: 

	Returns:
		images_raw: 1D tensor, raw images data
		label: scalar tensor with dtype tf.int32
		class_: string tf.string type
		frame_counts: scaler tensor with dtype tf.int32, number of frames in one 	video
		height,width: scalar tensor with dtype tf.int32
		group,clip: scalar tensor with dtype tf.int32, unique identifier for videos
	"""

	features = tf.parse_single_example(
	serialized_example,
	features={
		'images': tf.FixedLenFeature([], tf.string, default_value=''),
		'label': tf.FixedLenFeature([], tf.int64, default_value=-1),
		'class': tf.FixedLenFeature([], tf.string, default_value=''),
		'frame_counts': tf.FixedLenFeature([], tf.int64, default_value=-1),
		'group': tf.FixedLenFeature([], tf.int64, default_value=-1),
		'clip': tf.FixedLenFeature([], tf.int64, default_value=-1)
		})
	images_uint8 = tf.decode_raw(features['images'], tf.uint8)
	label = tf.cast(features['label'], tf.int32)
	class_ = tf.cast(features['class'], tf.string)
	frame_counts = tf.cast(features['frame_counts'], tf.int32)
	group = tf.cast(features['group'], tf.int32)
	clip = tf.cast(features['clip'], tf.int32)
	images = tf.cast(images_uint8, tf.float32)
	return images, label




def _distort_image(image, seed, scope=None):
	"""Distort the color and oreitation of the image

	Each color distortion is non-commutative and thus ordering of the color ops
	matters. Ideally we would randomly permute the ordering of the color ops.
	Rather then adding that level of complication, we select a distinct ordering
	of color ops for each preprocessing thread.

	Args:
	image: Tensor containing single image.
	thread_id: preprocessing thread ID.
	scope: Optional scope for op_scope.
	Returns:
	color-distorted image
	"""
	with tf.op_scope([image], scope, 'distort'):
		# Random flip
		image = tf.image.random_flip_left_right(
			image, seed=seed)

		# Distor color
		image = tf.image.random_brightness(
			image, max_delta=32. / 255., seed=seed)
		image = tf.image.random_saturation(
			image, lower=0.5, upper=1.5, seed=seed)
		image = tf.image.random_hue(
			image, max_delta=0.2, seed=seed)
		image = tf.image.random_contrast(
			image, lower=0.5, upper=1.5, seed=seed)

	return image

def _gen_fovea_stream(image):
	"""Produce a high resolution centered images

	Args:
		images: 3D tensor. with shape 
			[160, 160, 3]

	Returns:
		fovea_stream: 3D tensor, with shape 
			[120, 120, 3]
	"""
	tf.assert_rank(image, 3)
	fovea_stream = tf.image.resize_image_with_crop_or_pad(
		image, 89, 89)
	return fovea_stream




def _gen_context_stream(images):
	"""Produce a low resolution images

	# Use max_pool op to downsample image to produce a context stream. 
	# the resultant shape has to be identical to ones in fovea_image
	# this is done by setting ksize=[2,2,1] and stride=[2,2,1]
	# Formula for dimensions of pooled tensor is:
	# W2=(W1−F)/S+1W2=(W1−F)/S+1
	# H2=(H1−F)/S+1H2=(H1−F)/S+1
	# D2=D1

	Args:
		images: 4D tensor. with shape 
			[frame_count, 178, 178, 3]

	Returns:
		images: 4D tensor, which has shape 
			[frame_count, 89, 89]
	"""

	tf.assert_rank(images, 4)
	context_stream = tf.nn.max_pool(images, 
		ksize=[1,2,2,1], strides=[1,2,2,1],padding='VALID')
	assert context_stream.get_shape()[1:] == (89, 89, 3)
	return context_stream



def batch_inputs(dataset, train):
	"""
	
	Args:
		dataset: instance od Dataset class specifying the dataset
		batch_size: int
		train: boolean
		num_preprocess_threads: integer, total number of preprocessing threads

	Returns:
		context_images: list of 4D tensors. 
		fovea_images: 4-D float Tensor of a batch of central high resolution images
		labels: 1-D integer Tensor of [batch_size]
	"""

	reader = tf.TFRecordReader()
	with tf.name_scope('batch_porcessing'):
		filenames = dataset.filenames

		# Create filename queue
		if train:
			filename_queue = tf.train.string_input_producer(filenames,
															shuffle=True,
															capacity=50)
		else:
			filename_queue = tf.train.string_input_producer(filenames,
															shuffle=False,
															capacity=1)
		# Parse a serialized Example proto from filename_queue.
		#reader = dataset.reader
		_, serialized_example = reader.read(filename_queue)
		images_str, label_index = _parse_example_proto(serialized_example)
		

		# Reshape images to size (frame_counts, height, width, channel)
		# Value of the first dimension depends on the sampling trategy
		images_reshaped = tf.reshape(images_str, (
			FLAGS.frame_counts, IMAGE_HEIGHT, IMAGE_WIDTH, 3))

		# Resize images to a square shape, with size equal to  178
		# to be consistent with the paper
		images_reshaped =  [
			tf.image.resize_image_with_crop_or_pad(
				images_reshaped[i], 178, 178)
				for i in range(FLAGS.frame_counts)]
		images_reshaped = tf.pack(images_reshaped, axis=0)


		if train:
			# Retrive FLAGS.batch_size examples. Under the hood, its fires
			# batch_size threads to run enqueue ops.
			images_batch, label_batch = tf.train.shuffle_batch(
				[images_reshaped, label_index], 
				batch_size=FLAGS.batch_size,
				num_threads=FLAGS.num_threads,
				capacity=500,
				min_after_dequeue=250)
		else: 
			images_batch, label_batch = tf.train.batch(
				[images_reshaped, label_index],
				batch_size=1,
				num_threads=FLAGS.num_threads)
		
		# Apply flip and color distrotion consistently for images 
		# in the same clip.
		images_in_batch_list = tf.unpack(images_batch, axis=0)
		tf.assert_rank(images_in_batch_list[0], 4)

		fovea_stream_list = []
		context_stream_list = []
		for seed, images in enumerate(images_in_batch_list):
			images_list = tf.unpack(images, axis=0)

			assert len(images_list)==FLAGS.frame_counts 
			tf.assert_rank(images_list[0], 3)

			if train:
				images_list_distored = [
					_distort_image(img, seed) for img in images_list]
				# Generates fovea and context streams, seperately.
				# Since fovea stream operates on 3D tensors. 
				# And context stream operates on 4D tensors.
				fovea_stream_list.append(tf.pack(
					[_gen_fovea_stream(img) for img in images_list_distored]))
					
				context_stream_list.append(
					_gen_context_stream(tf.pack(images_list_distored, axis=0)))
				
			else:
				fovea_stream_list.append(tf.pack(
					[_gen_fovea_stream(img) for img in images_list]))
				context_stream_list.append(
					_gen_context_stream(tf.pack(images_list, axis=0)))
				break

		assert len(fovea_stream_list) == len(context_stream_list) 
		
		# sutract a mean value. TODO, Consider whitenning 
		mean_tensor = tf.constant(
			86, dtype=tf.float32, shape=(FLAGS.frame_counts, 89, 89, 3))

		fovea_stream_list = [tf.sub(video, mean_tensor) 
							for video in fovea_stream_list] 
		context_stream_list = [tf.sub(video, mean_tensor) 
								for video in context_stream_list]

		# Merge the first and second dimensions of the two streams
		# the result fisrt dimension accounts for all frames in a batch
		fovea_stream_batch = tf.reshape(tf.pack(fovea_stream_list),
							[FLAGS.batch_size*FLAGS.frame_counts, 
							89, 89, 3])
		context_stream_batch = tf.reshape(tf.pack(context_stream_list),
							[FLAGS.batch_size*FLAGS.frame_counts, 
							89, 89, 3])

		# Display the training images in the visualizer.
		tf.image_summary('fovea_images', fovea_stream_batch)
		tf.image_summary('context_images', context_stream_batch)

		return fovea_stream_batch, context_stream_batch, label_batch



if __name__=='__main__':
	config = tf.ConfigProto(inter_op_parallelism_threads=50)
	with tf.Session(config=config) as sess:
		dataset_train = Dataset()

		fovea_stream_list, context_stream_list, label_batch = batch_inputs(dataset_train, True)

		print(type(fovea_stream_list[0]))
		#print(dataset_train.filenames)
		init_op = tf.initialize_all_variables()
		sess.run(init_op)
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)
		#f.train.start_queue_runners(sess=sess)
		for i in range(2000):
			fovea_stream_list_,context_stream_list_,label_batch_ = \
				sess.run([fovea_stream_list,context_stream_list,label_batch])

			print(label_batch_)
			print(len(label_batch_))
			print()
		coord.request_stop()
		coord.join(threads)