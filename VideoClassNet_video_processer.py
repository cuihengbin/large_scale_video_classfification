"""
Read and decode TFRecords files with tf.queue type operation.
"""

tf.app.flags.DEFINE_integer('batch_size', 32,
                            """Number of images to process in a batch.""")

tf.app.flags.DEFINE_integer('image_size', 120, 'size of cropped squared images')

tf.app.flags.DEFINE_float('central_fraction', '0.70', 
						  """Fraction of central area to crop, the produced """
						  """image retains its resolution.""")

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

FLAGS = tf.app.flags.FLAGS

# UCF101 frames resolution
IMAGE_HEIGHT = 240
IMAGE_WIDTH = 320

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
return images_uint8, label




def _distort_color(image, thread_id=0, scope=None):
  """Distort the color of the image.

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
  with tf.op_scope([image], scope, 'distort_color'):
    color_ordering = thread_id % 2

    if color_ordering == 0:
      image = tf.image.random_brightness(image, max_delta=32. / 255.)
      image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      image = tf.image.random_hue(image, max_delta=0.2)
      image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    elif color_ordering == 1:
      image = tf.image.random_brightness(image, max_delta=32. / 255.)
      image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      image = tf.image.random_hue(image, max_delta=0.2)

    # The random_* ops do not necessarily clamp.
    #image = tf.clip_by_value(image, 0.0, 1.0)
    return image


def batch_inputs(dataset, batch_size, train, num_preporcess_threads=8):
	"""
	
	Args:
		dataset: instance od Dataset class specifying the dataset
		batch_size: int
		train: boolean
		num_preprocess_threads: integer, total number of preprocessing threads

	Returns:
		context_images: 4-D float Tensor of a batch of low resolution images
		fovea_images: 4-D float Tensor of a batch of central high resolution images
		labels: 1-D integer Tensor of [batch_size]
	"""
	with tf.name_scope('batch_porcessing'):
		filenames = dataset.filenames

		# Create filename queue
		if train:
			filename_queue = tf.train.string_input_producer(fileanmes,
															shuffle=True,
															capacity=50)
		else:
			filename_queue = tf.train.string_input_producer(filenames,
															shuffle=False,
															capacity=1)
		# Parse a serialized Example proto to extract the images and metadata.
		reader = dataset.reader()
		_, serialized_example = reader.read(filenamequeue)

		fovea_images_and_labels = []
		context_images_and_labels = []
		for thread_id in range(FLAGS.num_preprocess_threads):
			# each call dequeues one example from filenamequeue
			# For multiple frames based sampling strategy, consider pull frame_counts 
			# out, also add a tf.cond op for generization below.
			images_uint8, label_index = _parse_example_proto(serialized_example)

			# Reshape images
			# Caution! This only works for frame_counts = 1
			images_reshaped = tf.reshape(images_uint8, (IMAGE_HEIGHT, IMAGE_WIDTH, 3))

			# Crop images into a square one with FLAG.image_size 
			images_cropped = tf.image.resize_image_with_crop_or_pad(images_reshaped,
															240, 240)
															
			# Distor cropped images for trainning examples
			if train:
				# Randomly flip the image horizontally.
				distorted_image = tf.image.random_flip_left_right(distorted_image)

				# Randomly distort the colors.
				distorted_image = distort_color(distorted_image, thread_id)
			else:
				# Remain the same for evaluation
				distorted_image = images_cropped

			## Produce two streams of images, context_images and centered fovea_images
			# Crop the central region of the image with an area containing FLAG.central_fraction
			# of the original image. And then resize back to FLAG.image_size.

			fovea_image = tf.image.resize_image_with_crop_or_pad(image, 
																FLAG.image_size,
																FLAG.image_size)

			# Use max_pool op to downsample image to produce a context stream. 
			# the resultant shape has to be identical to ones in fovea_image
			# this is done by setting ksize=[2,2,1] and stride=[2,2,1]
			# Formula for dimensions of pooled tensor is:
			# W2=(W1−F)/S+1W2=(W1−F)/S+1
			# H2=(H1−F)/S+1H2=(H1−F)/S+1
			# D2=D1

			context_image = max_pool() 

			assert context_image.shape.value[0] == 120


			# Whitening both streams, after whitenning, image type is converted 
			# from tf.uint8 to tf.float32
			fovea_image = tf.image.per_image_whitening(fovea_image)
			context_image = tf.image.per_image_whitening(context_image)

			fovea_images_and_labels.append([fovea_image, label_index])
			context_images_and_labels.append([context_image, label_index])

		# batching
		fovea_batch, label_batch = tf.train.batch_join(
			fovea_images_and_labels,
			batch_size=FLAG.batch_size,
			capacity=2 * num_preprocess_threads * FLAG.batch_size)
		context_batch, label_batch = tf.train.batch_join(
			context_images_and_labels,
			batch_size=FLAG.batch_size,
			capacity=2 * num_preprocess_threads * FLAG.batch_size)

		# Display the training images in the visualizer.
		tf.image_summary('fovea_images', fovea_batch)
		tf.image_summary('context_images', context_batch)

		return fovea_batch, context_batch, tf.reshape(label_batch, [FLAG.batch_size])



if __name__=='__main__':
filenames = ['/tmp/UCF101/TFRecords/train/train-00000-of-01000.TFRecords']#,
       # '/tmp/UCF101/TFRecords/train/train-00001-of-01000.TFRecord',
      #  '/tmp/UCF101/TFRecords/train/train-00002-of-01000.TFRecord']
with tf.Session() as sess:
filename_queue = tf.train.string_input_producer(filenames)
images, label, width, height, frame_counts, class_ = read_tfrecords(filename_queue)
images = tf.reshape(images, tf.pack([frame_counts, height, width,3]))
init_op = tf.initialize_all_variables()
sess.run(init_op)
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord)
for i in range(20):
    
    images_ = sess.run([images])
    
    images_ = np.array(images_).reshape([240, 320, 3])
    plt.imshow(images_[1:,1:,...])
    plt.show()
coord.request_stop()
coord.join(threads)