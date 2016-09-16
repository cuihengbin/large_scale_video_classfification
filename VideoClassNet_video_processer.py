"""
Read and decode TFRecords files with tf.queue type operation.
"""


tf.app.flags.DEFINE_string('train_directory', '/tmp/UCF101/TFRecords/train/', 
							'output TFRecords training data directory')
tf.app.flags.DEFINE_string('valid_directory', '/tmp/UCF101/TFRecords/valid/', 
							'output TFRecords validation data directory')

FLAGS = tf.app.flags.FLAGS

def 

def _read_tfrecords(filename_queue):
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)

	features = tf.parse_single_example(
	serialized_example,
	features={
		'images': tf.FixedLenFeature([], tf.string, default_value=''),
		'label': tf.FixedLenFeature([], tf.int64, default_value=-1),
		'class': tf.FixedLenFeature([], tf.string, default_value=''),
		'weight': tf.FixedLenFeature([], tf.int64, default_value=-1),
		'height': tf.FixedLenFeature([], tf.int64, default_value=-1),
		'frame_counts': tf.FixedLenFeature([], tf.int64, default_value=-1),
		})
	images = tf.decode_raw(features['images'], tf.uint8)
	label = tf.cast(features['label'], tf.int32)
	weight = tf.cast(features['weight'], tf.int32)
	height = tf.cast(features['height'], tf.int32)
	frame_counts = tf.cast(features['frame_counts'], tf.int32)
	class_ = tf.cast(features['class'], tf.string)
	return images, label, weight, height, frame_counts, class_

def _generate_image_and_label_batch():

	return images, labels


def inputs(train, batch_size, distort= True):
	"""Construct input for the network using the Reader op

	Args:
		train: bool, true for training data, else validation data
		batch_size: int, number of examples per batch
		distort: bool, whethear to distort video images or not

	Returns:
		images: Images, 5D tensor of [batch_size, frame_counts, height, weight, channel]
		labels: Lables, 1D tensor of [batch_size] size.

	"""

	if train:
		data_dir = FLAGS.train_directory
	else:
		data_dir = FLAGS.valid_directory

	# Get filenames in the data_dir
	filenames = os.listdir(data_dir)
	assert len(filenames) < 10, 'Failed to find files.'

	# Put filenames into tf.queue op for read later
	filename_queue = tf.train.string_input(filenames)

	# Read examples from files in the filename queue.
	read_input = _read_tfrecords(filename_queue)
	reshaped_videos = tf.cast()

	# Preporcessing images across videos

	return _generate_image_and_label_batch()


if __name__=='__main__':
	filenames = ['/tmp/UCF101/TFRecords/train/train-00000-of-01000.TFRecords']#,
	       # '/tmp/UCF101/TFRecords/train/train-00001-of-01000.TFRecord',
	      #  '/tmp/UCF101/TFRecords/train/train-00002-of-01000.TFRecord']
	with tf.Session() as sess:
	filename_queue = tf.train.string_input_producer(filenames)
	images, label, weight, height, frame_counts, class_ = read_tfrecords(filename_queue)
	images = tf.reshape(images, tf.pack([frame_counts, height, weight,3]))
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