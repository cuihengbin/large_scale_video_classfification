"""
Read and decode TFRecords files with tf.queue type operation.
"""


tf.app.flags.DEFINE_string('train_directory', '/tmp/UCF101/TFRecords/train/', 
						'output TFRecords training data directory')
tf.app.flags.DEFINE_string('valid_directory', '/tmp/UCF101/TFRecords/valid/', 
						'output TFRecords validation data directory')
tf.app.flags.DEFINE_integer('image_size', 200, 'size of cropped images')

FLAGS = tf.app.flags.FLAGS



def _read_tfrecords(filename_queue):
"""Read TFRecords files from filename_queue

Args:
	filename_queue: tf.queue object, contains list of .TFRecords fileanmes

Returns:
	images_raw: 1D tensor, raw images data
	label: scalar tensor with dtype tf.int32
	class_: string tf.string type
	frame_counts: scaler tensor with dtype tf.int32, number of frames in one 	video
	height,width: scalar tensor with dtype tf.int32
	group,clip: scalar tensor with dtype tf.int32, unique identifier for videos
"""
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)

features = tf.parse_single_example(
serialized_example,
features={
	'images': tf.FixedLenFeature([], tf.string, default_value=''),
	'label': tf.FixedLenFeature([], tf.int64, default_value=-1),
	'class': tf.FixedLenFeature([], tf.string, default_value=''),
	'width': tf.FixedLenFeature([], tf.int64, default_value=-1),
	'height': tf.FixedLenFeature([], tf.int64, default_value=-1),
	'frame_counts': tf.FixedLenFeature([], tf.int64, default_value=-1),
	'group': tf.FixedLenFeature([], tf.int64, default_value=-1),
	'clip': tf.FixedLenFeature([], tf.int64, default_value=-1)
	})
images_raw = tf.decode_raw(features['images'], tf.uint8)
label = tf.cast(features['label'], tf.int32)
class_ = tf.cast(features['class'], tf.string)
frame_counts = tf.cast(features['frame_counts'], tf.int32)
height = tf.cast(features['height'], tf.int32)
width = tf.cast(features['width'], tf.int32)
group = tf.cast(features['group'], tf.int32)
clip = tf.cast(features['clip'], tf.int32)
return images_raw, label, class_, frame_counts, height, width, group, clip


def _generate_image_and_label_batch():

return images, labels


def inputs(train, batch_size):
"""Construct input for the network using the Reader op

Args:
	train: bool, true for training data, else validation data
	batch_size: int, number of examples per batch
Returns:
	images: Images, 5D tensor of [batch_size, frame_counts, height, width, channel]
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
images_raw, label, class_, frame_counts, height, width, group, clip
 = _read_tfrecords(filename_queue)


# Reshape images 
width = width.value # turn into int, for reshape op to consume
images_reshaped =

# unpack videos
images = tf.unpack()


# crop images to a fix dimension 
images_cropped = tf.cropp()





return _generate_image_and_label_batch()


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