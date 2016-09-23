"""
Convert video files to TFexample files
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import sys
import threading

import cv2
import numpy as np
import tensorflow as tf

tf.app.flags.DEFINE_string('video_data_directory', '/tmp/UCF101/videos/',
							'Extracted UCF101 data, contains *.avi')
tf.app.flags.DEFINE_string('ucfTrainTestlist', '/tmp/UCF101/ucfTrainTestlist/',
							'files specifying trainning and testing datasets')
tf.app.flags.DEFINE_string('sampling_strategy', 'single', 
							'sampling strategy for sampling frames from a video')
tf.app.flags.DEFINE_string('output_train_directory', '/tmp/UCF101/TFRecords/train/', 
							'output TFRecords training data directory')
tf.app.flags.DEFINE_string('output_valid_directory', '/tmp/UCF101/TFRecords/valid/', 
							'output TFRecords validation data directory')
tf.app.flags.DEFINE_string('output_test_directory', '/tmp/UCF101/TFRecords/test/', 
							'output TFRecords testing data directory')

tf.app.flags.DEFINE_integer('num_threads', 10, 
							'Number of threads to preporcess the videos')
tf.app.flags.DEFINE_integer('train_shards', 1000,
                            'Number of shards in training TFRecord files.')
tf.app.flags.DEFINE_integer('valid_shards', 200,
                            'Number of shards in validation TFRecord files.')
tf.app.flags.DEFINE_integer('test_shards', 600,
                            'Number of shards in validation TFRecord files.')



FLAGS = tf.app.flags.FLAGS

SQUARE_SIZE = 178

def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  if type(value) == str:
  	value = str.encode(value)
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(video_filename, images, class_label):
	"""Build an Example proto for an example

	Args:
		video_filename: string, path to a video
		images: np.array, with size [frame_counts*height*weight*channel]
		class_label: dictionary, mapping from class to label
	Returns:
		Exanmple proto with features
			images: string, converted from images array
			label: integer
			class: string, unique human-readable
			frame_counts: int, number of frames within the buffer
			height: integer
			weight: integer
			sampling_strategy: string, sampling strategy
			group: int, identifier for source large video 
			clip: int, identifier for clips in the source large video
	"""
	# Parse filename, class, group, clip, from video_filename which has format
	# e.g. '/tmp/UCF101/videos/v_Knitting_g15_c06.avi'
	filename = os.path.basename(video_filename)
	_, class_name, group, clip = filename.replace('.avi',"").split('_')
	label = class_label[class_name]

	# Parse dimension info frome image shape
	frame_counts, height, weight, channel = images.shape

	assert channel == 3 #make sure its RGB images 
	if FLAGS.sampling_strategy == 'single':
		assert frame_counts == 1

	example = tf.train.Example(features=tf.train.Features(feature={
		'images': _bytes_feature(images.tobytes()),
		'label': _int64_feature(label),
		'class': _bytes_feature(class_name),
		'frame_counts': _int64_feature(frame_counts),
		'height': _int64_feature(height),
		'weight': _int64_feature(weight),
		'channel': _int64_feature(channel),
		'group': _int64_feature(int(group[1:])),
		'clip': _int64_feature(int(clip[1:])),
		'sampling_strategy': _bytes_feature(FLAGS.sampling_strategy)}))
	return example


def _extract_imgs(name, video_filename, strategy):
	"""
	Given a video file, returns its frames bsaed on selected strategy

	Args:
		video_filename: 
		strategy: str, specifying which strategy to use. e.g
			single: Extract the middle frame of the given video
			slow, early, late
	
	Return:
		images: numpy.array([frame_counts, height, weight, channel])
		label: int32, the ground truth 
	"""
	vidcap = cv2.VideoCapture(filename=video_filename)

	images = []
	total_frames = vidcap.get(7)
	if total_frames < 3:
		print('{0} only has {1} frames, skipped!'
			.format(video_filename, total_frames))
		return np.array([])
	if strategy == 'single':
		images_per_vid = 1

		# Move video indexer to the middle
		middle_frame_index = int(total_frames/2)
		vidcap.set(1, middle_frame_index)

		# Read image 
		success, img = vidcap.read()

		# Resize img to square
		if name == 'train':
			# Random crop a SQUARE_SIZE img
			reshaped_img = tf.random_crop(img, [SQUARE_SIZE, SQUARE_SIZE, 3])
		else:
			# Crop the central SQUARE_SIZE img
			reshaped_img = tf.image.resize_image_with_crop_or_pad(img,
							SQUARE_SIZE, SQUARE_SIZE)

		# Whitenning img
		float_image = tf.image.per_image_whitening(reshaped_img)

		if success:
			images += [float_image]
		else:
			print('Video %s capture failed! Skipped!')
			return np.array([])
	images = np.array(images)
	assert images.shape[0] == images_per_vid
    
	return images
	


def _filename_parser(file_list_name):
	"""Pares filenames in given file_list_name
	"""
	f = open(file_list_name, 'r')
	filenames = []
	while True:
		line = f.readline().split(' ')
		if line[0] == "": 
			break
		if len(line) == 1:
			# test files
			filenames += [line[0].replace('\n',"")] # as last 2 strings are '\n'
		else:
			# train files
			filenames += [line[0]]
	f.close()
	return filenames



def _lookup_dicts():
	"""
	Return: A dictionary
	"""

	# check video dir and tr_te_data dir exists
	if not (tf.gfile.Exists(FLAGS.video_data_directory) or
			tf.gfile.Exists(FLAGS.ucfTrainTestlist)):
			raise ValueError('No valid dataset has been found.')

	# parse class label data
	class_label_fn = os.path.join(FLAGS.ucfTrainTestlist, 'classInd.txt')
	class_label = dict()
	f = open(class_label_fn, 'r') 
	while True:
		try:
			line = f.readline()
			label, _class = line.split(' ')
			class_label[_class.replace('\n',"")] = int(label)
		except Exception:
			f.close()
			break

	# parse video filenames for training, validation, testing
	# original seperations are three chunk of files for trainning and three 
	# for testing, we use one of the testing files as validation set
	train_files_list = [os.path.join(FLAGS.ucfTrainTestlist,trainlist) 
						for trainlist in os.listdir(FLAGS.ucfTrainTestlist) 
							if 'trainlist' in trainlist]
	test_files_list = [os.path.join(FLAGS.ucfTrainTestlist,testlist) 
						for testlist in os.listdir(FLAGS.ucfTrainTestlist) 
							if 'testlist' in testlist]
	valid_files_list = [test_files_list[0]]
	test_files_list.pop(0)

	# format in list.txt files are
	# ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c01.avi 1
	# ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c02.avi 1
	# since we have class label dict already, we simply ignore label info
	data_split_filenames = dict()
	data_split_filenames['train_filenames'] = [os.path.join(FLAGS.video_data_directory, filenames.split('/')[1]) 
												for fn in train_files_list
                                               	for filenames in _filename_parser(fn)]
	data_split_filenames['valid_filenames'] = [os.path.join(FLAGS.video_data_directory, filenames.split('/')[1]) 
												for fn in valid_files_list 
                                               for filenames in _filename_parser(fn)]
																										
	data_split_filenames['test_filenames'] = [os.path.join(FLAGS.video_data_directory, filenames.split('/')[1]) 
												for fn in test_files_list 
                                              for filenames in _filename_parser(fn)]

	for k, v in data_split_filenames.items():
		print('{0} videos in {1} data set'.format(len(v), k.split('_')[0]))

	return class_label, data_split_filenames



def _process_video_files_batch(name, thread_index, ranges, filenames, output_directory, 
								num_shards, class_label):
	"""Processes and saves list of images as TFRecord in 1 thread.
	
	"""
	# num_shards must divisiable by num_threads
	num_threads = len(ranges)
	assert not num_shards % num_threads 

	# Construct shards_ranges for current thread_index
	# i.e. Breaks ranges[thread_index] to shards_ranges
	num_shards_per_threads = int(num_shards / num_threads)
	shard_ranges =  np.linspace(ranges[thread_index][0],
								ranges[thread_index][1],
								num_shards_per_threads +1).astype(int)
	num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

	counter = 0
	failed_video_counts = 0
	for s in range(num_shards_per_threads):
		# Generate a sharded version of the file name, e.g. 'train-00002-of-01000'
		shard = thread_index * num_shards_per_threads + s
		output_filename = '%s-%.5d-of-%.5d.TFRecords' % (name, shard, num_shards)
		output_file = os.path.join(output_directory, output_filename)
		writer = tf.python_io.TFRecordWriter(output_file)

		shard_counter = 0
		files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
		for i in files_in_shard:
			filename = filenames[i]
			images = _extract_imgs(name, filename, FLAGS.sampling_strategy)

			# Check images have correct shape and contains no NaN
			if images.shape == (1,240,320,3) and \
				np.isnan(sum(images)).sum() == 0:
				# Write images 
				example = _convert_to_example(filename, images, class_label)
				writer.write(example.SerializeToString())
				shard_counter += 1
				counter += 1

				#print('{0} [thread {1}]: Processed {2} of {3} videos in thread batch.'
				#	.format(datetime.now(), thread_index, counter, num_files_in_thread))
				#sys.stdout.flush()
			else:
				failed_video_counts += 1

		print('{0} [thread {1}]: Wrote {2} videos to {3}'
			.format(datetime.now(), thread_index, shard_counter, output_filename))
		sys.stdout.flush()
	print('{0} [thread {1}]: Wrote {2} videos to {3} shards.'
		.format(datetime.now(), thread_index, counter, num_files_in_thread))
	sys.stdout.flush()

	print('Bad Images: ', failed_video_counts)






def _process_dataset(name, filenames, output_directory, num_shards, class_label):
	"""Multi-threading for Process a complete data set and save it as a TFRecord.

	Args:
		name: string, name of the dataset
		filenames: list of string, filenames of video data
		output_directory, string, where to output
		num_shards: int, number of shards in the dataset
		class_label: dictionary, mapping from class to label
	"""
	# Break all videos into batches with a [ranges[i][0], ranges[i][0]].
	spacing = np.linspace(0, len(filenames), FLAGS.num_threads +1).astype(np.int)
	ranges = []
	threads = []
	for i in range(len(spacing) -1):
		ranges += [[spacing[i], spacing[i+1]]]

	# Launch a thread for each batch
	print('Launching %d threads for spacings: %s' % (FLAGS.num_threads, ranges))
	sys.stdout.flush()

	# Create a mechanism for monitoring when all threads are finished
	coord = tf.train.Coordinator()

	#print(ranges)
	#_process_video_files_batch(name, 1, ranges, filenames, output_directory, num_shards, class_label)
	
	threads = []
	for thread_index in range(len(ranges)):
		args = (name, thread_index, ranges, filenames, output_directory, num_shards, class_label)
		t = threading.Thread(target=_process_video_files_batch, args= args)
		t.start()
		threads += [t]

	# wait for all the threads to terminate.
	coord.join(threads)
	print('%s: Finished writing all %d videos in data set.' %
		(datetime.now(), len(filenames)))
	sys.stdout.flush()
	


def main(unused_args):

	for output_dir in [FLAGS.output_train_directory, 
					   FLAGS.output_valid_directory,
					   FLAGS.output_test_directory]:
		if not tf.gfile.Exists(output_dir):
			print('%s not exists, creating one.' % output_dir)
			tf.gfile.MakeDirs(output_dir)
	class_label, data_split_filenames = _lookup_dicts()

	# Missing class labels in original file
	class_label['HandStandPushups'] = 102

	_process_dataset('train', data_split_filenames['train_filenames'],
					FLAGS.output_train_directory, 1000, class_label)
	_process_dataset('valid', data_split_filenames['valid_filenames'],
					FLAGS.output_valid_directory, 200, class_label)
	_process_dataset('test', data_split_filenames['test_filenames'],
					FLAGS.output_test_directory, 600, class_label)

	for k,v in class_label.items():
		print(k,v)

if __name__ == '__main__':
	tf.app.run()
