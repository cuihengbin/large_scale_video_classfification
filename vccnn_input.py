"""
Covert video data to TFexample files
"""


def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example():
	"""Build an Example proto for an example

	Args:
		filename:
		video_encoded: string,
		label: integer, 
		text: string, unique human-readable
		frame_counts: int, number of frames within the buffer
		height: integer,
		weight: integer,
		sampling_strategy: string
		group: int,
		clip: int, 

	Returns:
		Example proto
	"""
	pass


def _extract_imgs(video_filename, label_lookup, strategy='single'):
	"""
	Extract the middle frame of the given video

	Args:
		video_filename: 
		label_lookup: dictionary with keys are txt describtion, values are ground truth labels
		strategy: str, specifying which strategy to use. e.g
			single, slow, early, late
	
	Return:
		images: numpy.array([frame_counts, height, weight, channel])
		label: int32, the ground truth 
		txt: str, describtion of the image
	"""
	pass




def _label_lookup():
	"""
	Return: A dictionary
	"""
	return dict()


def _process_video_files_batch(thread_index, ranges, name, filenames,
								text, labels):
	"""Processes and saves list of images as TFRecord in 1 thread.
	
	"""
	pass


def _process_video_files(name, filenames, texts, labels):
	"""Multi-threading

	"""
	pass
