"""Train the model
"""

import time
import os

import numpy as np
import tensorflow as tf

from datasets import Dataset
from VideoClassNet_video_processer import batch_inputs
from VideoClassNet_model import inference, losses


tf.app.flags.DEFINE_integer('batch_size', 32,
                          """Number of images to process in a batch.""")
tf.app.flags.DEFINE_float('initial_lr', 0.,
					    """Initial learning rate""")
tf.app.flags.DEFINE_float('lr_decay_rate', 0.9,
					    """Initial learning rate""")

tf.app.flags.DEFINE_integer('max_steps', 50000,
                          """Number of steps to train.""")
tf.app.flags.DEFINE_string('summary_path', 'summary',
						"""Directory to save summaries""")
tf.app.flags.DEFINE_string('ckpt_path', 'ckpt',
						"""Directory to checkpoint files""")

FLAGS = tf.app.flags.FLAGS

NUM_EXAMPLES_PER_EPOCH = 10000
NUM_EPOCHS_PER_DECAY = 3



def _train_op(total_loss, global_step):
	"""Train the model

	Args:
		total_losses: 1D tensor with length [total_frames]
		global_step: Integer variable counting the number of
			training steps porcessed.

	Returns:
		train_op: op for training.
	"""
	num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH // FLAGS.batch_size
	decay_steps = num_batches_per_epoch * NUM_EPOCHS_PER_DECAY

	# Decay the learning rate exponentially based on the number of steps.
	lr = tf.train.exponential_decay(FLAGS.initial_lr,
									global_step,
									decay_steps,
									FLAGS.lr_decay_rate,
									staircase=True)
	tf.scalar_summary('learning_rate', lr)

	# Choose an optimizer and minimise loss
	optimizer = tf.train.GradientDescentOptimizer(lr)
	train_op = optimizer.minimize(total_loss, global_step=global_step)

	return train_op



def train():
	"""Train VideoClassNet model number of steps.
	"""
	global_step = tf.Variable(0, trainable=False)

	# Instanise a Dataset
	dataset = Dataset(used_for='train')

	# Get videos 
	fovea_stream_batch, context_stream_batch, label_batch = batch_inputs(
															dataset, True)

	#  Build a graph and get logits
	logits = inference(fovea_stream_batch, context_stream_batch, FLAGS.batch_size)

	# Ops for retriving losses
	total_loss = losses(logits, label_batch)
	tf.scalar_summary('loss', total_loss)

	# Optimise the trainable variables by minimise the loss
	train_op = _train_op(total_loss, global_step)

	# Create a saver
	saver = tf.train.Saver(tf.all_variables())

	# Build the summary operation based on the TF collection of Summaries.
	summary_op = tf.merge_all_summaries()

	# Build an initialzation op to run below
	init = tf.initialize_all_variables()

	# Start the session
	sess = tf.Session(config=tf.ConfigProto(
        log_device_placement=True))
	sess.run(init)

	# Start the queue runners
	tf.train.start_queue_runners(sess=sess)

	summary_writer = tf.train.SummaryWriter(FLAGS.summary_path, sess.graph)

	for step in range(FLAGS.max_steps):
		# train for each step
		"""
		f,c,l = sess.run([fovea_stream_batch, context_stream_batch, label_batch])
		
		print(f.mean(), 'mean for fovea')
		print(c.mean(), 'mean for context')
		print(l.mean(), 'mean for labels')

		if step == 50:
			break
		"""

		for var in tf.get_collection('weights'):
			print(sess.run(var).sum())

		
		start_time = time.time()
		f,c,l,_, loss_value = sess.run([fovea_stream_batch, context_stream_batch, label_batch, train_op, total_loss])
		sec_per_batch = time.time() - start_time

		assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
		if np.isnan(loss_value):
			#f,c,l = sess.run([fovea_stream_batch, context_stream_batch, label_batch])
			
			print(f.mean(), 'mean for fovea')
			print(c.mean(), 'mean for context')
			print(l.mean(), 'mean for labels')
			break

		if step % 10 == 0:
			example_per_sec = FLAGS.batch_size / sec_per_batch
			print('step: %d, loss = %.2f, (%.1f example/sec; %.1f sec/batch)' %
				(step, loss_value, example_per_sec, sec_per_batch))

		# save summaries
		if step % 100 == 0:
			summary_str = sess.run(summary_op)
			summary_writer.add_summary(summary_str, step)
			pass
		# Save the model periodically
		if step % 1000 == 0 or (step+1) == FLAGS.max_steps:
			ckpt_path = os.path.join(FLAGS.ckpt_path, 'model.ckpt')
			saver.save(sess, ckpt_path, global_step=step)
		

def main(argv=None):
	train()


if __name__=='__main__':
	tf.app.run()




