"""Simple example on how to log scalars and images to tensorboard without tensor ops.

License: BSD License 2.0
"""
__author__ = "Michael Gygli" # Modified version!

# Original code: https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514

import tensorflow as tf
from io import StringIO
import matplotlib.pyplot as plt
import numpy as np

class EpochMean:

    def __init__(self):

        self.dict_scalars = {}

class Logger(object):
    """Logging in tensorboard without tensorflow ops."""

    def __init__(self, NAME, D_PATH, only_per_episode=False):
        """Creates a summary writer logging to log_dir."""

        self.NAME             = NAME
        self.D_PATH           = D_PATH
        self.only_per_episode = only_per_episode
        log_dir               = D_PATH+'agent-logs/'+NAME
        
        self.dict_scalars = {}

        try:
            self.writer = tf.summary.FileWriter(log_dir)
        except Exception as e:
            print(e)
            self.writer = tf.summary.create_file_writer(log_dir)

    def add_to_dict(self, tag, value):

        if tag in self.dict_scalars:
            self.dict_scalars[tag] = np.append(self.dict_scalars[tag], value)
        else:
            self.dict_scalars[tag] = [value]


    def get_from_dict(self, tag):
        value = np.mean(self.dict_scalars[tag])
        self.dict_scalars[tag] = []
        return value


    def log_scalar(self, tag, value, step, done):
        """Log a scalar variable.

        Parameter
        ----------
        tag : basestring
            Name of the scalar
        value
        step : int
            training iteration
        """
        if self.only_per_episode == True:
            self.add_to_dict(tag, value)
            if done == True:
                value = self.get_from_dict(tag)

        if self.only_per_episode == False or done == True:
            try:
                summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
                self.writer.add_summary(summary, step)
            except:
                summary = tf.summary.scalar(value=[tf.Summary.Value(tag=tag, simple_value=value)])
                self.writer.add_summary(summary, step)

    def log_images(self, tag, images, step):
        """Logs a list of images."""

        im_summaries = []
        for nr, img in enumerate(images):
            # Write the image to a string
            s = StringIO()
            plt.imsave(s, img, format='png')

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            im_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, nr),
                                                 image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=im_summaries)
        self.writer.add_summary(summary, step)
        

    def log_histogram(self, tag, values, step, bins=1000):
        """Logs the histogram of a list/vector of values."""
        # Convert to a numpy array
        values = np.array(values)
        
        # Create histogram using numpy        
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill fields of histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
        # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
        # Thus, we drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()
