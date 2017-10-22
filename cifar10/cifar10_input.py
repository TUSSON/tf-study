import numpy as np
import tensorflow as tf
from tensorflow.contrib.keras.api.keras.datasets.cifar10 import load_data
import cifar10

train_data, test_data = load_data()

train_images = train_data[0]
train_labels = train_data[1]
test_images  = test_data[0]
test_labels  = test_data[1]

class DataSet(object):
    def __init__(self, images, labels, batch_size, is_train_data):
        self._num_examples = images.shape[0]
        self._images = tf.constant(images)
        self._is_train_data = is_train_data
        self._labels = tf.constant(labels)
        self._index_in_epoch = 0
        self._batch_size = batch_size
        self._distorted_data()

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    def _generate_image_and_label_batch(self, image, label):
        """Construct a queued batch of images and labels.

        Args:
            image: 3-D Tensor of [IMAGE_SIZE, IMAGE_SIZE, 3] of type.float32.
            label: 1-D Tensor of type.int32
            in the queue that provides of batches of examples.

        Returns:
            images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
            labels: Labels. 1D tensor of [batch_size] size.
        """

        min_queue_examples = int(self.num_examples * 0.4)
        # Create a queue that shuffles the examples, and then
        # read 'batch_size' images + labels from the example queue.
        num_preprocess_threads = 16
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=self._batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * self._batch_size,
            min_after_dequeue=min_queue_examples)

        # Display the training images in the visualizer.
        tf.summary.image('images', images)
        self._images = images
        self._labels = tf.reshape(label_batch, [self._batch_size])

    def _random_crop(self, image, size):
        if len(image.shape):
            W, H, D = image.shape
            w, h, d = size
        else:
            W, H = image.shape
            w, h = size
        left, top = np.random.randint(W - w + 1), np.random.randint(H - h + 1)
        return tf.image.crop_to_bounding_box(image, top, left, h, w)

    def _distorted_data(self):
        image, label = tf.train.slice_input_producer([self._images, self._labels])

        image = tf.cast(image, tf.float32)
        w = h = cifar10.IMAGE_SIZE
        if self._is_train_data:
            image = self._random_crop(image, (h, w, 3))
            image = tf.image.random_flip_left_right(image)
            #image = tf.image.random_brightness(image, max_delta=63)
            #image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
        else:
            image = tf.image.resize_image_with_crop_or_pad(image, h, w)
        image = tf.image.per_image_standardization(image)
        self._generate_image_and_label_batch(image, label)

