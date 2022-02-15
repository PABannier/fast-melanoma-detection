import tensorflow as tf
from .preprocessing import transform
from .utils import load_device_strategy


def read_tfrecord(example):
    """Reads a TFRecord file instance."""
    tfrec_format = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'image_name': tf.io.FixedLenFeature([], tf.string),
        'patient_id': tf.io.FixedLenFeature([], tf.int64),
        'sex': tf.io.FixedLenFeature([], tf.int64),
        'age_approx': tf.io.FixedLenFeature([], tf.int64),
        'anatom_site_general_challenge': tf.io.FixedLenFeature([], tf.int64),
        'diagnosis': tf.io.FixedLenFeature([], tf.int64),
        'target': tf.io.FixedLenFeature([], tf.int64)
    }
    example = tf.io.parse_single_example(example, tfrec_format)
    return example['image'], example['target']


def prepare_image(img, augment=True, dim=256):
    """Reads the image and optionally applies transformation to it.

    Parameters
    ----------
    img : tf.Tensor of shape (dim, dim, 3)
        The input image tensor
    augment : bool
        Applies data augmentation
    dim : float
        Output dimension of the image

    Returns
    -------
    tf.Tensor of shape (dim, dim, 3)
    """
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32) / 255.0

    if augment:
        img = transform(img, dim)
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_hue(img, 0.01)
        img = tf.image.random_saturation(img, 0.7, 1.3)
        img = tf.image.random_contrast(img, 0.8, 1.2)
        img = tf.image.random_brightness(img, 0.1)

    img = tf.reshape(img, [dim, dim, 3])

    return img


def get_dataset(files, auto, replicas, augment=False, shuffle=False, 
                repeat=False, batch_size=16, dim=256):
    """Gets the dataset to train the model on.

    Parameters
    ----------
    files:
        Paths to the TFRecord files
    augment : bool
        Applies data augmentation
    shuffle : bool
        Shuffles the dataset
    repeat : bool
        Repeats the dataset
    batch_size : int
        Batch size
    dim : int
        Image size
    """
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=auto)
    ds = ds.cache()

    if repeat:
        ds = ds.repeat()

    if shuffle:
        ds = ds.shuffle(1024 * 8)
        opt = tf.data.Options()
        opt.experimental_deterministic = False
        ds = ds.with_options(opt)

    ds = ds.map(read_tfrecord, num_parallel_calls=auto)
    ds = ds.map(
        lambda img, imgname_or_label:
            (prepare_image(img, augment=augment, dim=dim), imgname_or_label),
        num_parallel_calls=auto)

    ds = ds.batch(batch_size * replicas)
    ds = ds.prefetch(auto)
    return ds
