import tensorflow as tf
from preprocessing import generate_aug_image


def read_tfrecord(example, cfg):
    """Read a TFRecord file instance.

    Parameters
    ----------
    example : TFRecord
        The Tensorflow record.

    Returns
    -------
    image : tf.Tensor
        The image.

    target : int
        The diagnosis.
    """
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
    target = None
    num_classes = 0

    if cfg["data_year"] == 2019:
        # Diagnoses are initially labelled from 9 to 17
        # -9 to be between 0 and 8
        target = example["diagnosis"] - 9
        num_classes = 9
    else:
        target = example["target"]
        num_classes = 3

    target = tf.one_hot(target, num_classes)

    return example['image'], target


def get_dataset(files, cfg, auto, replicas, augment=False, shuffle=False,
                repeat=False, batch_size=16, dim=256):
    """Gets the dataset to train the model on.

    Parameters
    ----------
    files:
        Paths to the TFRecord files.

    augment : bool
        Applies data augmentation.

    shuffle : bool
        Shuffles the dataset.

    repeat : bool
        Repeats the dataset.

    batch_size : int
        Batch size.

    dim : int
        Image size.
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

    ds = ds.map(lambda x: read_tfrecord(x, cfg), num_parallel_calls=auto)
    ds = ds.map(
        lambda img, label: (tf.image.decode_jpeg(img, channels=3), label),
        num_parallel_calls=auto
    )
    ds = ds.map(
        lambda img, label: (tf.image.resize(img, (dim, dim)), label),
        num_parallel_calls=auto
    )

    if augment:
        ds = ds.map(
                lambda img, label: (
                    tf.py_function(
                        func=generate_aug_image,
                        inp=[img],
                        Tout=tf.float32
                    ),
                    label
                ),
                num_parallel_calls=auto
        )

    ds = ds.batch(batch_size * replicas)
    ds = ds.prefetch(auto)
    return ds
