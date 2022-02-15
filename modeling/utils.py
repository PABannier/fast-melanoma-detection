import re
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def load_device_strategy(device):
    """Helper function to load data on devices. Supported device: GPU and TPU.
    TPU is particularly fast with TFRecords as the I/O speed of TFRecords avods
    a bottleneck in the pipeline when reading new images.

    device: str, option = "TPU" or "GPU"
        The device type used to train the model on
    """

    if device == "TPU":
        print("connecting to TPU...")
        try:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            print('Running on TPU ', tpu.master())
        except ValueError:
            print("Could not connect to TPU")
            tpu = None

        if tpu:
            try:
                print("initializing  TPU ...")
                tf.config.experimental_connect_to_cluster(tpu)
                tf.tpu.experimental.initialize_tpu_system(tpu)
                strategy = tf.distribute.experimental.TPUStrategy(tpu)
                print("TPU initialized")
            except:  # noqa
                print("Failed to initialize TPU")
        else:
            device = "GPU"

    if device != "TPU":
        print("Using default strategy for CPU and single GPU")
        strategy = tf.distribute.get_strategy()

    if device == "GPU":
        print("Num GPUs Available: ",
              len(tf.config.experimental.list_physical_devices('GPU')))

    return tf.data.experimental.AUTOTUNE, strategy.num_replicas_in_sync


def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1))
         for filename in filenames]
    return np.sum(n)


def make_results_plot(history, epochs, fold, img_size, model_name):
    plt.figure(figsize=(15, 5))

    plt.plot(np.arange(epochs), history.history['auc'], '-o',
             label='Train AUC', color='#ff7f0e')
    plt.plot(np.arange(epochs), history.history['val_auc'], '-o',
             label='Val AUC', color='#1f77b4')

    x = np.argmax(history.history['val_auc'])
    y = np.max(history.history['val_auc'])
    xdist = plt.xlim()[1] - plt.xlim()[0]
    ydist = plt.ylim()[1] - plt.ylim()[0]

    plt.scatter(x, y, s=200, color='#1f77b4')
    plt.text(x - 0.03 * xdist, y - 0.13 * ydist, 'max auc\n%.2f' % y, size=14)

    plt.ylabel('AUC',size=14)
    plt.xlabel('Epoch',size=14)
    plt.legend(loc=2)

    plt2 = plt.gca().twinx()

    plt2.plot(np.arange(epochs), history.history['loss'], '-o',
              label='Train Loss', color='#2ca02c')
    plt2.plot(np.arange(epochs), history.history['val_loss'], '-o',
              label='Val Loss', color='#d62728')

    x = np.argmin( history.history['val_loss'] )
    y = np.min( history.history['val_loss'] )
    ydist = plt.ylim()[1] - plt.ylim()[0]

    plt.scatter(x, y, s=200, color='#d62728')
    plt.text(x - 0.03 * xdist, y + 0.05 * ydist, 'min loss', size=14)

    plt.ylabel('Loss',size=14)

    plt.title('FOLD %i - Image Size %i, %s'% (fold + 1, img_size, model_name),
              size=18)
    plt.legend(loc=3)
    plt.show()
