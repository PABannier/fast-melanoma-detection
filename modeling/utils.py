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
