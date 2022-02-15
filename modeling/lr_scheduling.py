import tensorflow as tf


def get_lr_callback(batch_size=8, replicas=8):
    lr_start = 5 * 1e-6
    lr_max = 1.25 * 1e-6 * replicas * batch_size
    lr_min = 1 * 1e-6
    lr_ramp_ep = 5
    lr_sus_ep = 0
    lr_decay = 0.8

    def lrfn(epoch):
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start

        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max

        else:
            exp = epoch - lr_ramp_ep - lr_sus_ep
            lr = (lr_max - lr_min) * lr_decay ** exp + lr_min

        return lr

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)
    return lr_callback
