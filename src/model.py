import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras import Model
from tensorflow.keras.applications import EfficientNetB6
from tensorflow.keras import backend as K
from tensorflow_addons.optimizers import AdamW


def build_model(dim=128, init_lr=3e-4, min_lr=1e-8, num_epochs=40, num_classes=3,
                dropout=0.2, weight_decay=5e-4, num_snapshots=5,
                multi_sample_dropout=True):
    """Build model.

    Parameters
    ----------
    dim : int
        Image size.

    init_lr : float
        Initial learning rate.

    min_lr : float
        Minimum learning rate in the scheduler.

    num_epochs : int
        Number of epochs.

    num_classes : int
        Number of classes (in the output of softmax layer).

    dropout : float
        Amount of dropout.

    weight_decay : float
        Weight decay used in AdamW.

    num_snapshots : int
        Number of warm starts for the scheduler.

    multi_sample_dropout : bool
        Enable multi sample dropout.

    Returns
    -------
    model : tf.keras.Model
        Compiled model.
    """
    backbone = EfficientNetB6(
        include_top=False, weights="imagenet", input_shape=(dim, dim, 3))
    dropout_layer = Dropout(dropout)
    fc = Dense(num_classes, activation="softmax")

    x = GlobalAveragePooling2D()(backbone.output)
    if multi_sample_dropout:
        x = K.mean(
            K.stack(
                [fc(dropout_layer(x)) for _ in range(5)],
                axis=0
            ),
            axis=0
        )
    else:
        x = fc(dropout_layer(x))

    model = Model(inputs=backbone.input, outputs=x)

    scheduler = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=init_lr,
        first_decay_steps=num_epochs / num_snapshots,
        alpha=min_lr
    )
    optimizer = AdamW(weight_decay, learning_rate=scheduler)

    loss = tf.keras.losses.CategoricalCrossentropy()
    model.compile(optimizer=optimizer, loss=loss, metrics=["AUC"])
    return model
