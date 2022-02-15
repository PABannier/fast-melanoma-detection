import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB1


def build_model(dim=128):
    """Builds and compiles a CNN model with an Adam optimizer"""
    input = tf.keras.layers.Input(shape=(dim, dim, 3))
    feature_extractor = EfficientNetB1(include_top=False, weights="imagenet",
                                       input_shape=(dim, dim, 3))
    x = feature_extractor(input)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs=input, outputs=x)

    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05)

    model.compile(optimizer=opt, loss=loss, metrics=["AUC"])
    return model
