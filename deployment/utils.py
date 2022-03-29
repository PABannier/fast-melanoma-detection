import tensorflow as tf
import efficientnet.tfkeras as efn
import numpy as np


effnet_models = [
    efn.EfficientNetB0, efn.EfficientNetB1, efn.EfficientNetB2, 
    efn.EfficientNetB3, efn.EfficientNetB4, efn.EfficientNetB5, 
    efn.EfficientNetB6, efn.EfficientNetB7
]


def build_model(dim=128, ef=0):
    inp = tf.keras.layers.Input(shape=(dim, dim, 3))
    base = effnet_models[ef](
        input_shape=(dim, dim, 3), weights='imagenet', include_top=False)
    x = base(inp)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inp, outputs=x)

    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05) 
    model.compile(optimizer=opt,loss=loss,metrics=['AUC'])

    return model


def preprocess_image(image):
    # Crop image
    width, height = image.size
    left = (width - 512) / 2
    top = (height - 512) / 2
    right = (width + 512) / 2
    bottom = (height + 512) / 2
    image = image.crop((left, top, right, bottom))

    # Convert image
    image = image.convert("RGB")
    image = np.asarray(image, dtype=np.float32) / 255
    image = image[:, :, :3]
    return tf.cast(tf.expand_dims(image, axis=0), tf.int16)

