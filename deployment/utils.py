import tensorflow as tf
import efficientnet.tfkeras as efn
import numpy as np

from preprocessing import transform


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
    model.compile(optimizer=opt, loss=loss, metrics=['AUC'])

    return model


def preprocess_image(image, dim):
    image = tf.cast(image, tf.float32) / 255.0
    image = transform(image, img_size=dim)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_saturation(image, 0.7, 1.3)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_brightness(image, 0.1)

    image = tf.reshape(image, [dim, dim, 3])

    return image


def build_dataset(image, dim):
    # Convert to tensor
    image = image.convert("RGB")
    image = np.array(image)
    image = image[:, :, :3]  # PNG have 4 channels: 3 colors + 1 transparency level

    image = tf.image.resize(image, [dim, dim])

    ds = tf.data.Dataset.from_tensors(image)
    ds = ds.repeat()
    ds = ds.map(lambda img: preprocess_image(img, dim=dim))
    ds = ds.batch(1)
    return ds
