import tensorflow as tf
import efficientnet.tfkeras as efn

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
