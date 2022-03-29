import numpy as np
import matplotlib.cm as cm
import tensorflow as tf
from tensorflow import keras


def grad_cam(model, image, last_conv_layer_name="top_conv"):
    """Generate a GradCAM visualization for model interpretability.

    Parameters
    ----------
    model : tf.model
        The CNN model.

    image : array, (img_size, img_size, 3)
        The image to predict.

    last_conv_layer_name : str
        The last convolution layer name (needed for GradCAM).

    Returns
    -------
    gc_image : array, (img_size, img_size, 3)
        The image with a GradCAM overlay.
    """
    last_conv_layer = model.get_layer("efficientnet-b3").get_layer(last_conv_layer_name)
    grad_model = tf.keras.models.Model(model.inputs, last_conv_layer.output)

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(image)
        class_channel = preds  # Output neuron (here a sigmoid logit)

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def get_superimposed_visualization(original_image, heatmap, alpha=0.4):
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((original_image.shape[1], original_image.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + original_image
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    return superimposed_img
