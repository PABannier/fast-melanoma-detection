import numpy as np
import matplotlib.cm as cm
import tensorflow as tf
from tensorflow import keras
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tk_keras_vis.saliency import Saliency
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear



def _score_function(logit):
    """Transforms a proba logit into a tuple of proba (positive, negative)"""
    return (logit, 1 - logit)


def grad_cam(model, image):
    """Generate a GradCAM++ visualization for model interpretability.

    Parameters
    ----------
    model : tf.keras.models.Model
        The CNN model.

    image : tf.Dataset, (1, img_size, img_size, 3)
        Dataset containing the image.

    Returns
    -------
    heatmap : array, (img_size, img_size, 3)
        The class activation map.
    """
    gradcam = GradcamPlusPlus(model, model_modifier=ReplaceToLinear(), clone=True)
    cam = gradcam(_score_function, next(iter(image)), penultimate_layer=-1)
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(cm.jet(cam)[..., :3] * 255)
    return heatmap


def smooth_grad(model, image):
    """Generate a SmoothGrad visualization for model interpretability.

    Parameters
    ----------
    model : tf.keras.models.Model
        The CNN model.

    image : tf.Dataset, (1, img_size, img_size, 3)
        Dataset containing the image.

    Returns
    -------
    saliency_map : array, (img_size, img_size, 3)
        The class activation map.

    Notes
    -----
    smooth_samples controls the number of calculating gradient iterations. It has a
    critical impact on the saliency map generation time (up to 2-3 minutes on CPU).

    smooth_noise is the noise spread level.
    """
    saliency = Saliency(model, model_modifier=ReplaceToLinear(), clone=True)
    saliency_map = saliency(_score_function, next(iter(image)),
                            smooth_samples=20, smooth_noise=0.2)
    return saliency_map



def get_superimposed_visualization(original_image, heatmap, alpha=0.4):
    # Change to Numpy array
    original_image = original_image.convert("RGB")
    original_image = np.array(original_image)
    # PNG have 4 channels: 3 colors + 1 transparency level
    original_image = original_image[:, :, :3]

    # Create an image with RGB colorized heatmap
    heatmap = keras.preprocessing.image.array_to_img(heatmap)
    heatmap = heatmap.resize((original_image.shape[1], original_image.shape[0]))
    heatmap = keras.preprocessing.image.img_to_array(heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = heatmap * alpha + original_image
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    return superimposed_img
