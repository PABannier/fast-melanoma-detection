import numpy as np
import tensorflow as tf
from imgaug import augmenters as iaa
import imgaug as ia


tf.random.set_seed(42)
ia.seed(42)
np.random.seed(42)


def get_rand_aug(mean=12.):
    """Generate a RandAugment object with a random mean for extra stochasticity.
    
    Parameters
    ----------
    mean : float
        Mean of Poisson distribution.
    
    Returns
    -------
    RandAugment : iaa.RandAugment
        RandAugment object.
    """
    m = np.random.poisson(mean)
    return iaa.RandAugment(n=3, m=m)


def augment(images):
    """Generate random data augmentation from a set of images.
    
    Parameters
    ----------
    images : tf.Tensor
        The images.
    
    Returns
    -------
    augmented_images : np.array
        The augmented images.
    
    Notes 
    -----
    RandAugment does not accept tf.Tensor, hence the explicit casting to numpy arrays.
    """
    images = tf.cast(images, tf.uint8)
    return get_rand_aug()(images=images.numpy())
