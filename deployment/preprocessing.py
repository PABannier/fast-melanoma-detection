import math
import tensorflow as tf
import tensorflow.keras.backend as K


def get_mat(rotation, shear, height_zoom, width_zoom, height_shift,
            width_shift):
    """Returns 3x3 transformation matrix for indices. These matrix are used to
    rotate, shear, zoom and shift images.

    To combine those transformations into one single transformation, these
    matrices are multiplied to output one single transformation matrix.

    Rotation matrix
    ---------------
        | cos(θ)  sin(θ)  0 |
        | -sin(θ) cos(θ)  0 |
        | 0       0       1 |

    Shear matrix
    ------------
        | 1       sin(θ)  0 |
        | 0       cos(θ)  0 |
        | 0       0       1 |

    Zoom matrix
    -----------
        | 1 / hz  0       0 |
        | 0       1 / wz  0 |
        | 0       0       1 |

    Shift matrix
    ------------
        | 1       0      hs |
        | 0       1      ws |
        | 0       0      1  |

    Parameters
    ----------
    rotation : tf.float32
        Rotation angle
    shear : tf.float32
        Shear magnitude
    height_zoom : tf.float32
        Zoom amplitude on height
    width_zoom : tf.float32
        Zoom amplitude on width
    height_shift : tf.float32
        Magnitude of height shifting
    width_shift : tf.float32
        Magnitude of width shifting

    Returns
    -------
    tf.Tensor of shape (3, 3)
        Combined transformation matrix
    """

    # Convert degrees to radians
    rotation = math.pi * rotation / 180.
    shear = math.pi * shear / 180.

    def get_3x3_mat(lst):
        return tf.reshape(tf.concat([lst], axis=0), [3, 3])

    # Rotation
    c1 = tf.math.cos(rotation)
    s1 = tf.math.sin(rotation)
    one = tf.constant([1], dtype='float32')
    zero = tf.constant([0], dtype='float32')

    rotation_matrix = get_3x3_mat([c1,   s1,   zero,
                                   -s1,  c1,   zero,
                                   zero, zero, one])
    # Shear
    c2 = tf.math.cos(shear)
    s2 = tf.math.sin(shear)

    shear_matrix = get_3x3_mat([one,  s2,   zero,
                                zero, c2,   zero,
                                zero, zero, one])
    # Zoom
    zoom_matrix = get_3x3_mat([one / height_zoom, zero,             zero,
                               zero,              one / width_zoom, zero,
                               zero,              zero,             one])
    # Shift
    shift_matrix = get_3x3_mat([one,  zero, height_shift,
                                zero, one,  width_shift,
                                zero, zero, one])

    return K.dot(K.dot(rotation_matrix, shear_matrix),
                 K.dot(zoom_matrix,     shift_matrix))


def transform(image, img_size=256, rotation=180., shear=2., height_zoom=8.,
              width_zoom=8., height_shift=8., width_shift=8.):
    """Creates a transformed version of an image to avoid overfitting and
    implement data augmentation.

    Parameters
    ----------
    image : tf.Tensor of shape (img_size, img_size, 3)
        The image tensor
    img_size : float
        The image dimension
    rotation : float
        Rotation angle
    shear : float
        Shear magnitude
    height_zoom : float
        Zoom amplitude on height
    width_zoom : float
        Zoom amplitude on width
    height_shift : float
        Magnitude of height shifting
    width_shift : float
        Magnitude of width shifting

    Returns
    -------
    tf.Tensor of shape (img_size, img_size, 3)
        A randomly rotated, sheared, zoomed and shifted image

    """
    aux_img_size = img_size % 2

    # Random sampling of augmentation hyperparameters
    rot = rotation * tf.random.normal([1], dtype='float32')
    shr = shear * tf.random.normal([1], dtype='float32')
    h_zoom = 1.0 + tf.random.normal([1], dtype='float32') / height_zoom
    w_zoom = 1.0 + tf.random.normal([1], dtype='float32') / width_zoom
    h_shift = height_shift * tf.random.normal([1], dtype='float32')
    w_shift = width_shift * tf.random.normal([1], dtype='float32')

    transformation_matrix = get_mat(rot, shr, h_zoom, w_zoom, h_shift, w_shift)

    # LIST DESTINATION PIXEL INDICES
    x   = tf.repeat(tf.range(img_size//2, -img_size//2,-1), img_size)
    y   = tf.tile(tf.range(-img_size//2, img_size//2), [img_size])
    z   = tf.ones([img_size*img_size], dtype='int32')
    idx = tf.stack( [x,y,z] )

    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS
    idx2 = K.dot(transformation_matrix, tf.cast(idx, dtype='float32'))
    idx2 = K.cast(idx2, dtype='int32')
    idx2 = K.clip(idx2, -img_size//2+aux_img_size+1, img_size//2)

    # FIND ORIGIN PIXEL VALUES
    idx3 = tf.stack([img_size//2-idx2[0,], img_size//2-1+idx2[1,]])
    out  = tf.gather_nd(image, tf.transpose(idx3))

    return tf.reshape(out, [img_size, img_size, 3])
