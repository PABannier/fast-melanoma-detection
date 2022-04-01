import streamlit as st
import numpy as np

from PIL import Image

from utils import build_model, build_dataset
from gradcam import grad_cam, get_superimposed_visualization
from dataset import classes
from html_markdown import (app_off, app_off2, model_predicting, image_uploaded_success,
                           class0, class1, class0_side, class1_side)


DIM = 384
TTA = 50

banner_path = "payload/Banner.png"

st.set_option("deprecation.showfileUploaderEncoding", False)

st.title("Melanoma classification Web App")

# Main presentation
st.write("The app classifies benign and malignant tumors. The model was trained with \
          the [SIIM-ISIC Melanoma 2020 dataset on Kaggle] \
          (https://www.kaggle.com/c/siim-isic-melanoma-classification). ")
st.markdown('***')

# Banner
st.sidebar.image(banner_path, use_column_width=True)


def make_prediction(image):
    """Predicts image by using model. Note that this function is cached to
    prevent Streamlit to constantly re-predict the image at every user iteraction
    with the app.

    Parameters
    ----------
    image : PIL Image
        Image to predict.

    Returns
    -------
    pred_class : str
        The predicted class.

    pred_score : float
        The confidence score of the prediction.
    """
    model = build_model(dim=DIM, ef=3)
    # model.load_weights("models/B3-512.h5")

    st.markdown("***")
    st.markdown(model_predicting, unsafe_allow_html=True)
    st.sidebar.markdown(image_uploaded_success, unsafe_allow_html=True)
    st.sidebar.image(image, width=301, channels="RGB")

    inference_ds = build_dataset(image, dim=DIM)
    preds = model.predict(inference_ds, steps=TTA)

    proba = np.mean(preds)

    if proba > 0.5:
        st.markdown(class0, unsafe_allow_html=True)
        st.sidebar.markdown(class0_side, unsafe_allow_html=True)
        st.write("The predicted class is: **Malignant**")
    else:
        st.markdown(class1, unsafe_allow_html=True)
        st.sidebar.markdown(class1_side, unsafe_allow_html=True)
        st.write("The predicted class is: **Benign**")

    st.sidebar.markdown("**Scroll down to read the full report (class probabilities)**")

    # Display GradCAM visualization
    st.title("GradCAM visualization")
    st.write("GradCAM *(Class Activation Map)* highlights the important regions in" +
             "the image for predicting the class concept. It helps to understand" +
             "if the model based its predictions on the correct regions of the image.")
    heatmap = grad_cam(model, inference_ds, "top_conv")
    out_gradcam = get_superimposed_visualization(image, heatmap)
    st.image(out_gradcam, width=528, channels="RGB")

    # Display the class probabilities table
    st.title('**Class predictions**')
    classes['class probability %'] = [proba, 1 - proba]
    classes['class probability %'] = classes['class probability %'] * 100
    classes_proba = classes.style.background_gradient(cmap='Reds')
    st.write(classes_proba)


# Set the box for the user to upload an image
st.write("**Upload your image**")
uploaded_file = st.file_uploader("Upload your image in JPG or PNG format",
                                 type=["jpg", "png"])


if not uploaded_file:
    st.sidebar.markdown(app_off, unsafe_allow_html=True)
    st.sidebar.markdown(app_off2, unsafe_allow_html=True)
else:
    image = Image.open(uploaded_file)
    make_prediction(image)
    del image
