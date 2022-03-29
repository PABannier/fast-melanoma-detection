import streamlit as st
import tensorflow as tf
import numpy as np

from PIL import Image

# from utils import preprocess_image
from utils import build_model, build_dataset
from dataset import classes
from html_markdown import (app_off, app_off2, model_predicting, loading_bar, 
                           result_pred, image_uploaded_success, more_options,
                           class0, class1, class0_side, class1_side, s_load_bar,
                           unknown, unknown_side, unknown_w, unknown_msg)


DIM = 384
TTA = 50

banner_path = "payload/Banner.png"

st.title("Melanoma classification Web App")

# Main presentation
st.write("The app classifies benign and malignant tumors. The model was trained with the [SIIM-ISIC Melanoma 2020 dataset on Kaggle](https://www.kaggle.com/c/siim-isic-melanoma-classification). ")
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
    # model_path = models[model_key]
    model = build_model(dim=DIM, ef=3)
    model.load_weights("models/B3-512.h5")

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
    
    #Display the class probabilities table
    st.title('**Class predictions:**') 
    classes['class probability %'] = [proba, 1 - proba]
    classes['class probability %'] = classes['class probability %'] * 100
    classes_proba = classes.style.background_gradient(cmap='Reds')
    st.write(classes_proba)


#Set the box for the user to upload an image
st.write("**Upload your image**")
uploaded_file = st.file_uploader("Upload your image in JPG or PNG format", type=["jpg", "png"])


if not uploaded_file:
    st.sidebar.markdown(app_off, unsafe_allow_html=True)
    st.sidebar.markdown(app_off2, unsafe_allow_html=True)
else:
    image = Image.open(uploaded_file)
    make_prediction(image)
    del image


# # Create app logic
# if not uploaded_file:
#     st.warning("Please upload an image.")
#     st.stop()
# else:
#     st.session_state["image_to_predict"] = Image.open(uploaded_file)
#     st.image(st.session_state["image_to_predict"], use_column_width=True)
#     pred_button = st.button("Predict")

# if pred_button:
#     st.session_state["pred_button"] = True

# # If the user has pressed the button, we display the image and predict the mole
# if st.session_state["pred_button"]:
#     st.session_state["pred_class"], st.session_state["pred_conf"] = make_prediction(
#         st.session_state["image_to_predict"])
#     st.write("Prediction: %s, Confidence: %.2f" % \
#         (st.session_state["pred_class"], st.session_state["pred_conf"]))
