import streamlit as st
import tensorflow as tf
# from utils import preprocess_image


# Main presentation
st.title("Melanoma classification")
st.header("Take a picture of your mole and have it diagnosed in seconds!")

@st.cache
def make_prediction(image, model):
    """Predicts image by using model. Note that this function is cached to
    prevent Streamlit to constantly re-predict the image at every user iteraction
    with the app.

    Parameters
    ----------
    image :
        Original image to predict

    model :
        Tensorflow model

    Returns
    -------
    pred_class : str
        The predicted class

    pred_score : float
        The confidence score of the prediction
    """
    # image = preprocess_image(image)
    # For memory purposes, cast to int16
    image = tf.cast(tf.expand_dims(image, axis=0), tf.int16)
    preds = None
    pred_class = None
    pred_conf = None
    return pred_class, pred_conf


choose_model = st.sidebar.selectbox(
    "Pick a model for inference",
    ("EfficientNet-B3",
     "EfficientNet-B5",
     "ResNet-50")
)

st.sidebar.markdown(
    "Warning: This machine learning model has not gone through an extensive \
     performance review process, and is not intended to be used in \
     production. Any use of this model as well as subsequent diagnosis shall \
     be taken with extreme precaution. The author declines any responsibility in \
     case of any erroneous diagnosis and potential detrimental consequences to \
     a patient health. Patients should always consult a qualified practitioner.")


models = {
    "effnet_b3": "Hello",
    "effnet_b5": "World",
    "se-resnext-50": "barbaz"
}


# Model choice logic
if choose_model == "EfficientNet-B3":
    MODEL = models["effnet_b3"]
elif choose_model == "EfficientNet-B5":
    MODEL = models["effnet_b5"]
else:
    MODEL = models["resnet-50"]


# File uploader allows user to add their own image
uploaded_file = st.file_uploader(label="Upload the picture of your mole",
                                 type=["png", "jpeg", "jpg"])


# Setup session state to remember state of app so refresh isn't always needed
st.session_state["pred_button"] = False


# Create app logic
if not uploaded_file:
    st.warning("Please upload an image.")
    st.stop()
else:
    st.session_state["uploaded_image"] = uploaded_file.read()
    st.image(st.session_state["uploaded_image"], use_column_width=True)
    pred_button = st.button("Predict")

if pred_button:
    st.session_state["pred_button"] = True

# If the user has pressed the button, we display the image and predict the mole
if st.session_state["pred_button"]:
    st.session_state["image"], st.session_state["pred_class"], \
    st.session_state["pred_conf"] = make_prediction(
        st.session_state["uploaded_image"], model=MODEL)
    st.write("Prediction: %s, Confidence: %.2f" % \
        (st.session_state["pred_class"], st.session_state["pred_conf"]))
