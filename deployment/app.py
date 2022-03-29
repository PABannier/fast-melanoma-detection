import streamlit as st
import tensorflow as tf

from PIL import Image

# from utils import preprocess_image
from utils import build_model, preprocess_image

models = {
    "effnet_b3": "./models/B3-512.h5",
    "effnet_b5": "./models/B5-512.h5"
}


# Main presentation
st.title("Melanoma classification")
st.header("Take a picture of your mole and have it diagnosed in seconds!")

@st.cache
def make_prediction(image, model_key):
    """Predicts image by using model. Note that this function is cached to
    prevent Streamlit to constantly re-predict the image at every user iteraction
    with the app.

    Parameters
    ----------
    image :
        Image to predict

    model :
        Tensorflow model

    Returns
    -------
    pred_class : str
        The predicted class

    pred_score : float
        The confidence score of the prediction
    """
    model_path = models[model_key]
    model = build_model(dim=512, ef=int(model_key[-1]))
    model.load_weights(model_path)

    image = preprocess_image(image)

    preds = model(image)
    pred_conf = float(preds[0][0])
    pred_class = "Malignant" if pred_conf > 0.5 else "Benign"
    return pred_class, pred_conf


choose_model = st.sidebar.selectbox(
    "Pick a model for inference",
    ("EfficientNet-B3",
     "EfficientNet-B5",)
)

st.sidebar.markdown(
    "Warning: This machine learning model has not gone through an extensive \
     performance review process, and is not intended to be used in \
     production. Any use of this model as well as subsequent diagnosis shall \
     be taken with extreme precaution. The author declines any responsibility in \
     case of any erroneous diagnosis and potential detrimental consequences to \
     a patient health. Patients should always consult a qualified practitioner.")


# Model choice logic
if choose_model == "EfficientNet-B3":
    MODEL = "effnet_b3"
else:
    MODEL = "effnet_b5"


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
    st.session_state["image_to_predict"] = Image.open(uploaded_file)
    st.image(st.session_state["image_to_predict"], use_column_width=True)
    pred_button = st.button("Predict")

if pred_button:
    st.session_state["pred_button"] = True

# If the user has pressed the button, we display the image and predict the mole
if st.session_state["pred_button"]:
    st.session_state["pred_class"], st.session_state["pred_conf"] = make_prediction(
        st.session_state["image_to_predict"], MODEL)
    st.write("Prediction: %s, Confidence: %.2f" % \
        (st.session_state["pred_class"], st.session_state["pred_conf"]))
