import streamlit as st
import tensorflow as tf
import pandas as pd
import altair as alt
import requests  # pip install requests
from streamlit_lottie import st_lottie  # pip install streamlit-lottie
from utils import load_and_prep, get_classes

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

@st.cache(suppress_st_warning=True)
def predicting(image, model):
    image = load_and_prep(image)
    image = tf.cast(tf.expand_dims(image, axis=0), tf.int16)
    preds = model.predict(image)
    pred_class = class_names[tf.argmax(preds[0])]
    pred_conf = tf.reduce_max(preds[0])
    top_5_i = sorted((preds.argsort())[0][-5:][::-1])
    values = preds[0][top_5_i] * 100
    labels = []
    for x in range(5):
        labels.append(class_names[top_5_i[x]])
    df = pd.DataFrame({"Top 5 Predictions": labels,
                       "F1 Scores": values,
                       'color': ['#EC5953', '#EC5953', '#EC5953', '#EC5953', '#EC5953']})
    df = df.sort_values('F1 Scores')
    return pred_class, pred_conf, df

st.set_page_config(page_title="Food Lens",
                   page_icon="🌯")
lottie_hello = load_lottieurl("https://lottie.host/2a17b956-8600-4744-bbf5-8a3ee5f8ac31/GzC608FMEZ.json")
lottie_food = load_lottieurl("https://lottie.host/3d2371fc-47db-4e3f-ba91-bcde31e012c6/YlergFxi3G.json")
lottie_predict = load_lottieurl("https://lottie.host/e11e083a-c060-4541-9849-eae84a5736e5/euDSJ9fBlW.json")

class_names = get_classes()


#### SideBar ####

st.sidebar.title("What's Food Lens?")
st.sidebar.write("""
Foodlens is an advanced end-to-end **CNN Image Classification project** that comes with Keras and also utilizes our own build **FOOD LENS** model inspired from EfficientNetB1 model to accurately identify different types of food items from images.

The model has been trained on the Food101 dataset. It can identify over 100 different food classes

The project also includes a user-friendly web application with some good front-end built with **Streamlit**, allowing users to upload food images and receive instant predictions along with the **top-5 predictions** from the model.

**Accuracy :** **`85%`**

**Model :** **`FOOD LENS`**

**Dataset :** **`Food101`**
""")

st.sidebar.markdown("Created by **`ML GROUP 4, JNU`**")

#### Main Body ####

col1, col2 = st.columns(2)

with col1:
    st.title("Food Lens 🌯🎥")
    st.header("Identify & Discover what's in your food photos!")  

with col2:
    st_lottie(lottie_food, key = "food")

file = st.file_uploader(label="Upload a Pic of Your Delicious food Now.",
                        type=["jpg", "jpeg", "png"])


model = tf.keras.models.load_model("../models/Foodlens_model.hdf5")

if not file:
    st.warning("Please upload an image")
    col3, col4 = st.columns(2)
    with col3:
        st.write("**Don't know what's in your food photos!??**")
        st_lottie(lottie_predict, key = "Predictimage")
    with col4:
        st.write("**Upload the photo and Predict it Right Now!!**")
        st_lottie(lottie_hello, key = "hello")
    st.stop()

else:
    image = file.read()
    st.image(image, use_column_width=True)
    pred_button = st.button("Predict")

if pred_button:
    pred_class, pred_conf, df = predicting(image, model)
    st.success(f'Prediction : {pred_class} \nConfidence : {pred_conf*100:.2f}%')
    st.write(alt.Chart(df).mark_bar().encode(
        x='F1 Scores',
        y=alt.X('Top 5 Predictions', sort=None),
        color=alt.Color("color", scale=None),
        text='F1 Scores'
    ).properties(width=600, height=400))