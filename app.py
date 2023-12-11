import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import streamlit as st

model = load_model("model.keras")

class_names = [
    "apple",
    "banana",
    "beetroot",
    "bell pepper",
    "cabbage",
    "capsicum",
    "carrot",
    "cauliflower",
    "chilli pepper",
    "corn",
    "cucumber",
    "eggplant",
    "garlic",
    "ginger",
    "grapes",
    "jalepeno",
    "kiwi",
    "lemon",
    "lettuce",
    "mango",
    "onion",
    "orange",
    "paprika",
    "pear",
    "peas",
    "pineapple",
    "pomegranate",
    "potato",
    "raddish",
    "soy beans",
    "spinach",
    "sweetcorn",
    "sweetpotato",
    "tomato",
    "turnip",
    "watermelon",
]

image_path = "apple.jpg"

img_height, img_width = 180, 180

img_load = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width))
img_array = tf.keras.utils.img_to_array(img_load)
img_array = tf.expand_dims(img_array, 0)  # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

st.image(img_load, caption="Uploaded Image.", use_column_width=True)

st.write(
    "This image most likely belongs to {} with a {:.2f} percent confidence.".format(
        class_names[np.argmax(score)], 100 * np.max(score)
    )
)
