import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.preprocessing import image

# Load model and class names
model = tf.keras.models.load_model("blood_cell.h5")
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)
class_names = {v: k for k, v in class_indices.items()}

# PATH TO A TEST IMAGE (Change this to a real path on your PC)
img_path = r"C:\Users\durga\OneDrive\Desktop\Hematovision\test_LYMPHOCYTE.jpeg"

def predict(path):
    img = image.load_img(path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    score = predictions[0]
    print(f"Prediction: {class_names[np.argmax(score)]} ({100 * np.max(score):.2f}%)")

predict(img_path)