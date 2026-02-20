import os
import json
import numpy as np
import cv2
import base64
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)

# Load the model and class indices
MODEL_PATH = "blood_cell.h5"
model = load_model(MODEL_PATH)

with open("class_indices.json", "r") as f:
    indices = json.load(f)
# Ensure labels match the model's output indices
class_labels = [k for k, v in sorted(indices.items(), key=lambda item: item[1])]

def predict_image(image_path, model):
    # Read and prepare image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    
    # Preprocess (Same as training)
    img_preprocessed = preprocess_input(img_resized.reshape((1, 224, 224, 3)))
    
    # Predict
    predictions = model.predict(img_preprocessed)
    predicted_idx = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions) * 100
    
    return class_labels[predicted_idx], confidence, img_rgb

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename != "":
            # Ensure folder exists
            if not os.path.exists("static"): os.makedirs("static")
            
            file_path = os.path.join("static", file.filename)
            file.save(file_path)
            
            label, score, img_rgb = predict_image(file_path, model)

            # Convert image to base64 for display
            _, img_encoded = cv2.imencode('.png', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            img_str = base64.b64encode(img_encoded).decode('utf-8')

            # Pass variables to HTML
            return render_template("result.html", 
                                   label=label, 
                                   score_value=round(score, 2), 
                                   img_data=img_str)
                                   
    return render_template("home.html")

if __name__ == "__main__":
    app.run(debug=True)