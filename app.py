from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

model = tf.keras.models.load_model("tm-my-image-model")

classes = ['diamond', 'heart', 'oblong', 'oval', 'round', 'square', 'triangle']  # 학습한 클래스 이름과 일치하게

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize((224, 224))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

@app.route('/')
def home():
    return '🧠 Face shape classification server is running!'

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    try:
        image_bytes = request.files['image'].read()
        input_image = preprocess_image(image_bytes)
        predictions = model.predict(input_image)[0]
        predicted_class = classes[np.argmax(predictions)]
        return jsonify({'face_shape': predicted_class})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
