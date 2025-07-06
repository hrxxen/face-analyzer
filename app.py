from flask import Flask, request, jsonify
import onnxruntime as ort
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# ONNX 모델 로딩
session = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])

# Teachable Machine 모델 입력 이름 확인 필요
input_name = session.get_inputs()[0].name
classes = ['diamond', 'heart', 'oblong', 'oval', 'round', 'square', 'triangle']

def preprocess(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0).astype(np.float32)
    return image

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    image_bytes = request.files["image"].read()
    input_tensor = preprocess(image_bytes)
    
    outputs = session.run(None, {input_name: input_tensor})
    prediction = outputs[0]
    label = classes[np.argmax(prediction)]

    return jsonify({"face_shape": label})

@app.route("/")
def home():
    return "✅ Face Shape Classifier (ONNX) is running!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
