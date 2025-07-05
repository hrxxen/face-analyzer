from flask import Flask, request, jsonify
import tensorflowjs as tfjs
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# 모델 로드 (TensorFlow.js)
model = tfjs.converters.load_keras_model("model.json")  # model.json 경로

# 예측 API
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    try:
        image_bytes = request.files['image'].read()
        img = Image.open(io.BytesIO(image_bytes))
        img = img.resize((224, 224))  # 모델의 입력 크기에 맞게 조정
        img = np.array(img) / 255.0  # 정규화
        img = np.expand_dims(img, axis=0)  # 배치 차원 추가

        # 예측
        predictions = model.predict(img)
        predicted_label = np.argmax(predictions)  # 예측된 레이블
        
        classes = ['diamond', 'heart', 'oblong', 'oval', 'round', 'square', 'triangle']
        predicted_class = classes[predicted_label]
        
        return jsonify({'face_shape': predicted_class})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 상태 확인
@app.route('/')
def home():
    return "Face shape classification server is running!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
