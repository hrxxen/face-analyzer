from fastapi import FastAPI, File, UploadFile
import onnxruntime as ort
from PIL import Image
import numpy as np
import io

app = FastAPI()

# ✅ 모델 로딩
try:
    session = ort.InferenceSession("model.onnx")  # 모델 파일 경로 확인
    print("✅ ONNX 모델 로딩 완료")
except Exception as e:
    print("❌ 모델 로딩 실패:", e)
    session = None  # 이후 예측 방지

# ✅ 예측 라우터
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if session is None:
        return {"error": "모델 로딩 실패로 예측할 수 없습니다."}

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize((224, 224))  # Teachable Machine에 맞는 크기
        image_data = np.array(image).astype("float32") / 255.0
        image_data = np.expand_dims(image_data, axis=0)

        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: image_data})
        scores = output[0][0]
        predicted_index = int(np.argmax(scores))

        labels = [
            "둥근 얼굴", "계란형 얼굴", "각진 얼굴",
            "하트형 얼굴", "다이아몬드형 얼굴", "삼각형 얼굴", "긴 얼굴"
        ]
        prediction_label = labels[predicted_index]

        print("✅ 예측 성공:", prediction_label)

        return {
            "prediction": predicted_index,
            "prediction_label": prediction_label,
            "raw_output": scores.tolist()
        }

    except Exception as e:
        print("❌ 예측 중 오류:", e)
        return {"error": str(e)}
