from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import onnxruntime as ort
import io

app = FastAPI()

# 모델 세션 전역 변수로 선언
session = None
input_name = None
output_name = None

@app.on_event("startup")
def load_model():
    global session, input_name, output_name
    session = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print("✅ ONNX 모델 로딩 완료")

@app.get("/")
def read_root():
    return {"message": "ONNX 서버가 실행 중입니다."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 이미지 불러오기
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    
    # 이미지 전처리 (Teachable Machine 기준 224x224)
    image = image.resize((224, 224))
    img_array = np.asarray(image).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)

    # ONNX 모델 추론
    result = session.run([output_name], {input_name: img_array})[0]
    predicted_class = int(np.argmax(result))

    return JSONResponse(content={
        "prediction": predicted_class,
        "raw_output": result[0].tolist()
    })
