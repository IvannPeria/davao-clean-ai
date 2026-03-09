from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "https://davao-clean-ai.vercel.app/"
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Use the TFLite Interpreter (Much lighter than load_model)
# Ensure garbage_model.tflite is in the same folder as main.py
interpreter = tf.lite.Interpreter(model_path="garbage_model.tflite")
interpreter.allocate_tensors()

# Get input and output details for the model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

LABELS = ["Cardboard", "Glass", "Metal", "Paper", "Plastic", "Trash"]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Read the image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # 2. Preprocess: Resize to 224x224 and normalize
    image = image.resize((224, 224))
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # 3. Predict using the TFLite Interpreter
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    
    # Get the results from the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_index = np.argmax(output_data[0])
    
    return {
        "label": LABELS[predicted_index],
        "confidence": float(np.max(output_data[0]))
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)