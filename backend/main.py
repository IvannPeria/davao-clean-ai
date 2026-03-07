from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

# This is critical: It tells your browser to allow the Frontend (port 3000)
# to talk to your Backend (port 8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model (ensure this path is correct relative to main.py)
model = tf.keras.models.load_model('garbage_model.h5')

# Define your categories in the order your model expects them
LABELS = ["Cardboard", "Glass", "Metal", "Paper", "Plastic", "Trash"]


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Read the image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # 2. Preprocess: Resize to match the model (224x224 is standard for MobileNet)
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # 3. Predict
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    
    return {
        "label": LABELS[predicted_index],
        "confidence": float(np.max(predictions[0]))
    }