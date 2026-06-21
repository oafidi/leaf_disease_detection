import os
import io
import logging
from contextlib import asynccontextmanager

import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobilenet_preprocess, decode_predictions
import uvicorn


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB
MODEL_PATH = os.getenv("MODEL_PATH", "apple_leaf_model_final.keras")
CLASS_NAMES = [
    "apple_frogeye_leaf_spot",
    "apple_leaf_healthy",
    "apple_mosaic_leaf",
    "apple_powdery_mildew_leaf",
    "apple_rust_leaf",
    "apple_scab_leaf"
]
IMG_SIZE = (224, 224)

VALID_KEYWORDS = [
    "leaf", "plant", "flower", "fruit", "tree", "pot", "cabbage",
    "acorn", "fig", "vegetable", "daisy", "rose", "grass", "lemon",
    "strawberry", "orange", "apple", "nematode", "slug", "insect",
    "beetle", "spider", "snail", "ant", "bee", "butterfly", "moth",
    "mantis", "grasshopper", "fly", "bug", "mushroom", "fungus",
    "fern", "vine", "greenhouse", "lacewing", "chameleon"
]

model = None
imagenet_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load the model
    global model, imagenet_model
    logger.info("Loading model...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        
    logger.info("Loading ImageNet MobileNetV2...")
    try:
        imagenet_model = MobileNetV2(weights="imagenet")
        logger.info("ImageNet model loaded successfully!")
    except Exception as e:
        logger.error(f"Error loading ImageNet model: {e}")
    
    yield  # API is now running and ready to accept requests
    
    # Shutdown: Clean up resources if necessary
    logger.info("Shutting down API...")

# Initialize FastAPI with lifespan
app = FastAPI(
    title="Apple Leaf Disease Classification API",
    description="Diagnose apple leaves (Healthy, Rust, Scab).",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def prepare_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32)
    return np.expand_dims(img_array, axis=0)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None
    }

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    # Verify that the model is loaded
    if model is None:
        return JSONResponse(
            status_code=503,
            content={"message": "Model unavailable, please try again in a few seconds."}
        )

    # Verify that the file is an image
    if not file.content_type.startswith("image/"): # type: ignore
        return JSONResponse(
            status_code=400,
            content={"message": "The uploaded file is not a valid image."}
        )

    try:
        contents = await file.read()

        # Check file size
        if len(contents) > MAX_UPLOAD_SIZE:
            return JSONResponse(
                status_code=413,
                content={"message": f"Image too large. Max size: {MAX_UPLOAD_SIZE // (1024*1024)} MB."}
            )

        # Phase 1: OOD Detection (Is it a plant/leaf/insect?)
        if imagenet_model is not None:
            try:
                img_for_imagenet = Image.open(io.BytesIO(contents)).convert("RGB").resize((224, 224))
                img_array_imagenet = np.array(img_for_imagenet, dtype=np.float32)
                img_array_imagenet = np.expand_dims(img_array_imagenet, axis=0)
                img_array_imagenet = mobilenet_preprocess(img_array_imagenet)
                
                preds = imagenet_model.predict(img_array_imagenet)
                top_preds = decode_predictions(preds, top=10)[0]
                
                is_valid = False
                for _, desc, _ in top_preds:
                    desc_lower = desc.lower()
                    if any(kw in desc_lower for kw in VALID_KEYWORDS):
                        is_valid = True
                        break
                        
                if not is_valid:
                    return JSONResponse(
                        status_code=400,
                        content={"message": "L'image ne semble pas être une feuille ou une plante reconnue. Veuillez essayer avec une photo plus claire."}
                    )
            except Exception as e:
                logger.error(f"OOD detection failed: {e}")
                # Fallback to normal prediction if OOD fails

        # Phase 2: Disease classification
        img_array = prepare_image(contents)
        
        # Use model(..., training=False) for faster single-image inference
        predictions = model(img_array, training=False).numpy()

        predicted_index = int(np.argmax(predictions[0]))
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(predictions[0][predicted_index])

        all_probs = {
            CLASS_NAMES[i]: round(float(predictions[0][i]) * 100, 2)
            for i in range(len(CLASS_NAMES))
        }

        return {
            "filename": file.filename,
            "prediction": predicted_class,
            "confidence": f"{round(confidence * 100, 2)}%",
            "all_probabilities": all_probs
        }

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"message": f"Error during prediction: {str(e)}"}
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)