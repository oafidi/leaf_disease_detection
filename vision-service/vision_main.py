from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import uvicorn
import os

MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB

# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────
app = FastAPI(
    title="API Classification Maladies Pommier",
    description="Diagnostiquer les feuilles de pommier (Healthy, Rust, Scab)."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.getenv("MODEL_PATH", "apple_leaf_model_final.keras")
CLASS_NAMES = ["apple_leaf_healthy", "apple_rust_leaf", "apple_scab_leaf"]
IMG_SIZE = (224, 224)

# ─────────────────────────────────────────────
# 2. CHARGEMENT DU MODÈLE AU DÉMARRAGE
# ─────────────────────────────────────────────
model = None

@app.on_event("startup")
async def load_model():
    global model
    print("Chargement du modèle...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Modèle chargé avec succès !")
    except Exception as e:
        print(f"Erreur chargement modèle : {e}")

# ─────────────────────────────────────────────
# 3. PREPROCESSING
# ─────────────────────────────────────────────
def prepare_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32)
    # Pas de division par 255 car EfficientNet gère son propre preprocessing
    return np.expand_dims(img_array, axis=0)

# ─────────────────────────────────────────────
# 4. ENDPOINTS
# ─────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None
    }

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    # Vérifier que le modèle est chargé
    if model is None:
        return JSONResponse(
            status_code=503,
            content={"message": "Modèle non disponible, réessayez dans quelques secondes."}
        )

    # Vérifier que c'est bien une image
    if not file.content_type.startswith("image/"):
        return JSONResponse(
            status_code=400,
            content={"message": "Le fichier envoyé n'est pas une image valide."}
        )

    try:
        contents = await file.read()

        # Vérifier la taille du fichier
        if len(contents) > MAX_UPLOAD_SIZE:
            return JSONResponse(
                status_code=413,
                content={"message": f"Image trop volumineuse. Taille max : {MAX_UPLOAD_SIZE // (1024*1024)} MB."}
            )
        img_array = prepare_image(contents)
        predictions = model.predict(img_array)

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
        return JSONResponse(
            status_code=500,
            content={"message": f"Erreur lors de la prédiction : {str(e)}"}
        )

# ─────────────────────────────────────────────
# 5. LANCEMENT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)