import base64
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import uvicorn
import os

from Vision import build_galaxy_eye
import Src

# ==========================================
# SETUP
# ==========================================
app = FastAPI(title="Galaxy Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_FILENAME = "galaxy_eye.weights.h5"
try:
    eye = build_galaxy_eye()
    eye.load_weights(MODEL_FILENAME)
    print("✅ Vision System loaded (MobileNetV2 backbone).")
except Exception as e:
    print(f"⚠️ Error loading brain: {e}")
    eye = None


# ==========================================
# GRAD-CAM
# ==========================================
def make_gradcam_heatmap(img_array, model):
    """
    Generates a Grad-CAM heatmap for a model with a nested backbone.
    Works by finding the MobileNetV2 sub-model and using its output
    (the last convolutional feature map) as the gradient target.
    """
    # Find the backbone sub-model (MobileNetV2) inside our top-level model
    backbone = None
    for layer in model.layers:
        if hasattr(layer, 'layers') and len(layer.layers) > 10:
            backbone = layer
            break

    if backbone is None:
        raise ValueError("Could not find backbone sub-model.")

    backbone_idx = model.layers.index(backbone)
    head_layers = model.layers[backbone_idx + 1:]

    # Eagerly compute the convolutional features
    conv_outputs = backbone(img_array, training=False)

    with tf.GradientTape() as tape:
        tape.watch(conv_outputs)
        
        # Pass features through the classification head
        preds = conv_outputs
        for layer in head_layers:
            preds = layer(preds, training=False)
            
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    grads = tape.gradient(top_class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


# ==========================================
# ROUTES
# ==========================================
@app.post("/predict")
async def predict_galaxy(file: UploadFile = File(...)):
    if eye is None:
        return {"error": "Model not loaded on server."}

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        return {"error": "Invalid image file."}

    # 1. Feature Extraction
    img_resized = cv2.resize(img_bgr, (64, 64))
    img_input = tf.keras.applications.mobilenet_v2.preprocess_input(
        img_resized.astype(np.float32)
    )
    img_tensor = np.expand_dims(img_input, axis=0)

    probs = eye.predict(img_tensor, verbose=0)[0]

    percepts = {
        "P_EL": float(probs[0]),
        "P_CW": float(probs[1]),
        "P_ACW": float(probs[2]),
        "EDGE_ON": float(probs[3]),
        "MERGER": float(probs[4])
    }

    # 2. Logic Reasoning
    beliefs, decision = Src.intelligent_agent(percepts)

    # 3. Grad-CAM
    heatmap_b64 = None
    try:
        heatmap = make_gradcam_heatmap(img_tensor, eye)
        heatmap_resized = cv2.resize(heatmap, (img_bgr.shape[1], img_bgr.shape[0]))
        heatmap_resized = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

        superimposed = cv2.addWeighted(img_bgr, 0.6, heatmap_colored, 0.4, 0)
        _, buffer = cv2.imencode('.jpg', superimposed)
        heatmap_b64 = base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        print(f"Heatmap Error: {e}")

    _, buffer_raw = cv2.imencode('.jpg', img_bgr)
    raw_b64 = base64.b64encode(buffer_raw).decode('utf-8')

    return {
        "decision": decision,
        "percepts": percepts,
        "beliefs": beliefs,
        "heatmapBase64": heatmap_b64,
        "rawBase64": raw_b64
    }


# ==========================================
# FEEDBACK ENDPOINT (Human-in-the-Loop)
# ==========================================
class FeedbackRequest(BaseModel):
    true_label: str  # "Elliptical", "Spiral", or "Uncertain"
    percepts: dict

@app.post("/feedback")
async def submit_feedback(req: FeedbackRequest):
    """
    Receives expert correction from the UI.
    The agent re-runs its reasoning engine, then adjusts
    its rule confidences based on the true label.
    """
    beliefs, fired_rules = Src.reasoning_engine(req.percepts, Src.rules)
    Src.learn_from_feedback(Src.rules, fired_rules, req.true_label)

    updated_rules = [
        {"conclusion": r["conclusion"], "confidence": round(r["confidence"], 3)}
        for r in Src.rules
    ]

    return {
        "status": "ok",
        "message": f"Agent updated. Feedback '{req.true_label}' applied.",
        "updatedRules": updated_rules
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
# Trigger reload
