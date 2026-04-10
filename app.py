import streamlit as st
import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from Vision import build_galaxy_eye
import Src

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(page_title="Galaxy Classifier AI", page_icon="🌌", layout="wide")
MODEL_FILENAME = "galaxy_eye.weights.h5"

@st.cache_resource
def load_vision_system():
    if not os.path.exists(MODEL_FILENAME):
        return None
    eye = build_galaxy_eye()
    eye.load_weights(MODEL_FILENAME)
    return eye

def make_gradcam_heatmap(img_array, model, last_conv_layer_name="conv2d_2"):
    """
    Generates a Grad-CAM heatmap to show where the AI is looking.
    """
    # Create a model functionally to avoid Keras 3 Sequential layer errors
    inputs = tf.keras.Input(shape=(64, 64, 3))
    x = inputs
    last_conv_layer_output = None
    
    for layer in model.layers:
        x = layer(x)
        if layer.name == last_conv_layer_name:
            last_conv_layer_output = x

    if last_conv_layer_output is None:
        raise ValueError(f"Layer {last_conv_layer_name} not found in model.")

    grad_model = tf.keras.Model(inputs, [last_conv_layer_output, x])

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    # Gradient of the top predicted class with regard to the output feature map
    grads = tape.gradient(top_class_channel, last_conv_layer_output)
    
    # Vector where each entry is the mean intensity of the gradient over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each channel in the feature map array by "how important this channel is" 
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


# ==========================================
# 2. UI LAYOUT
# ==========================================
st.title("🔭 Galaxy Classifier AI")
st.markdown("Upload a picture of a galaxy, and our AI pipeline will extract visual features and logically deduce its type.")

# Load Model
eye = load_vision_system()
if eye is None:
    st.error("Model weights not found. Please train the model first by running `Train_Vision.py`.")
    st.stop()

# File Uploader
uploaded_file = st.file_uploader("Choose a galaxy image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    col1, col2, col3 = st.columns(3)

    # ------------------------------------------
    # STEP A: Read Target Image
    # ------------------------------------------
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    with col1:
        st.subheader("1. Raw Input")
        st.image(img_rgb, use_container_width=True)

    # ------------------------------------------
    # STEP B: Feature Extraction (CNN) & Grad-CAM
    # ------------------------------------------
    img_resized = cv2.resize(img_bgr, (64, 64))
    img_input = tf.keras.applications.mobilenet_v2.preprocess_input(
        img_resized.astype(np.float32)
    )
    img_tensor = np.expand_dims(img_input, axis=0)
    
    probs = eye.predict(img_tensor, verbose=0)[0]
    
    # Generate Heatmap
    # We must find the name of the last Conv2D layer dynamically or hardcode it
    # Based on our architecture, we want the last Conv2D layer
    last_conv_layer_name = None
    for layer in reversed(eye.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer_name = layer.name
            break

    try:
        heatmap = make_gradcam_heatmap(img_tensor, eye, last_conv_layer_name)
        # Resize heatmap to match original image
        heatmap_resized = cv2.resize(heatmap, (img_bgr.shape[1], img_bgr.shape[0]))
        heatmap_resized = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        
        # Superimpose
        superimposed_img = cv2.addWeighted(img_bgr, 0.6, heatmap_colored, 0.4, 0)
        superimposed_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
        
        with col2:
            st.subheader("2. AI Focus (Grad-CAM)")
            st.image(superimposed_rgb, use_container_width=True, caption="Where the CNN is looking")
    except Exception as e:
        st.warning(f"Could not generate heatmap: {e}")

    # ------------------------------------------
    # STEP C: Logical Reasoning (Agent)
    # ------------------------------------------
    percepts = {
        "P_EL": float(probs[0]),
        "P_CW": float(probs[1]),
        "P_ACW": float(probs[2]),
        "EDGE_ON": float(probs[3]),
        "MERGER": float(probs[4])
    }

    beliefs, decision = Src.intelligent_agent(percepts)

    with col3:
        st.subheader("3. Logical Deduction")
        
        st.markdown("**Extracted Percepts (Probabilities):**")
        st.progress(percepts['P_EL'], text=f"Elliptical: {percepts['P_EL']:.2f}")
        st.progress(percepts['P_CW'], text=f"Spiral CW: {percepts['P_CW']:.2f}")
        st.progress(percepts['P_ACW'], text=f"Spiral ACW: {percepts['P_ACW']:.2f}")
        
        st.markdown("---")
        st.markdown("**Agent Belief State:**")
        for category, score in beliefs.items():
            st.write(f"- **{category}**: {score:.2f}")
            
        st.success(f"### Final Decision: {decision.upper()}")
