import os
import tensorflow as tf
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from Vision import build_galaxy_eye
import Src  # Reasoning Agent

# ==========================================
# CONFIGURATION
# ==========================================
IMAGE_DIR = "galaxy_images"
CSV_FILE = "GalaxyZoo1_DR_table2.csv"
MODEL_FILENAME = "galaxy_eye.weights.h5"

print("--- EVALUATION METRICS SCRIPT ---")

# 1. Load Data (Evaluation Set)
df = pd.read_csv(CSV_FILE)
valid_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')]

if not valid_files:
    print("❌ Error: No images found. Run Get_Images.py first.")
    exit()

print(f"Loading {len(valid_files)} images for evaluation...")

# We will test on a subset to keep evaluation fast, 
# or all of them if there are few.
eval_files = valid_files[:200]  # Take up to 200 for evaluation

y_true = []
y_pred = []

# Load the Brain
if not os.path.exists(MODEL_FILENAME):
    print("❌ Error: Model missing. Run Train_Vision.py first.")
    exit()

eye = build_galaxy_eye()
eye.load_weights(MODEL_FILENAME)
print("✅ Vision System loaded.")

print("\nProcessing images and extracting CNN probabilities...")
for filename in eval_files:
    try:
        obj_id = int(filename.split('.')[0])
        row = df[df['OBJID'] == obj_id]

        if not row.empty:
            img_path = os.path.join(IMAGE_DIR, filename)
            img = cv2.imread(img_path)
            if img is not None:
                # Prepare image
                img_resized = cv2.resize(img, (64, 64))
                img_input = tf.keras.applications.mobilenet_v2.preprocess_input(
                    img_resized.astype(np.float32)
                )
                img_input = np.expand_dims(img_input, axis=0)
                
                # Get CNN predictions
                probs = eye.predict(img_input, verbose=0)[0]
                percepts = {
                    "P_EL": probs[0],
                    "P_CW": probs[1],
                    "P_ACW": probs[2],
                    "EDGE_ON": probs[3],
                    "MERGER": probs[4]
                }
                
                # Let the logical agent decide based on the thresholds
                beliefs, decision = Src.intelligent_agent(percepts)
                
                # Find the True Label from the CSV based on probability thresholding
                # (The exact logic scientists use: Probability > 0.6)
                if row['P_EL'].values[0] > 0.6:
                    true_label = "Elliptical"
                elif row['P_CW'].values[0] > 0.6 or row['P_ACW'].values[0] > 0.6:
                    true_label = "Spiral"
                else:
                    true_label = "Uncertain"
                    
                y_true.append(true_label)
                y_pred.append(decision)
                
    except Exception as e:
        print(f"Skipping {filename}: {e}")

# ==========================================
# 2. CONFUSION MATRIX VISUALIZATION
# ==========================================
labels = ["Elliptical", "Spiral", "Uncertain"]

print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(y_true, y_pred, target_names=labels, zero_division=0))

cm = confusion_matrix(y_true, y_pred, labels=labels)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Galaxy Classification Confusion Matrix')
plt.xlabel('AI Predicted Label')
plt.ylabel('True Label (Based on Prob > 0.6)')
plt.tight_layout()

# Save the confusion matrix to a file
plt.savefig('confusion_matrix.png')
print("✅ Confusion Matrix saved as 'confusion_matrix.png'")
plt.show()
