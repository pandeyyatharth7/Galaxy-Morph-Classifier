"""
BatchClassifier.py — Autonomous Batch Classification Agent
============================================================
Processes an entire folder of galaxy images without human
intervention. Outputs a CSV file with predictions.

Usage:
    python BatchClassifier.py
    python BatchClassifier.py --input galaxy_images --output results.csv
"""

import os
import sys
import csv
import cv2
import numpy as np
import tensorflow as tf
from Vision import build_galaxy_eye
import Src

# ==========================================
# CONFIGURATION
# ==========================================
IMAGE_DIR = "galaxy_images"
OUTPUT_CSV = "classifications_output.csv"
MODEL_FILENAME = "galaxy_eye.weights.h5"

# Allow command-line overrides
if "--input" in sys.argv:
    IMAGE_DIR = sys.argv[sys.argv.index("--input") + 1]
if "--output" in sys.argv:
    OUTPUT_CSV = sys.argv[sys.argv.index("--output") + 1]

# ==========================================
# 1. LOAD VISION SYSTEM
# ==========================================
print("=" * 50)
print("  AUTONOMOUS BATCH CLASSIFIER")
print("=" * 50)

if not os.path.exists(MODEL_FILENAME):
    print("❌ Error: Model weights not found. Train the model first.")
    sys.exit(1)

eye = build_galaxy_eye()
eye.load_weights(MODEL_FILENAME)
print(f"✅ Vision System loaded ({MODEL_FILENAME})")

# ==========================================
# 2. DISCOVER IMAGES
# ==========================================
valid_extensions = ('.jpg', '.jpeg', '.png')
image_files = [
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(valid_extensions)
]

if not image_files:
    print(f"❌ No images found in '{IMAGE_DIR}'. Nothing to classify.")
    sys.exit(1)

print(f"📂 Found {len(image_files)} images in '{IMAGE_DIR}'")
print(f"📄 Output will be saved to '{OUTPUT_CSV}'")
print("-" * 50)

# ==========================================
# 3. CLASSIFY ALL IMAGES
# ==========================================
results = []
errors = 0

for i, filename in enumerate(image_files):
    try:
        img_path = os.path.join(IMAGE_DIR, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"  ⚠️ Skipping unreadable file: {filename}")
            errors += 1
            continue

        # Preprocess for MobileNetV2
        img_resized = cv2.resize(img, (64, 64))
        img_input = tf.keras.applications.mobilenet_v2.preprocess_input(
            img_resized.astype(np.float32)
        )
        img_tensor = np.expand_dims(img_input, axis=0)

        # Extract percepts via the CNN
        probs = eye.predict(img_tensor, verbose=0)[0]
        percepts = {
            "P_EL": float(probs[0]),
            "P_CW": float(probs[1]),
            "P_ACW": float(probs[2]),
            "EDGE_ON": float(probs[3]),
            "MERGER": float(probs[4])
        }

        # Reason via the symbolic agent
        beliefs, decision = Src.intelligent_agent(percepts)

        results.append({
            "filename": filename,
            "decision": decision,
            "P_EL": f"{percepts['P_EL']:.4f}",
            "P_CW": f"{percepts['P_CW']:.4f}",
            "P_ACW": f"{percepts['P_ACW']:.4f}",
            "EDGE_ON": f"{percepts['EDGE_ON']:.4f}",
            "MERGER": f"{percepts['MERGER']:.4f}",
            "belief_spiral": f"{beliefs['Spiral']:.3f}",
            "belief_elliptical": f"{beliefs['Elliptical']:.3f}",
            "belief_uncertain": f"{beliefs['Uncertain']:.3f}",
        })

        # Progress indicator every 50 images
        if (i + 1) % 50 == 0 or (i + 1) == len(image_files):
            print(f"  Processed {i + 1}/{len(image_files)} images...")

    except Exception as e:
        print(f"  ⚠️ Error processing {filename}: {e}")
        errors += 1

# ==========================================
# 4. WRITE OUTPUT CSV
# ==========================================
if results:
    fieldnames = list(results[0].keys())
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("-" * 50)
    print(f"✅ Classification complete!")
    print(f"   Classified: {len(results)} images")
    print(f"   Errors:     {errors}")
    print(f"   Output:     {OUTPUT_CSV}")

    # Quick summary
    from collections import Counter
    counts = Counter(r["decision"] for r in results)
    print("\n📊 Distribution:")
    for label, count in counts.most_common():
        pct = count / len(results) * 100
        print(f"   {label:12s}: {count:4d} ({pct:.1f}%)")
else:
    print("❌ No images were classified successfully.")
