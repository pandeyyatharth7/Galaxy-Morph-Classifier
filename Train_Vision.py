import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from Vision import build_galaxy_eye

# ==========================================
# CONFIGURATION
# ==========================================
IMAGE_DIR = "galaxy_images"
CSV_FILE = "GalaxyZoo1_DR_table2.csv"
MODEL_FILENAME = "galaxy_eye.weights.h5"

# ==========================================
# 1. LOAD DATA
# ==========================================
print("--- LOADING TRAINING DATA ---")

df = pd.read_csv(CSV_FILE)

images = []
labels = []

valid_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')]
print(f"Found {len(valid_files)} images in folder.")

if len(valid_files) < 200:
    print("⚠️ WARNING: You have very few images. Accuracy will be low.")
    print("   Please run 'Get_Images.py' with LIMIT=1000 first.")

for filename in valid_files:
    try:
        obj_id = int(filename.split('.')[0])
        row = df[df['OBJID'] == obj_id]

        if not row.empty:
            img_path = os.path.join(IMAGE_DIR, filename)
            img = cv2.imread(img_path)

            if img is not None:
                img = cv2.resize(img, (64, 64))
                # MobileNetV2 preprocess: scales pixels to [-1, 1]
                img_processed = tf.keras.applications.mobilenet_v2.preprocess_input(
                    img.astype(np.float32)
                )
                images.append(img_processed)

                label_vector = [
                    row['P_EL'].values[0],
                    row['P_CW'].values[0],
                    row['P_ACW'].values[0],
                    row['P_EDGE'].values[0],
                    row['P_MG'].values[0]
                ]
                labels.append(label_vector)
    except Exception as e:
        print(f"Skipping bad file {filename}: {e}")

X = np.array(images)
y = np.array(labels)

print(f"✅ Ready to train on {len(X)} samples.")

# ==========================================
# 2. LOAD OR BUILD MODEL
# ==========================================
if len(X) > 0:

    if os.path.exists(MODEL_FILENAME):
        print("\n🧠 LOADING EXISTING BRAIN (Transfer Learning Model)...")
        print("   (Continuing training from where we left off)")
        try:
            model = build_galaxy_eye()
            model.load_weights(MODEL_FILENAME)
        except Exception as e:
            print(f"⚠️ Error loading brain: {e}. Building a new one.")
            model = build_galaxy_eye()
    else:
        print("\n👶 STARTING FRESH WITH MOBILENETV2 BACKBONE...")
        print("   (Pre-trained on 1.4M ImageNet images — Transfer Learning)")
        model = build_galaxy_eye()

    # ==========================================
    # 3. TRAIN (LEARN)
    # ==========================================
    print("\n--- TRAINING START ---")

    history = model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2)

    model.save_weights(MODEL_FILENAME)
    print(f"\n✅ Training Complete! Saved weights to '{MODEL_FILENAME}'")

    # ==========================================
    # 4. SHOW PROGRESS
    # ==========================================
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Acc')
    plt.plot(history.history['val_accuracy'], label='Validation Acc')
    plt.title('How smart is the AI?')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Error Rate (Lower is better)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_progress.png')
    print("✅ Training progress saved as 'training_progress.png'")
    plt.show()

else:
    print("❌ Critical Error: No images loaded. Run 'Get_Images.py' first!")
