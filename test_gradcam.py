import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import traceback

def build_galaxy_eye(input_shape=(64, 64, 3)):
    inputs = layers.Input(shape=input_shape)
    x = layers.RandomFlip("horizontal_and_vertical")(inputs)
    x = layers.RandomRotation(0.2)(x)
    x = layers.RandomZoom(height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2))(x)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    x = base_model(x, training=False)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(5, activation='sigmoid')(x)

    model = models.Model(inputs, outputs)
    return model

model = build_galaxy_eye()
img_tensor = np.random.rand(1, 64, 64, 3).astype(np.float32)

backbone = None
for layer in model.layers:
    if hasattr(layer, 'layers') and len(layer.layers) > 10:
        backbone = layer
        break

print("Backbone found:", backbone.name)

try:
    backbone_idx = model.layers.index(backbone)
    head_layers = model.layers[backbone_idx+1:]
    
    # Run backbone
    conv_outputs = backbone(img_tensor, training=False)
    
    # tape watches the conv_outputs
    with tf.GradientTape() as tape:
        tape.watch(conv_outputs)
        
        preds = conv_outputs
        for layer in head_layers:
            preds = layer(preds, training=False)
            
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]
        
    grads = tape.gradient(top_class_channel, conv_outputs)
    print("SUCCESS full manual eager pass. Grads shape:", grads.shape)
except Exception as e:
    print("FAILED full manual eager pass")
    traceback.print_exc()

