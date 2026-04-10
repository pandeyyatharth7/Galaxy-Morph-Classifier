import tensorflow as tf
from tensorflow.keras import layers, models

def build_galaxy_eye(input_shape=(64, 64, 3)):
    """
    Builds the Galaxy Eye vision model using Transfer Learning.
    
    Architecture:
      - MobileNetV2 (pre-trained on ImageNet) acts as the frozen 
        feature-extraction backbone — the agent's "Visual Cortex".
      - A lightweight custom head maps the extracted features 
        to our 5 galactic-morphology percepts.
    """
    # ==========================================
    # 1. DATA AUGMENTATION (Perception Noise)
    # ==========================================
    # Teaches the agent that a spiral is still a spiral
    # regardless of orientation, scale, or flip.
    inputs = layers.Input(shape=input_shape)
    x = layers.RandomFlip("horizontal_and_vertical")(inputs)
    x = layers.RandomRotation(0.2)(x)
    x = layers.RandomZoom(height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2))(x)

    # ==========================================
    # 2. FOUNDATION MODEL BACKBONE (Transfer Learning)
    # ==========================================
    # MobileNetV2 was pre-trained on 1.4M images (ImageNet).
    # We freeze its weights so it acts as a powerful, fixed
    # feature extractor — far superior to a 3-layer custom CNN.
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,        # Remove ImageNet classification head
        weights='imagenet'
    )
    base_model.trainable = False  # Freeze the backbone

    x = base_model(x, training=False)

    # ==========================================
    # 3. CLASSIFICATION HEAD (Reasoning Bridge)
    # ==========================================
    # Global Average Pooling compresses the spatial feature maps
    # into a single vector per filter — reducing overfitting.
    x = layers.GlobalAveragePooling2D()(x)

    # Dropout: randomly silences 50% of neurons during training
    # to prevent memorisation of the small training set.
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(128, activation='relu')(x)

    # Output: 5 independent probabilities (percepts)
    #   P_EL, P_CW, P_ACW, EDGE_ON, MERGER
    outputs = layers.Dense(5, activation='sigmoid')(x)

    model = models.Model(inputs, outputs)

    # Lower learning rate — the backbone is already well-tuned;
    # we only need gentle updates to the new head.
    opt = tf.keras.optimizers.Adam(learning_rate=0.0005)

    model.compile(
        optimizer=opt,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model
