#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Run in Bash: export TF_METAL_DISABLE=1

#Force TensorFlow into single-threaded, non-XLA mode to work on MacOS
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)



import numpy as np
from tensorflow.keras import layers, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

# =====================
# Paths and parameters
# =====================
PATH = '/Users/maisievarcoe/Desktop/AI/Coursework/images/Originals'
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 50
SEED = 42

# =====================
# Detect classes safely
# =====================
class_names = sorted([
    d for d in os.listdir(PATH)
    if os.path.isdir(os.path.join(PATH, d))
])

num_classes = len(class_names)
print("Detected classes:", class_names)
print("Number of classes:", num_classes)

assert num_classes >= 2, "You must have at least 2 classes."

# =====================
# Load datasets
# =====================
train_ds = tf.keras.utils.image_dataset_from_directory(
    PATH,
    labels="inferred",
    label_mode="int",
    color_mode="grayscale",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    shuffle=True
)


val_ds = tf.keras.utils.image_dataset_from_directory(
    PATH,
    labels="inferred",
    label_mode="int",
    color_mode="grayscale",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    shuffle=False
)

# =====================
# Augmentation + preprocessing
# =====================
data_augmentation = Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomContrast(0.1),
    layers.RandomZoom(0.1)
])

def preprocess_train(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    x = data_augmentation(x)
    return x, y

def preprocess_val(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    return x, y

train_ds = train_ds.map(preprocess_train)
val_ds = val_ds.map(preprocess_val)

# ⚠️ CRITICAL for macOS stability
train_ds = train_ds.prefetch(1)
val_ds = val_ds.prefetch(1)

# =====================
# Compute class weights
# =====================
all_labels = np.concatenate([y.numpy() for _, y in train_ds], axis=0)

classes = np.unique(all_labels)
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=all_labels
)

class_weight_dict = dict(zip(classes, class_weights))
print("Class weights:", class_weight_dict)

# =====================
# Build CNN (dynamic output!)
# =====================
model = Sequential([
    layers.Input(shape=(224, 224, 1)),

    layers.Conv2D(32, 3, activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(num_classes, activation="softmax")
])


model.compile(
    optimizer=Adam(1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =====================
# Callbacks
# =====================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=1e-6
)

# =====================
# Train
# =====================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weight_dict,
    callbacks=[early_stop, reduce_lr]
)

# =====================
# Save model
# =====================
model.save("cnn_brain_model.h5")
print("✅ Model saved as cnn_brain_model.h5")


# In[ ]:




