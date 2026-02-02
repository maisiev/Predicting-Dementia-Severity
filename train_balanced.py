#!/usr/bin/env python3
"""
Training with Class-Balanced Sampling
Uses weighted random sampling to balance classes during training
"""

import os
os.environ["TF_METAL_DISABLE"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Sequential, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from collections import Counter

PATH = "/Users/maisievarcoe/Desktop/AI/Coursework/images/Originals"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32  # Increased since we're balancing
EPOCHS = 80
SEED = 42

print("="*60)
print("BALANCED TRAINING WITH WEIGHTED SAMPLING")
print("="*60)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Get classes
class_names = sorted([d for d in os.listdir(PATH)
                      if os.path.isdir(os.path.join(PATH, d)) 
                      and not d.startswith('.')])
num_classes = len(class_names)
print(f"Classes: {class_names}\n")

# Collect all file paths and labels
all_files = []
all_labels = []

for class_idx, class_name in enumerate(class_names):
    class_path = os.path.join(PATH, class_name)
    images = [os.path.join(class_path, f) 
              for f in os.listdir(class_path)
              if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
              and not f.startswith('.')]
    all_files.extend(images)
    all_labels.extend([class_idx] * len(images))
    print(f"{class_name}: {len(images)} images")

# Convert to numpy
all_files = np.array(all_files)
all_labels = np.array(all_labels)

# Shuffle
np.random.seed(SEED)
indices = np.random.permutation(len(all_files))
all_files = all_files[indices]
all_labels = all_labels[indices]

# Split train/val
split_idx = int(0.8 * len(all_files))
train_files = all_files[:split_idx]
train_labels = all_labels[:split_idx]
val_files = all_files[split_idx:]
val_labels = all_labels[split_idx:]

print(f"\nTraining samples: {len(train_files)}")
print(f"Validation samples: {len(val_files)}")

# Calculate class weights for training
train_class_counts = Counter(train_labels)
print(f"\nTraining class distribution:")
for class_idx, count in sorted(train_class_counts.items()):
    print(f"  {class_names[class_idx]}: {count}")

# Calculate sample weights (inverse frequency)
max_count = max(train_class_counts.values())
sample_weights = np.array([max_count / train_class_counts[label] 
                           for label in train_labels])

print(f"\nSample weight range: {sample_weights.min():.2f} - {sample_weights.max():.2f}")

# Create datasets with weighted sampling
def load_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=1, expand_animations=False)
    img.set_shape([None, None, 1])
    img = tf.image.resize(img, IMG_SIZE)
    return img, label

# Training dataset with sampling weights
train_ds = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
# Create weighted sampler
weighted_indices = np.random.choice(
    len(train_files), 
    size=len(train_files) * 2,  # Oversample
    p=sample_weights / sample_weights.sum(),
    replace=True
)
weighted_files = train_files[weighted_indices]
weighted_labels = train_labels[weighted_indices]

train_ds = tf.data.Dataset.from_tensor_slices((weighted_files, weighted_labels))
train_ds = train_ds.shuffle(1000, seed=SEED)
train_ds = train_ds.map(load_image, num_parallel_calls=2)

# Validation dataset (no weighting)
val_ds = tf.data.Dataset.from_tensor_slices((val_files, val_labels))
val_ds = val_ds.map(load_image, num_parallel_calls=2)

# Preprocessing
def augment(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_brightness(x, 0.15)
    x = tf.image.random_contrast(x, 0.85, 1.15)
    return x, y

def normalize(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    return x, y

train_ds = train_ds.map(augment, num_parallel_calls=2)
val_ds = val_ds.map(normalize, num_parallel_calls=2)

train_ds = train_ds.batch(BATCH_SIZE).prefetch(2)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(2)

print("\n✓ Data pipeline ready\n")

# Build model
print("Building model...")
model = Sequential([
    layers.Input(shape=(224, 224, 1)),
    
    layers.Conv2D(32, 3, activation="relu", padding="same", 
                  kernel_regularizer=regularizers.l2(0.0005)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    
    layers.Conv2D(64, 3, activation="relu", padding="same",
                  kernel_regularizer=regularizers.l2(0.0005)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    
    layers.Conv2D(128, 3, activation="relu", padding="same",
                  kernel_regularizer=regularizers.l2(0.0005)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.3),
    
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu",
                 kernel_regularizer=regularizers.l2(0.0005)),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation="softmax"),
])

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# Callbacks
callbacks = [
    EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-7, verbose=1),
    ModelCheckpoint("best_balanced_model.h5", monitor="val_loss", save_best_only=True, verbose=1),
]

print("\n" + "="*60)
print("TRAINING")
print("="*60 + "\n")

# Train
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1,
)

# Save
model.save("final_balanced_model.h5")
print("\n✓ Model saved")

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history.history['accuracy'], label='Training', linewidth=2)
ax1.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.set_title('Model Accuracy (Balanced Training)')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(history.history['loss'], label='Training', linewidth=2)
ax2.plot(history.history['val_loss'], label='Validation', linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.set_title('Model Loss (Balanced Training)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('balanced_training_results.png', dpi=300)
print("✓ Plot saved\n")

# Results
print("="*60)
print("FINAL RESULTS")
print("="*60)
final_val_acc = history.history['val_accuracy'][-1]
best_val_acc = max(history.history['val_accuracy'])
print(f"Final Validation Accuracy: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")
print(f"Best Validation Accuracy:  {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
print("="*60)
