"""
Diagnostic script: runs a single test inference through the model 
to verify the normalization and prediction pipeline are correct.
"""
import numpy as np
import tensorflow as tf

print("Loading model...")
model = tf.keras.models.load_model('plant_disease_mobilenetv2.keras', compile=False)

# Simulate the browser inference pipeline:
# 1. Create a dummy RGB image (uint8, 0-255)
dummy_img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)

print("\n--- Testing ORIGINAL model (with data_augmentation) ---")
# The full model takes uint8 input
inp = dummy_img.astype('float32')[np.newaxis, ...]  # add batch dim
pred_full = model(inp, training=False)
print(f"Output shape: {pred_full.shape}")
print(f"Prediction sum (should be ~1.0 for softmax): {np.sum(pred_full.numpy()):.6f}")
print(f"Top class: {np.argmax(pred_full.numpy())}")

print("\n--- Rebuilding stripped model (layers 2-5, skipping data_aug + normalization) ---")
# Replicate what convert_model.py does
new_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3), name='image'),
    model.layers[2],  # mobilenetv2_1.00_224
    model.layers[3],  # global_average_pooling
    model.layers[4],  # dropout
    model.layers[5],  # predictions
])

# Now test with manual normalization (as done in index.html)
normalized = (dummy_img.astype('float32') / 127.5 - 1.0)[np.newaxis, ...]
pred_stripped = new_model(normalized, training=False)
print(f"Output shape: {pred_stripped.shape}")
print(f"Prediction sum (should be ~1.0 for softmax): {np.sum(pred_stripped.numpy()):.6f}")
print(f"Top class: {np.argmax(pred_stripped.numpy())}")

print("\n--- Comparing predictions ---")
print(f"Full model top-5: {np.argsort(pred_full.numpy()[0])[-5:][::-1]}")
print(f"Stripped model top-5: {np.argsort(pred_stripped.numpy()[0])[-5:][::-1]}")
match = np.argmax(pred_full.numpy()) == np.argmax(pred_stripped.numpy())
print(f"Top prediction match: {match}")

# Check if normalization is baked into the full model's data_augmentation at inference
print("\n--- What does data_augmentation do at inference time? ---")
data_aug = model.layers[1]
aug_out = data_aug(inp, training=False)
print(f"After data_aug output range: [{aug_out.numpy().min():.2f}, {aug_out.numpy().max():.2f}]")
# This tells us if data_augmentation applies normalization at inference
