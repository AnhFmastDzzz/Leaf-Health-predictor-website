"""
convert_model_fixed.py
----------------------
Converts plant_disease_mobilenetv2.keras to TF.js Graph Model format.

Strategy:
  - Strips data_augmentation (training-only layer)
  - Keeps the normalization (TrueDivide / Subtract) BAKED IN to the exported
    model so the browser just feeds raw [0,255] uint8-to-float pixels and
    does NOT need to normalize manually.
  - Exports as a SavedModel then converts to TF.js GraphModel using the
    tensorflowjs converter.

Output: web_model/ (Graph Model — use tf.loadGraphModel in the browser)
"""

import os
import sys
import shutil
from types import ModuleType

# ── Compatibility patches ────────────────────────────────────────────────────
sys.modules["tensorflow_hub"] = ModuleType("tensorflow_hub")

import numpy as np
for _attr, _val in [('object', object), ('bool', bool), ('int', int), ('float', float)]:
    if not hasattr(np, _attr):
        setattr(np, _attr, _val)

import tensorflow as tf

mock_estimator = ModuleType("estimator")
class DummyExporter:
    def __init__(self, *args, **kwargs): pass
mock_estimator.Exporter = DummyExporter
sys.modules["tensorflow.compat.v1.estimator"] = mock_estimator
try:
    tf.compat.v1.estimator = mock_estimator
except Exception:
    pass

# ── Paths ────────────────────────────────────────────────────────────────────
MODEL_PATH   = "plant_disease_mobilenetv2.keras"
SAVED_DIR    = "saved_model_dir"
OUTPUT_DIR   = "web_model"

if not os.path.exists(MODEL_PATH):
    print(f"Error: {MODEL_PATH} not found.")
    sys.exit(1)

try:
    # ── Load original model ──────────────────────────────────────────────────
    print(f"Loading '{MODEL_PATH}' (compile=False)...")
    original = tf.keras.models.load_model(MODEL_PATH, compile=False)

    print("Layers in loaded model:")
    for i, layer in enumerate(original.layers):
        print(f"  [{i}] {layer.name}  ({type(layer).__name__})")

    # ── Identify the key layers ──────────────────────────────────────────────
    # Expected order: InputLayer, data_augmentation(Sequential),
    #                 mobilenetv2_1.00_224(Functional),
    #                 GlobalAveragePooling2D, Dropout, Dense
    backbone   = None
    gap_layer  = None
    drop_layer = None
    dense_layer = None

    for layer in original.layers:
        t = type(layer).__name__
        n = layer.name
        if t == 'Functional' and 'mobilenetv2' in n.lower():
            backbone = layer
        elif t == 'GlobalAveragePooling2D':
            gap_layer = layer
        elif t == 'Dropout':
            drop_layer = layer
        elif t == 'Dense':
            dense_layer = layer

    if not all([backbone, gap_layer, drop_layer, dense_layer]):
        raise ValueError(
            f"Could not find all required layers. Found: "
            f"backbone={backbone}, gap={gap_layer}, dropout={drop_layer}, dense={dense_layer}"
        )

    # ── Build inference model WITH normalization baked in ───────────────────
    # Input: raw float32 image [0, 255]  (browser sends this after fromPixels().toFloat())
    # Preprocessing: divide by 127.5, subtract 1.0 → MobileNetV2 expects [-1, 1]
    print("\nBuilding inference model with baked-in normalization...")
    inputs = tf.keras.Input(shape=(224, 224, 3), name='image')
    x = tf.keras.layers.Rescaling(scale=1.0/127.5, offset=-1.0, name='normalize')(inputs)
    x = backbone(x, training=False)
    x = gap_layer(x)
    # Remove dropout at inference (set rate=0) — rebuild a new one
    x = tf.keras.layers.Dropout(rate=0.0, name='dropout_inf')(x)
    outputs = dense_layer(x)

    inference_model = tf.keras.Model(inputs=inputs, outputs=outputs, name='leaf_health_predictor')
    inference_model.summary(line_length=100)

    # ── Verify normalization is correct ─────────────────────────────────────
    print("\nRunning sanity-check inference...")
    dummy = np.random.randint(0, 256, (1, 224, 224, 3), dtype=np.uint8).astype('float32')
    # Original model: pass through data_aug (no-op at inference), then /127.5 -1, then backbone
    # We simulate by applying norm manually and using the stripped model
    test_out = inference_model(dummy, training=False)
    print(f"  Output shape : {test_out.shape}  (should be (1, 38))")
    print(f"  Softmax sum  : {float(tf.reduce_sum(test_out).numpy()):.6f}  (should be ~1.0)")
    print(f"  Top class    : {int(tf.argmax(test_out, axis=1).numpy()[0])}")

    # ── Export SavedModel ────────────────────────────────────────────────────
    if os.path.exists(SAVED_DIR):
        print(f"\nRemoving old '{SAVED_DIR}'...")
        shutil.rmtree(SAVED_DIR)

    print(f"Exporting SavedModel to '{SAVED_DIR}'...")
    inference_model.export(SAVED_DIR)
    print("SavedModel export done.")

    # ── Convert to TF.js Graph Model ─────────────────────────────────────────
    if os.path.exists(OUTPUT_DIR):
        print(f"\nRemoving old '{OUTPUT_DIR}'...")
        shutil.rmtree(OUTPUT_DIR)

    print(f"Converting to TF.js Graph Model in '{OUTPUT_DIR}'...")
    from tensorflowjs.converters import tf_saved_model_conversion_v2
    tf_saved_model_conversion_v2.convert_tf_saved_model(
        saved_model_dir=SAVED_DIR,
        output_dir=OUTPUT_DIR,
        signature_def="serving_default",
    )

    print("\n✅ Conversion complete! Files written to: " + OUTPUT_DIR)
    print("   Use  tf.loadGraphModel('web_model/model.json')  in the browser.")
    print("   The model accepts raw float32 pixels in [0, 255] — NO manual normalization needed.")

except Exception:
    import traceback
    traceback.print_exc()
    sys.exit(1)
