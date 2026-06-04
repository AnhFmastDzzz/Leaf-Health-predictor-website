import os
import sys
import shutil
from types import ModuleType

# Mock tensorflow_hub in sys.modules to prevent protobuf/estimator issues
sys.modules["tensorflow_hub"] = ModuleType("tensorflow_hub")

# Monkeypatch numpy for tensorflowjs 3.x compatibility on NumPy 2.x
import numpy as np
if not hasattr(np, 'object'):
    np.object = object
if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float

# Monkeypatch tensorflow compat.v1.estimator using wrapper injection
import tensorflow as tf
mock_estimator = ModuleType("estimator")
class DummyExporter:
    def __init__(self, *args, **kwargs):
        pass
mock_estimator.Exporter = DummyExporter
sys.modules["tensorflow.compat.v1.estimator"] = mock_estimator
if hasattr(tf.compat.v1, "_tfmw_wrapped_module"):
    tf.compat.v1._tfmw_wrapped_module.estimator = mock_estimator
else:
    tf.compat.v1.estimator = mock_estimator

# Import the SavedModel conversion module directly
from tensorflowjs.converters import tf_saved_model_conversion_v2

model_path = "plant_disease_mobilenetv2.keras"
saved_model_dir = "saved_model_dir"
output_dir = "web_model"

if not os.path.exists(model_path):
    print(f"Error: {model_path} not found.")
    sys.exit(1)

try:
    print(f"Loading Keras model (compile=False) from '{model_path}'...")
    model = tf.keras.models.load_model(model_path, compile=False)
    
    print("Rebuilding model to strip data_augmentation layer...")
    # Extract only the inference layers, skipping data_augmentation (index 1)
    new_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3), name='image'),
        model.layers[2], # mobilenetv2_1.00_224 functional base
        model.layers[3], # global_average_pooling
        model.layers[4], # dropout
        model.layers[5], # predictions
    ])
    new_model.summary()
    
    # Remove existing saved_model_dir if exists to avoid conflicts
    if os.path.exists(saved_model_dir):
        print(f"Cleaning up old directory '{saved_model_dir}'...")
        shutil.rmtree(saved_model_dir)
        
    print(f"Exporting clean model to SavedModel directory '{saved_model_dir}'...")
    new_model.export(saved_model_dir)
    
    # Remove existing output_dir if exists to avoid conflicts
    if os.path.exists(output_dir):
        print(f"Cleaning up old output directory '{output_dir}'...")
        shutil.rmtree(output_dir)
        
    print(f"Converting SavedModel from '{saved_model_dir}' to TF.js Graph Model in '{output_dir}'...")
    tf_saved_model_conversion_v2.convert_tf_saved_model(
        saved_model_dir=saved_model_dir,
        output_dir=output_dir,
        signature_def="serving_default"
    )
    print("Graph model conversion completed successfully!")
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
