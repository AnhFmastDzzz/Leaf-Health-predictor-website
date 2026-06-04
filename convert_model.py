import os
import sys
import json

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

import tensorflow as tf
from tensorflowjs.converters.keras_h5_conversion import write_artifacts

model_path = "plant_disease_mobilenetv2.keras"
output_dir = "web_model"

if not os.path.exists(model_path):
    print(f"Error: {model_path} not found.")
    sys.exit(1)

def translate_inbound_nodes(nodes):
    """Translate Keras 3 inbound_nodes dictionary to legacy Keras 2 list format."""
    if not isinstance(nodes, list):
        return nodes
    
    keras2_nodes = []
    for node in nodes:
        if not isinstance(node, dict):
            keras2_nodes.append(node)
            continue
            
        args = node.get('args', [])
        tensors = []
        
        def find_tensors(item):
            if isinstance(item, dict):
                if item.get('class_name') == '__keras_tensor__':
                    tensors.append(item.get('config', {}))
                else:
                    for val in item.values():
                        find_tensors(val)
            elif isinstance(item, list):
                for val in item:
                    find_tensors(val)
                    
        find_tensors(args)
        
        node_connections = []
        for tensor_config in tensors:
            history = tensor_config.get('keras_history', [])
            if len(history) >= 3:
                layer_name, node_index, tensor_index = history[0], history[1], history[2]
                node_connections.append([layer_name, node_index, tensor_index, {}])
                
        if node_connections:
            keras2_nodes.append(node_connections)
            
    return keras2_nodes

def fix_config(obj):
    """Recursively map Keras 3 config keys to Keras 2 formats expected by TF.js."""
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            new_key = k
            if k == 'batch_shape':
                new_key = 'batch_input_shape'
            
            # Map Keras 3 'Functional' class_name to Keras 2 'Model'
            new_val = v
            if k == 'class_name' and v == 'Functional':
                new_val = 'Model'
            elif k == 'inbound_nodes':
                new_val = translate_inbound_nodes(v)
            else:
                new_val = fix_config(v)
                
            new_dict[new_key] = new_val
        return new_dict
    elif isinstance(obj, list):
        return [fix_config(x) for x in obj]
    else:
        return obj

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
    
    print("Extracting model configuration...")
    # Serialize the stripped model configuration
    model_config = json.loads(new_model.to_json())
    
    # Fix Keras 3 config bugs for TF.js compatibility (input shape, inbound nodes, class names)
    model_config = fix_config(model_config)
    
    topology_json = {
        'keras_version': tf.keras.__version__,
        'backend': tf.keras.backend.backend(),
        'model_config': model_config
    }
    
    print("Extracting model weights...")
    weight_list = []
    for w in new_model.weights:
        name = w.path
        if name.endswith(':0'):
            name = name[:-2]
        
        weight_list.append({
            'name': name,
            'data': w.numpy()
        })
    
    weight_groups = [weight_list]
    
    print(f"Writing TF.js artifacts to '{output_dir}'...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    write_artifacts(
        topology=topology_json,
        weights=weight_groups,
        output_dir=output_dir
    )
    
    print("Conversion completed successfully!")
    
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
