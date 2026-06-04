import zipfile
import json
import os

model_path = "plant_disease_mobilenetv2.keras"

if not os.path.exists(model_path):
    print(f"Error: {model_path} not found.")
    exit(1)

try:
    with zipfile.ZipFile(model_path, 'r') as zip_ref:
        print("Files inside Keras model archive:")
        zip_ref.printdir()
        
        # Look for config.json or metadata.json
        file_list = zip_ref.namelist()
        
        if 'config.json' in file_list:
            print("\nReading config.json...")
            config_data = json.loads(zip_ref.read('config.json').decode('utf-8'))
            print("Model Keras version:", config_data.get('keras_version'))
            print("Model backend:", config_data.get('backend'))
            
            # Let's inspect layers, especially inputs and outputs
            model_config = config_data.get('config', {})
            layers = model_config.get('layers', [])
            if layers:
                print(f"Number of layers: {len(layers)}")
                # Print input layer config
                print("First layer:", layers[0])
                # Print last layer config
                print("Last layer:", layers[-1])
        
        if 'metadata.json' in file_list:
            print("\nReading metadata.json...")
            metadata_data = json.loads(zip_ref.read('metadata.json').decode('utf-8'))
            print(json.dumps(metadata_data, indent=2))
            
except Exception as e:
    print(f"Error reading Keras zip archive: {e}")
