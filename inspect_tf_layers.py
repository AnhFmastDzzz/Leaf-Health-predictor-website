import tensorflow as tf

model = tf.keras.models.load_model('plant_disease_mobilenetv2.keras', compile=False)
print("All layers with index:")
for i, layer in enumerate(model.layers):
    print(f"  [{i}] {layer.name} ({type(layer).__name__})")
    if hasattr(layer, 'output_shape'):
        try:
            print(f"       output_shape: {layer.output_shape}")
        except:
            pass
