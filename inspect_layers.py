import zipfile, json

with zipfile.ZipFile('plant_disease_mobilenetv2.keras', 'r') as z:
    config = json.loads(z.read('config.json'))
    layers = config['config']['layers']
    print(f"Total layers: {len(layers)}")
    for i, layer in enumerate(layers):
        cn = layer.get('class_name', '?')
        nm = layer.get('name', '?')
        print(f"  Layer {i}: class={cn}, name={nm}")
        # Show extra info for normalization/rescaling layers
        if cn in ('Rescaling', 'Normalization', 'RandomFlip', 'RandomRotation', 'RandomZoom', 'Sequential'):
            print(f"    config: {layer.get('config', {})}")
