import zipfile, json

with zipfile.ZipFile('plant_disease_mobilenetv2.keras', 'r') as z:
    config = json.loads(z.read('config.json'))
    layers = config['config']['layers']
    
    # Print full detail for data_augmentation and TrueDivide/Subtract layers
    for i, layer in enumerate(layers):
        cn = layer.get('class_name', '?')
        nm = layer.get('name', '?')
        if i in [1, 2, 3]:
            print(f"\n=== Layer {i}: {cn} ({nm}) ===")
            if cn == 'Sequential':
                sub_layers = layer['config']['layers']
                print(f"  Sub-layers:")
                for j, sl in enumerate(sub_layers):
                    sl_cn = sl.get('class_name', '?')
                    sl_nm = sl.get('name', sl.get('config', {}).get('name', '?'))
                    print(f"    [{j}] {sl_cn} - {sl_nm}")
                    if sl_cn in ('Rescaling', 'TrueDivide', 'Subtract', 'Normalization'):
                        print(f"         config: {sl.get('config', {})}")
            else:
                print(f"  config keys: {list(layer.get('config', {}).keys())}")
                inbound = layer.get('inbound_nodes', [])
                print(f"  inbound_nodes: {inbound}")
