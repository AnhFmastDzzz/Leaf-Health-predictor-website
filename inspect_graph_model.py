import json

with open('web_model/model.json') as f:
    model_json = json.load(f)

# Print the signature info
sig = model_json.get('signature', {})
print("=== Model Signature ===")
print(f"Inputs:  {list(sig.get('inputs', {}).keys())}")
print(f"Outputs: {list(sig.get('outputs', {}).keys())}")

# Full input details
print("\n=== Input details ===")
for k, v in sig.get('inputs', {}).items():
    print(f"  {k}: shape={v.get('tensorShape')}, dtype={v.get('dtype')}")

print("\n=== Output details ===")
for k, v in sig.get('outputs', {}).items():
    print(f"  {k}: shape={v.get('tensorShape')}, dtype={v.get('dtype')}")
