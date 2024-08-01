import onnxruntime as ort

# Get available execution providers
available_providers = ort.get_available_providers()

print("Available execution providers:")
for provider in available_providers:
    print(provider)