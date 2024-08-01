import onnx

# Load the ONNX model
model = onnx.load("etc/programs/image-to-text/image_to_text.onnx")

# Print the input dimensions
for input in model.graph.input:
    print(f"Input name: {input.name}")
    print(f"Shape: {input.type.tensor_type.shape}")
