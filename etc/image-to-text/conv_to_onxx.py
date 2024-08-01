from transformers import VisionEncoderDecoderModel, AutoTokenizer
import torch

# Load the pre-trained model and tokenizer
model = VisionEncoderDecoderModel.from_pretrained('etc/programs/models/image to text/nlpconnectvit-gpt2-image-captioning')
tokenizer = AutoTokenizer.from_pretrained('etc/programs/models/image to text/nlpconnectvit-gpt2-image-captioning')

# Prepare dummy inputs
dummy_pixel_values = torch.randn(1, 3, 224, 224)  # Example image input; adjust size if necessary
dummy_input_ids = torch.randint(0, tokenizer.vocab_size, (1, 1))  # Example input_ids for the decoder; ensure shape matches model expectations

# Define the path for the ONNX model
onnx_model_path = 'etc/programs/image-to-text/image_to_text.onnx'

# Export the model to ONNX format
torch.onnx.export(
    model,
    (dummy_pixel_values, dummy_input_ids),  # Tuple of dummy inputs
    onnx_model_path,  # Path to save the ONNX model
    export_params=True,  # Export trained parameter weights
    input_names=['pixel_values', 'input_ids'],  # Names for the input tensors
    output_names=['output_ids'],  # Names for the output tensors
    dynamic_axes={
        'pixel_values': {0: 'batch_size'},  # Allow batch size to vary
        'input_ids': {0: 'batch_size'},  # Allow batch size to vary
        'output_ids': {0: 'batch_size'}   # Allow batch size to vary
    }
)

print(f"Model exported to {onnx_model_path}")
