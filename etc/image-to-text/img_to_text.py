from PIL import Image
import numpy as np
import torch
import onnxruntime
from transformers import GPT2Tokenizer, ViTImageProcessor

# Initialize tokenizer and model path
tokenizer = GPT2Tokenizer.from_pretrained(
    'etc/programs/models/image to text/nlpconnectvit-gpt2-image-captioning',
    
)

feature_extractor = ViTImageProcessor.from_pretrained('nlpconnect/vit-gpt2-image-captioning')

# Create ONNX Runtime session
session = onnxruntime.InferenceSession(
    'etc/programs/image-to-text/image_to_text.onnx',
    providers=["VitisAIExecutionProvider"],
    provider_options=[{"config_file": "etc/programs/supp_files/vaip_config.json"}]
)

# Print model input details
for input in session.get_inputs():
    print(f"Input name: {input.name}, Shape: {input.shape}, Type: {input.type}")

for output in session.get_outputs():
    output_names = [output.name for output in session.get_outputs()]
    print(f"Output name: {output.name}, Shape: {output.shape}, Type: {output.type}")

def preprocess_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        return image
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def predict_step(image_paths):
    images = [preprocess_image(image_path) for image_path in image_paths]
    if any(img is None for img in images):
        raise ValueError("One or more images could not be processed.")

    # Preprocess images to get pixel values
    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values_np = pixel_values.detach().cpu().numpy().astype(np.float32)

    # Prepare initial input_ids for the decoder
    input_ids = tokenizer("<|endoftext|>", return_tensors="pt").input_ids
    input_ids_np = input_ids.detach().cpu().numpy().astype(np.int64)

    # Prepare input feed for ONNX Runtime
    input_feed = {}
    input_names = [input.name for input in session.get_inputs()]

    if len(input_names) == 1:
        input_feed[input_names[0]] = pixel_values_np
    elif len(input_names) == 2:
        input_feed[input_names[0]] = pixel_values_np
        input_feed[input_names[1]] = input_ids_np
    else:
        raise ValueError("Unsupported number of input tensors")

    # Run inference
    try:
        result = session.run(output_names,input_feed)
        
        output_ids=result[0]
        
        # Debug: Print the type and content of output_ids
        print(f"Type of output_ids: {type(output_ids)}")
        print(f"Content of output_ids: {output_ids}")

        # Ensure output_ids is a list or array
        if isinstance(output_ids, (np.ndarray, torch.Tensor)):
            output_ids = output_ids.flatten().tolist()
        elif isinstance(output_ids, int):
            output_ids = [output_ids]
        elif not isinstance(output_ids, list):
            raise ValueError("Unsupported output_ids format")

        # Clean output_ids by removing None values and ensuring they are integers
        cleaned_output_ids = [int(abs(i)) for i in output_ids if isinstance(i, (int, float, np.integer, torch.Tensor)) and i is not None]

        # Check if cleaned_output_ids is empty
        if len(cleaned_output_ids) == 0:
            raise ValueError("Output IDs are empty or contain only non-integer values")
        print(f"Input ids: {input_ids}")
        print(f"pixel_values: {pixel_values}")
        # Decode the list of integers to get the caption
        preds = tokenizer.decode(cleaned_output_ids, skip_special_tokens=True)

        # Debug: Print the decoded prediction
        print(f"Decoded prediction: {preds}")

        preds = preds.strip()
        return preds

    except Exception as e:
        print(f"Error during inference or decoding: {e}")
        return None

# Example usage with specified image path
caption = predict_step(['etc/programs/calib_data/unsplash-images-collection/photo-1639046033583-390653f5c09a.jpg'])
print("Generated Caption:", caption)
