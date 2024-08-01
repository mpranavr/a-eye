import vai_q_onnx
import numpy as np
import os
from PIL import Image
from torchvision import transforms
from onnxruntime.quantization import CalibrationDataReader

def preprocess_image(image_path, target_size=(224, 224)):
    # Define a transformation pipeline
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension

class ImageCalibrationDataReader(CalibrationDataReader):
    def __init__(self, image_folder, batch_size=10, target_size=(224, 224)):
        self.image_folder = image_folder
        self.batch_size = batch_size
        self.target_size = target_size
        self.image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.png')]
        self.num_batches = len(self.image_files) // batch_size
        self.batch_index = 0
    
    def create_batch(self):
        batch_images = []
        batch_input_ids = []  # Assuming you also need input_ids

        # Load a batch of images
        start_index = self.batch_index * self.batch_size
        end_index = min(start_index + self.batch_size, len(self.image_files))
        
        for i in range(start_index, end_index):
            image_path = os.path.join(self.image_folder, self.image_files[i])
            preprocessed_image = preprocess_image(image_path, self.target_size)
            batch_images.append(preprocessed_image)
            # Assuming you need a dummy input for input_ids
            batch_input_ids.append(np.zeros((1, 1), dtype=np.int64))  # Adjust as needed
        
        if batch_images:
            batch_data = {
                'pixel_values': np.concatenate(batch_images, axis=0),
                'input_ids': np.concatenate(batch_input_ids, axis=0)  # Assuming input_ids needs to be concatenated
            }
            return batch_data
        return None
    
    def get_next(self):
        if self.batch_index < self.num_batches:
            batch_data = self.create_batch()
            self.batch_index += 1
            return batch_data
        else:
            return None

image_folder = "etc/programs/image-to-text/calib_data/unsplash-images-collection"  
calibration_data_reader = ImageCalibrationDataReader(image_folder=image_folder, batch_size=10)
files = os.listdir(image_folder)
print(files)

# Define the paths for the input and output models
model_input = "etc/programs/image-to-text/preprocess_float_model.onnx"
model_output = "etc/programs/image-to-text/img_to_text_quant_model.onnx"

# **Optional: Check model compatibility**
# Try running inference on the original model without quantization to verify it works as expected.

print("Quantization started")

try:
  # Quantize the model
  vai_q_onnx.quantize_static(
      model_input,
      model_output,
      calibration_data_reader=calibration_data_reader,
      quant_format=vai_q_onnx.QuantFormat.QDQ,
      calibrate_method=vai_q_onnx.CalibrationMethod.MinMax,
      activation_type=vai_q_onnx.QuantType.QInt8,
      weight_type=vai_q_onnx.QuantType.QInt8,
  )
except RuntimeError as e:
  # Handle potential broadcasting error or other exceptions
  print(f"Quantization failed: {e}")
  # Consider using an alternative quantization library or investigating the model structure.

print(f"Quantized model saved to: {model_output}")
