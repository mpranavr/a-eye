import cv2
import numpy as np
from PIL import Image
from transformers import ViTImageProcessor, VisionEncoderDecoderModel, GPT2Tokenizer
from one_img_text_main import image_processing


def capture_image():
    cap = cv2.VideoCapture(0)  # Open the default camera (0 is usually the default webcam)
    if not cap.isOpened():
        raise Exception("Could not open video device")

    # Capture a single frame
    ret, frame = cap.read()
    if not ret:
        raise Exception("Failed to capture image from video feed")

    # Convert the captured frame to RGB and then to a Pillow Image
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)

    # Release the video capture object and close the window
    cap.release()
    cv2.destroyAllWindows()

    return image

if __name__ == "__main__":
    try:
        # Capture an image from the camera
        image = capture_image()
        # Generate the caption for the captured image
        caption = image_processing(image)
        print("Generated Caption:", caption)
        
    except Exception as e:
        print("Error:", e)



