# A Eye (Preception through AI)
Improving the perception of the world for the blind using AI models for image recognition, object detection and face detection etc.

# A Eye (Preception through AI)

## Things used in this project

### Hardware components
Webcam, Logitech® HD Pro ×	1	
Minisforum Venus UM790 Pro with AMD Ryzen™ 9	×	1	
Generic Interface Devices (Keyboard, Mouse, Etc) ×	1	
### Software apps and online services
AMD Ryzen AI Software
Hugging Face
Windows 10	
Microsoft Windows 10
Microsoft Visual Studio 201
nlpconnect/vit-gpt2-image-captioning model
microsoft/speecht5_tts model


# Story
## 1. Project Description:


### 1.1. Problem Statement

While existing technologies have significantly improved navigation and obstacle detection for the blind, they primarily focus on these aspects alone. Our goal is to extend this technology to include not just detection but also recognition of objects, faces, and images. This broader application of AI aims to enhance the overall experience of visually impaired individuals by providing more detailed and meaningful information about their surroundings.

### 1.2. Proposed Solution

Our project seeks to enhance spatial awareness for the blind using advanced AI models for image, object, and face recognition. Unlike current solutions that primarily focus on navigation, our approach leverages AI to interpret and convey detailed visual information, aiming to provide users with a richer and more immersive understanding of their surroundings. By interpreting and conveying detailed visual information, we aim to provide users with an immersive experience that improves their overall perception of the world, addressing a critical need beyond mere navigation.

## 2. Project Features
The planned project includes four main features: image recognition, object recognition, face recognition, text to speech conversion and GPS tracking.

### 2.1. Image and Object Recognition:
These are achieved through an image-to-text conversion model that analyzes the surroundings using the user's video capture device. This model generates suitable captions to describe the images, serving the dual purpose of image and object recognition within a single implementation.

### 2.2. Face Recognition:

Face recognition functionality identifies and distinguishes different faces in the user’s surroundings. This feature can recognize and relay information about known individuals, enhancing social interactions and personal awareness. If any unknown individuals are encountered, the user is prompted to designate a name for that individual, and their facial encoding is saved to the user directory.

### 2.3. Text-to-Speech Conversion:

To relay the information to the user, we use text-to-speech (TTS) model. This converts the generated text descriptions and recognitions into audible speech, leveraging the user's sound perception to convey visual information effectively.

### 2.4. GPS Tracking:

An assisted GPS helps navigate through the streets and provide directional cues to the user. This feature supports efficient and accurate navigation, ensuring the user can safely and effectively reach their destinations.



## 3. Project Implementation
The project has been implemented using a desktop video capture device, which captures image frames for image captioning. Each model has been individually implemented to fulfill the project's objectives effectively.

3.1. Image and Object Recognition Model:

The nlpconnect/vit-gpt2-image-captioning model from Hugging Face is employed to generate descriptive captions for the captured images. Here’s a detailed explanation of the workflow:

### 3.1.1. Model Selection and Functionality

Model Choice: The nlpconnect/vit-gpt2-image-captioning model is chosen for its ability to generate detailed and accurate captions that describe the content of an image. It combines Vision Transformer (ViT) for image feature extraction and GPT-2 for generating natural language descriptions.

### 3.1.2. Conversion to ONNX Format

ONNX Format: Open Neural Network Exchange (ONNX) is an open-source format for AI models, which makes them portable and interoperable across different platforms. Converting the model to ONNX format allows for deployment in environments that support ONNX, ensuring better compatibility and perf
Conversion Process: The model is converted from its original format to ONNX using tools and libraries provided by Hugging Face and ONNX. This step involves exporting the model's architecture and weights into a standardized ONNX file, which can then be used for further optimization and deployment.

### 3.1.3. Optimization through Preprocessing

Preprocessing: Once in ONNX format, the model undergoes preprocessing to optimize its performance. This includes:

Input Normalization: Adjusting the input image data to match the expected format of the model.
Resizing: Ensuring that all input images are resized to a consistent dimension suitable for the model (e.g., 224x224 pixels).
Data Augmentation: Applying techniques to improve model robustness and performance, such as random cropping, flipping, and color adjustments
3.1.4. Quantization Using Vitis AI Quantizer

Quantization: This process involves converting the model's weights and activations from floating-point precision (32-bit) to a lower precision (e.g., 8-bit integers). Quantization significantly reduces the model size and improves inference speed without substantial loss in accuracy

Vitis AI Quantizer: Vitis AI Quantizer is a tool provided by Xilinx for optimizing AI models for deployment on their hardware accelerators. It supports various quantization techniques, ensuring that the model remains efficient and accurate.

Calibration Data: To maintain accuracy during quantization, suitable calibration data is used. This data represents typical inputs the model will process, ensuring that the quantized model performs well on real-world data. Calibration data helps in fine-tuning the quantization parameters, minimizing the impact on model performance.

By following this detailed workflow, the project ensures that the nlpconnect/vit-gpt2-image-captioning model operates efficiently and accurately, providing high-quality image and object recognition for visually impaired users.


## 3.2. Face Recognition Model:

The face_recognition Python package is utilized to identify and differentiate faces in the video frames. The system notifies the user of recognized individuals by name, and for unknown faces, it prompts the user to assign a name, storing the facial encoding for future identification. This feature enhances social interactions and personal awareness for the user.

## 3.3. Text-to-Speech Conversion Model:

The microsoft/speecht5_tts model from Hugging Face is used to convert the generated text captions into audible speech. This allows the user to receive verbal descriptions of their surroundings, making the information accessible and easy to understand.

## 3.4 User Interaction and Feedback:

The system is designed with user interaction in mind. The face recognition model prompts the user to assign names to unknown faces, allowing for personalized and continually improving recognition. The text-to-speech conversion ensures that the information is relayed in a clear and understandable manner, leveraging the user's auditory perception.

## 3.5. Desktop Video Capture Device:

The desktop video capture device continuously records the user's surroundings, providing real-time image frames for processing. These frames are used as input for the image captioning model, ensuring that the user receives up-to-date information about their environment.

## 3.6. Exclusion of GPS Tracking

Due to hardware limitations, GPS tracking has not been implemented in the current version of the project. This decision was made to ensure that the focus remains on optimizing the core functionalities of image, object, and face recognition, as well as text-to-speech conversion.

By combining these technologies, the project provides a comprehensive solution that enhances the spatial awareness of visually impaired users, enabling them to perceive and interact with their environment more effectively.

# 4. Conclusion:
By integrating advanced AI technologies, our project significantly enhances spatial awareness for visually impaired individuals. Through the combined use of image and object recognition, face recognition, and text-to-speech conversion, we provide a comprehensive solution that goes beyond traditional navigation aids. This project empowers users to gain a deeper understanding of their surroundings, fostering greater independence and social interaction. By focusing on improving perception and interaction with the environment, we aim to enrich the lives of visually impaired users, offering them a more immersive and informative experience of the world.

# 5. Future Scope
The future scope of this project is vast, with numerous opportunities for enhancement and expansion:

## 5.1. User Interaction:

In the current project, user interaction is limited. This limitation can be addressed by exploring more diverse ways for users to interact with the system, such as integrating voice commands, touch interfaces, and haptic feedback. These enhancements would provide a more intuitive and accessible user experience, allowing visually impaired users to engage with the system more effectively and

## 5.2. Enhanced Spatial Awareness
Currently, spatial awareness is primarily provided through the image and object recognition features of the project. To further enhance this capability, we plan to implement a wearable band equipped with vibrating motors. This band will provide directional cues by vibrating motors corresponding to the direction the user needs to be aware of, such as north. For instance, if the user needs to know which way is north, only the motors aligned with that direction will vibrate, giving the user tactile feedback about their orientation.

This tactile directional sense is a simple yet effective solution, but it will be expanded in future developments to integrate GPS and advanced object detection. With GPS, the system will be able to provide more precise location and directional information, helping users navigate and recognize various places and objects around them. The vibrations will serve as both informative cues and warnings, based on the combined analysis from image recognition and GPS positioning, thereby significantly improving the user's spatial awareness and overall navigation experience.

## 5.3. Smart Glasses for Video Capture

In the future, integrating video capture into smart glasses would be ideal for this project. Smart glasses can offer a more seamless and discreet way to continuously capture and analyze visual information, enhancing the overall effectiveness of our spatial awareness technology. This wearable solution would provide users with real-time, hands-free video capture, making the system more convenient and intuitive for daily use.


## 5.4. Expanded Recognition Capabilities:
Extend recognition capabilities to include more categories, such as landmarks, text, and animals, providing a richer set of information to the user. Implement real-time object tracking to provide continuous updates about moving objects in the user's environment.


## 5.5. Collaboration with Accessibility Organizations:
Partner with organizations that support visually impaired individuals to ensure the project meets the community's needs and standards. Engage in user testing and community workshops to gather insights and validate the system's effectiveness.


## 5.6. Robust Security and Privacy Measures:
Develop secure methods for storing and handling personal data, such as facial encodings, to ensure user privacy. Implement encryption and secure access protocols to protect the system from unauthorized access and data breaches.

By pursuing these future developments, the project can continue to evolve and offer even greater benefits to visually impaired individuals, ultimately enhancing their independence and quality of life.

