import onnxruntime
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, ViTImageProcessor
import os





# Process image
def image_processing(image):


    model_path = 'etc/programs/image-to-text/image_to_text.onnx'

    # Load the ONNX model
    session = onnxruntime.InferenceSession(
        model_path,
        providers=["VitisAIExecutionProvider"],
        provider_options=[{"config_file": "etc/programs/supp_files/vaip_config.json"}]
    )
    
    tokenizer = GPT2Tokenizer.from_pretrained(
        'etc/programs/models/image to text/nlpconnectvit-gpt2-image-captioning'
    )

    feature_extractor = ViTImageProcessor.from_pretrained('nlpconnect/vit-gpt2-image-captioning')


    if image is None:
        raise ValueError("Image could not be processed.")

    # Preprocess image to get pixel values
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
    pixel_values_np = pixel_values.detach().cpu().numpy().astype(np.float32)

    # Beam search settings
    max_length = 16
    num_beams = 4

    # Prepare initial input_ids for the decoder
    input_ids = tokenizer("<|endoftext|>", return_tensors="pt").input_ids
    input_ids_np = input_ids.detach().cpu().numpy().astype(np.int64)

    # Prepare input feed for ONNX Runtime
    input_names = [input.name for input in session.get_inputs()]
    input_feed = {}

    if len(input_names) == 1:
        input_feed[input_names[0]] = pixel_values_np
    elif len(input_names) == 2:
        input_feed[input_names[0]] = pixel_values_np
        input_feed[input_names[1]] = input_ids_np
    else:
        raise ValueError("Unsupported number of input tensors")

    output_names = [output.name for output in session.get_outputs()]

    # Initialize variables for beam search
    sequences = [[tokenizer.bos_token_id]]
    scores = [0.0]
    finished_sequences = []

    for _ in range(max_length):
        new_sequences = []
        new_scores = []
        
        for seq, score in zip(sequences, scores):
            input_ids_np = np.array(seq).reshape(1, -1).astype(np.int64)
            input_feed[input_names[1]] = input_ids_np
            print(input_ids_np)
            # Run inference
            outputs = session.run([output_names[0]], input_feed)
            logits = outputs[0]
            
            # Convert logits to log probabilities
            log_probs = F.log_softmax(torch.tensor(logits[:, -1, :]), dim=-1).numpy()
            
            # Apply beam search
            top_k_indices = np.argsort(log_probs[0])[-num_beams:]
            for idx in top_k_indices:
                new_seq = seq + [int(idx)]
                new_sequences.append(new_seq)
                new_scores.append(score + log_probs[0, idx])
        
        # Select top num_beams sequences
        top_indices = np.argsort(new_scores)[-num_beams:]
        sequences = [new_sequences[i] for i in top_indices]
        scores = [new_scores[i] for i in top_indices]
        
        # Check if any sequences have reached the end token
        finished_sequences.extend([seq for seq in sequences if seq[-1] == tokenizer.eos_token_id])
        sequences = [seq for seq in sequences if seq[-1] != tokenizer.eos_token_id]

        if not sequences:
            break

    # Decode and print the best sequence
    best_sequence = finished_sequences[0] if finished_sequences else sequences[0]
    decoded_output = tokenizer.decode(best_sequence, skip_special_tokens=True)
    print("Decoded output:", decoded_output)
    return decoded_output

