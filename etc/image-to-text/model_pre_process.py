from onnxruntime.quantization import shape_inference

shape_inference.quant_pre_process(
    input_model_path='etc/programs/image-to-text/image_to_text.onnx',
    output_model_path='etc/programs/image-to-text/preprocess_float_model.onnx',
    skip_optimization=False,
    skip_onnx_shape=False,
    skip_symbolic_shape=True,
    auto_merge=False,
    int_max=2**31 - 1,
    guess_output_rank=False,
    verbose=3,
    save_as_external_data=False,
    all_tensors_to_one_file=False,
    external_data_location="./",
    external_data_size_threshold=1024
)
print("Shape inference and preprocessing completed successfully.")