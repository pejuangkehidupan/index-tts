import os
import torch
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def torch_to_onnx(model, dummy_input, onnx_path, input_names, output_names, dynamic_axes, verbose=False):
    """Converts a PyTorch model to an ONNX file."""
    # Ensure the model is in evaluation mode
    model.eval()

    # Move model and dummy input to the correct device
    device = next(model.parameters()).device

    # Handle single tensor input
    if isinstance(dummy_input, torch.Tensor):
        dummy_input = dummy_input.to(device)
    # Handle tuple of tensor inputs
    elif isinstance(dummy_input, tuple):
        dummy_input = tuple(d.to(device) for d in dummy_input)
    else:
        raise TypeError(f"Unsupported dummy_input type: {type(dummy_input)}")

    print(f"Exporting model to {onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,  # Using a recent opset version
        verbose=verbose,
    )
    print(f"Model exported to {onnx_path}")


def build_engine(onnx_path, engine_path, use_fp16=False, max_workspace_size=None, dynamic_shapes=None):
    """Builds a TensorRT engine from an ONNX file."""
    if os.path.exists(engine_path):
        print(f"Engine file {engine_path} already exists. Skipping build.")
        # still need to load it
        return load_engine(engine_path)

    print(f"Building TensorRT engine for {onnx_path}...")
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse the ONNX file
    with open(onnx_path, "rb") as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    config = builder.create_builder_config()

    # Set workspace size
    if max_workspace_size:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)
    else:
        # Default to 1GB if not specified, which is a reasonable default
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    # Enable FP16 if requested
    if use_fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("FP16 mode enabled.")
    else:
        print("FP16 not supported or not requested. Using FP32.")

    if dynamic_shapes:
        profile = builder.create_optimization_profile()
        for name, min_shape, opt_shape, max_shape in dynamic_shapes:
            profile.set_shape(name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)

    # Build the engine
    serialized_engine = builder.build_serialized_network(network, config)
    if not serialized_engine:
        print("ERROR: Failed to build the TensorRT engine.")
        return None

    # Save the engine to a file
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    print(f"TensorRT engine built and saved to {engine_path}")

    # deserialize and return the engine
    with trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(serialized_engine)

    return engine


def load_engine(engine_path, device="cuda"):
    """Loads a TensorRT engine from a file."""
    if not os.path.exists(engine_path):
        print(f"Engine file {engine_path} not found.")
        return None

    print(f"Loading TensorRT engine from {engine_path}...")
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    print("TensorRT engine loaded.")
    return engine
