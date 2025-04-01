import tensorflow as tf
import numpy as np
import os

def representative_dataset():
    # Generate sample input matching your model's expected shape
    for _ in range(100):
        # For grayscale models
        yield [np.random.rand(1, 224, 224, 1).astype(np.float32)]
        
        # For RGB models (uncomment below)
        # yield [np.random.rand(1, 224, 224, 3).astype(np.float32)]

def convert_model(h5_path, tflite_path):
    # Load model and verify input shape
    model = tf.keras.models.load_model(h5_path)
    print(f"Model input shape: {model.input_shape}")
    
    # Configure converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    
    # Set input type based on your model
    converter.inference_input_type = tf.uint8  # or tf.float32
    converter.inference_output_type = tf.uint8  # or tf.float32
    
    # Convert and save
    tflite_model = converter.convert()
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Saved quantized model to {tflite_path}")

if __name__ == "__main__":
    models = {
        "BodyParts": "ResNet50_BodyParts.h5",
        "HandFrac": "ResNet50_Hand_frac.h5",
        "ElbowFrac": "ResNet50_Elbow_frac.h5",
        "ShoulderFrac": "ResNet50_Shoulder_frac.h5"
    }
    
    for name, h5_path in models.items():
        output_path = f"tflite_models/{name}_quant.tflite"
        convert_model(h5_path, output_path)