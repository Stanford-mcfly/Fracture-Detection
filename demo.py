import tensorflow as tf
import os
import argparse
import logging

def convert_keras_to_tflite(model_path, output_dir="tflite_models"):
    """
    Convert Keras (.h5) model to TensorFlow Lite format (.tflite)
    
    Args:
        model_path (str): Path to input Keras .h5 model
        output_dir (str): Output directory for TFLite models
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Validate input file
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if not model_path.endswith('.h5'):
            raise ValueError("Input file must be a .h5 Keras model")

        # Load Keras model
        logging.info(f"Loading Keras model from {model_path}")
        model = tf.keras.models.load_model(model_path)

        # Convert to TFLite
        logging.info("Converting to TFLite format...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Optional optimization (uncomment to enable)
        # converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()

        # Save converted model
        base_name = os.path.basename(model_path).replace('.h5', '')
        tflite_path = os.path.join(output_dir, f"{base_name}.tflite")
        
        logging.info(f"Saving TFLite model to {tflite_path}")
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)

        # Verify conversion
        if os.path.exists(tflite_path):
            logging.info(f"Successfully converted {model_path}")
            logging.info(f"Original size: {os.path.getsize(model_path)/1e6:.2f} MB")
            logging.info(f"TFLite size: {os.path.getsize(tflite_path)/1e6:.2f} MB")
        else:
            raise RuntimeError("Conversion failed - output file not created")

    except Exception as e:
        logging.error(f"Error converting {model_path}: {str(e)}")
        raise

def convert_directory(input_dir, output_dir="tflite_models"):
    """Convert all .h5 models in a directory"""
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"Input directory not found: {input_dir}")

    logging.info(f"Processing directory: {input_dir}")
    
    for file in os.listdir(input_dir):
        if file.endswith('.h5'):
            model_path = os.path.join(input_dir, file)
            convert_keras_to_tflite(model_path, output_dir)

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Set up command line arguments
    parser = argparse.ArgumentParser(
        description='Convert Keras models to TensorFlow Lite format'
    )
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Input path (file or directory)'
    )
    parser.add_argument(
        '-o', '--output',
        default='tflite_models',
        help='Output directory for TFLite models'
    )
    
    args = parser.parse_args()

    try:
        if os.path.isdir(args.input):
            convert_directory(args.input, args.output)
        else:
            convert_keras_to_tflite(args.input, args.output)
            
        logging.info("Conversion process completed")
        
    except Exception as e:
        logging.error(f"Fatal error during conversion: {str(e)}")
        exit(1)


        