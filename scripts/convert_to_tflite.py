"""
Convert Keras Model to TensorFlow Lite

This script converts a trained Keras model to TensorFlow Lite format with optional quantization.
Supports standard float32 and quantized int8 conversions.

Usage:
    python scripts/convert_to_tflite.py --model models/model.keras
    python scripts/convert_to_tflite.py --model models/model.keras --quantize --test_images test_images
"""

import argparse
import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image
import time
import matplotlib.pyplot as plt


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Convert Keras Model to TensorFlow Lite',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model paths
    parser.add_argument('--model', type=str, required=True,
                        help='Path to input Keras model (.keras or .h5)')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Directory to save TFLite models')
    parser.add_argument('--output_name', type=str, default='model',
                        help='Base name for output TFLite models (without extension)')
    
    # Conversion options
    parser.add_argument('--quantize', action='store_true',
                        help='Create quantized int8 model in addition to standard')
    parser.add_argument('--quantize_only', action='store_true',
                        help='Only create quantized model (skip standard conversion)')
    
    # Quantization parameters
    parser.add_argument('--test_images', type=str, default='test_images',
                        help='Directory with test images for quantization calibration')
    parser.add_argument('--num_calibration_samples', type=int, default=100,
                        help='Number of images to use for quantization calibration')
    
    # Image parameters
    parser.add_argument('--image_size', type=int, default=128,
                        help='Image size (height and width)')
    
    # Benchmarking
    parser.add_argument('--benchmark', action='store_true',
                        help='Run inference benchmark on converted models')
    parser.add_argument('--benchmark_runs', type=int, default=100,
                        help='Number of benchmark iterations')
    parser.add_argument('--test_image', type=str, default=None,
                        help='Path to single test image for benchmarking')
    
    # Visualization
    parser.add_argument('--save_plots', action='store_true', default=True,
                        help='Save comparison plots')
    parser.add_argument('--plot_dir', type=str, default='assets',
                        help='Directory to save plots')
    
    return parser.parse_args()


def load_keras_model(model_path):
    """Load Keras model"""
    print("\n" + "="*60)
    print("LOADING KERAS MODEL")
    print("="*60)
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)
    
    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"✓ Model loaded successfully")
    print(f"Model size: {model_size_mb:.2f} MB")
    
    return model, model_size_mb


def convert_to_tflite_standard(model, output_path):
    """Convert to standard TFLite (float32)"""
    print("\n" + "="*60)
    print("CONVERTING TO TFLITE (STANDARD)")
    print("="*60)
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    print("Converting...")
    tflite_model = converter.convert()
    
    # Save model
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    model_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✓ Standard TFLite model saved: {output_path}")
    print(f"Model size: {model_size_mb:.2f} MB")
    
    return model_size_mb


def representative_dataset_generator(test_images_dir, num_samples, image_size):
    """Generator for representative dataset (quantization calibration)"""
    def generator():
        if not os.path.exists(test_images_dir):
            print(f"Warning: Test images directory not found: {test_images_dir}")
            print("Using random data for calibration (not recommended)")
            for _ in range(num_samples):
                yield [np.random.rand(1, image_size, image_size, 3).astype(np.float32)]
            return
        
        image_files = [f for f in os.listdir(test_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(image_files) < num_samples:
            print(f"Warning: Found only {len(image_files)} images, using all of them")
            num_samples = len(image_files)
        
        image_files = image_files[:num_samples]
        
        for img_name in image_files:
            img_path = os.path.join(test_images_dir, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize((image_size, image_size))
                img_array = np.array(img, dtype=np.float32)
                img_array = img_array / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                yield [img_array]
            except Exception as e:
                print(f"Error loading {img_name}: {e}")
                continue
    
    return generator


def convert_to_tflite_quantized(model, output_path, test_images_dir, num_samples, image_size):
    """Convert to quantized TFLite (int8)"""
    print("\n" + "="*60)
    print("CONVERTING TO TFLITE (QUANTIZED)")
    print("="*60)
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Representative dataset for quantization
    print(f"Calibrating with {num_samples} images from: {test_images_dir}")
    converter.representative_dataset = representative_dataset_generator(
        test_images_dir, num_samples, image_size
    )
    
    # Full integer quantization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    
    print("Converting...")
    tflite_quantized_model = converter.convert()
    
    # Save model
    with open(output_path, 'wb') as f:
        f.write(tflite_quantized_model)
    
    model_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✓ Quantized TFLite model saved: {output_path}")
    print(f"Model size: {model_size_mb:.2f} MB")
    
    return model_size_mb


def benchmark_inference(keras_model, tflite_path, tflite_quantized_path, test_image_path, image_size, num_runs):
    """Benchmark inference speed"""
    print("\n" + "="*60)
    print("BENCHMARKING INFERENCE SPEED")
    print("="*60)
    
    # Load test image
    if not os.path.exists(test_image_path):
        print(f"Test image not found: {test_image_path}")
        print("Skipping benchmark")
        return None
    
    img = Image.open(test_image_path).convert('RGB')
    img = img.resize((image_size, image_size))
    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    print(f"Test image: {test_image_path}")
    print(f"Running {num_runs} iterations per model...\n")
    
    # Benchmark Keras
    keras_times = []
    for _ in range(num_runs):
        start = time.time()
        keras_model.predict(img_array, verbose=0)
        keras_times.append((time.time() - start) * 1000)
    keras_avg = np.mean(keras_times)
    
    # Benchmark TFLite Standard
    tflite_times = []
    if os.path.exists(tflite_path):
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        for _ in range(num_runs):
            start = time.time()
            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            tflite_times.append((time.time() - start) * 1000)
        tflite_avg = np.mean(tflite_times)
    else:
        tflite_avg = None
    
    # Benchmark TFLite Quantized
    tflite_quant_times = []
    if tflite_quantized_path and os.path.exists(tflite_quantized_path):
        interpreter_quant = tf.lite.Interpreter(model_path=tflite_quantized_path)
        interpreter_quant.allocate_tensors()
        input_details_quant = interpreter_quant.get_input_details()
        
        img_uint8 = (img_array * 255).astype(np.uint8)
        
        for _ in range(num_runs):
            start = time.time()
            interpreter_quant.set_tensor(input_details_quant[0]['index'], img_uint8)
            interpreter_quant.invoke()
            tflite_quant_times.append((time.time() - start) * 1000)
        tflite_quant_avg = np.mean(tflite_quant_times)
    else:
        tflite_quant_avg = None
    
    # Print results
    print("Benchmark Results:")
    print(f"  Keras Model:           {keras_avg:.2f} ms (baseline)")
    if tflite_avg:
        print(f"  TFLite Standard:       {tflite_avg:.2f} ms ({keras_avg/tflite_avg:.2f}x faster)")
    if tflite_quant_avg:
        print(f"  TFLite Quantized:      {tflite_quant_avg:.2f} ms ({keras_avg/tflite_quant_avg:.2f}x faster)")
    
    return {
        'keras': keras_avg,
        'tflite': tflite_avg,
        'tflite_quantized': tflite_quant_avg
    }


def save_comparison_plots(keras_size, tflite_size, tflite_quant_size, benchmark_results, output_dir):
    """Save size and speed comparison plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data
    model_names = []
    model_sizes = []
    
    if keras_size:
        model_names.append('Keras\n(Original)')
        model_sizes.append(keras_size)
    if tflite_size:
        model_names.append('TFLite\n(Standard)')
        model_sizes.append(tflite_size)
    if tflite_quant_size:
        model_names.append('TFLite\n(Quantized)')
        model_sizes.append(tflite_quant_size)
    
    # Create figure
    if benchmark_results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    
    # Size comparison
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1'][:len(model_names)]
    bars = ax1.bar(model_names, model_sizes, color=colors, edgecolor='black', linewidth=1.5)
    
    for bar, size in zip(bars, model_sizes):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{size:.2f} MB',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax1.set_ylabel('Model Size (MB)', fontsize=12, fontweight='bold')
    ax1.set_title('Model Size Comparison', fontsize=14, fontweight='bold', pad=20)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Speed comparison
    if benchmark_results:
        times = []
        if benchmark_results['keras']:
            times.append(benchmark_results['keras'])
        if benchmark_results['tflite']:
            times.append(benchmark_results['tflite'])
        if benchmark_results['tflite_quantized']:
            times.append(benchmark_results['tflite_quantized'])
        
        bars2 = ax2.bar(model_names[:len(times)], times, color=colors[:len(times)], 
                        edgecolor='black', linewidth=1.5)
        
        for bar, time_val in zip(bars2, times):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time_val:.2f} ms',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax2.set_ylabel('Inference Time (ms)', fontsize=11, fontweight='bold')
        ax2.set_title('Average Inference Time', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'conversion_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Comparison plots saved to: {plot_path}")


def print_summary(keras_size, tflite_size, tflite_quant_size, tflite_path, tflite_quant_path):
    """Print conversion summary"""
    print("\n" + "="*60)
    print("CONVERSION COMPLETE")
    print("="*60)
    
    print("\n📦 Output Files:")
    if tflite_path and os.path.exists(tflite_path):
        size_reduction = ((keras_size - tflite_size) / keras_size) * 100
        print(f"  Standard TFLite:  {tflite_path}")
        print(f"                    {tflite_size:.2f} MB ({size_reduction:.1f}% reduction)")
    
    if tflite_quant_path and os.path.exists(tflite_quant_path):
        size_reduction = ((keras_size - tflite_quant_size) / keras_size) * 100
        print(f"  Quantized TFLite: {tflite_quant_path}")
        print(f"                    {tflite_quant_size:.2f} MB ({size_reduction:.1f}% reduction)")
    
    print("\n💡 Deployment Recommendations:")
    print("  • Standard TFLite: Use when maximum accuracy is critical")
    print("  • Quantized TFLite: Use for mobile/embedded devices (2-4x faster)")
    
    print("\n" + "="*60)


def main():
    """Main conversion function"""
    args = parse_arguments()
    
    # Validate arguments
    if args.quantize_only and not args.quantize:
        args.quantize = True
    
    # Load Keras model
    keras_model, keras_size = load_keras_model(args.model)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Convert to standard TFLite
    tflite_path = None
    tflite_size = None
    if not args.quantize_only:
        tflite_path = os.path.join(args.output_dir, f'{args.output_name}.tflite')
        tflite_size = convert_to_tflite_standard(keras_model, tflite_path)
    
    # Convert to quantized TFLite
    tflite_quant_path = None
    tflite_quant_size = None
    if args.quantize or args.quantize_only:
        tflite_quant_path = os.path.join(args.output_dir, f'{args.output_name}_quantized.tflite')
        tflite_quant_size = convert_to_tflite_quantized(
            keras_model, tflite_quant_path, args.test_images,
            args.num_calibration_samples, args.image_size
        )
    
    # Benchmark if requested
    benchmark_results = None
    if args.benchmark:
        if args.test_image is None:
            # Try to find a test image
            if os.path.exists(args.test_images):
                images = [f for f in os.listdir(args.test_images) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if images:
                    args.test_image = os.path.join(args.test_images, images[0])
        
        if args.test_image:
            benchmark_results = benchmark_inference(
                keras_model, tflite_path, tflite_quant_path,
                args.test_image, args.image_size, args.benchmark_runs
            )
    
    # Save plots
    if args.save_plots:
        save_comparison_plots(keras_size, tflite_size, tflite_quant_size, 
                            benchmark_results, args.plot_dir)
    
    # Print summary
    print_summary(keras_size, tflite_size, tflite_quant_size, tflite_path, tflite_quant_path)


if __name__ == '__main__':
    main()
