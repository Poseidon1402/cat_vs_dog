"""
Train Cat vs Dog Classification Model

This script trains a CNN model for binary classification of cat and dog images.
Supports GPU acceleration and various hyperparameter configurations.

Usage:
    python scripts/train.py --data_dir data/1/kagglecatsanddogs_3367a/PetImages --epochs 50
    python scripts/train.py --data_dir data/1/kagglecatsanddogs_3367a/PetImages --epochs 30 --batch_size 16 --learning_rate 0.0001
"""

import argparse
import os
import sys
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train Cat vs Dog Classification Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, 
                        default='data/1/kagglecatsanddogs_3367a/PetImages',
                        help='Path to training data directory containing Cat/ and Dog/ folders')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Directory to save trained model')
    parser.add_argument('--model_name', type=str, default='model.keras',
                        help='Output model filename')
    
    # Image parameters
    parser.add_argument('--image_size', type=int, default=128,
                        help='Image height and width (square images)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--validation_split', type=float, default=0.2,
                        help='Fraction of data to use for validation (0.0-1.0)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                        help='Early stopping patience (epochs)')
    parser.add_argument('--reduce_lr_patience', type=int, default=5,
                        help='Learning rate reduction patience (epochs)')
    
    # Data augmentation parameters
    parser.add_argument('--rotation_range', type=int, default=20,
                        help='Rotation range for data augmentation (degrees)')
    parser.add_argument('--shift_range', type=float, default=0.2,
                        help='Width/height shift range for augmentation')
    parser.add_argument('--zoom_range', type=float, default=0.2,
                        help='Zoom range for data augmentation')
    parser.add_argument('--no_augmentation', action='store_true',
                        help='Disable data augmentation')
    
    # Model architecture parameters
    parser.add_argument('--dropout_rate', type=float, default=0.5,
                        help='Dropout rate for dense layers')
    
    # GPU parameters
    parser.add_argument('--disable_gpu', action='store_true',
                        help='Disable GPU and use CPU only')
    parser.add_argument('--gpu_memory_growth', action='store_true', default=True,
                        help='Enable GPU memory growth (prevents OOM errors)')
    
    # Visualization
    parser.add_argument('--save_plots', action='store_true', default=True,
                        help='Save training history plots')
    parser.add_argument('--plot_dir', type=str, default='assets',
                        help='Directory to save plots')
    
    return parser.parse_args()


def configure_gpu(disable_gpu=False, memory_growth=True):
    """Configure GPU settings"""
    if disable_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        print("GPU disabled. Using CPU only.")
        return
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            if memory_growth:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ GPU Available: {len(gpus)} GPU(s) found")
            for gpu in gpus:
                print(f"  - {gpu}")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("✗ No GPU found. Training will use CPU.")


def create_data_generators(args):
    """Create training and validation data generators"""
    print("\n" + "="*60)
    print("CREATING DATA GENERATORS")
    print("="*60)
    
    if args.no_augmentation:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=args.validation_split
        )
        print("Data augmentation: DISABLED")
    else:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=args.validation_split,
            rotation_range=args.rotation_range,
            width_shift_range=args.shift_range,
            height_shift_range=args.shift_range,
            shear_range=args.shift_range,
            zoom_range=args.zoom_range,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        print("Data augmentation: ENABLED")
        print(f"  - Rotation range: ±{args.rotation_range}°")
        print(f"  - Shift range: ±{args.shift_range}")
        print(f"  - Zoom range: {args.zoom_range}")
        print(f"  - Horizontal flip: True")
    
    # Training generator
    train_generator = train_datagen.flow_from_directory(
        args.data_dir,
        target_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        class_mode='binary',
        subset='training',
        seed=42
    )
    
    # Validation generator (no augmentation)
    val_generator = train_datagen.flow_from_directory(
        args.data_dir,
        target_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        class_mode='binary',
        subset='validation',
        seed=42
    )
    
    print(f"\nTraining samples: {train_generator.samples}")
    print(f"Validation samples: {val_generator.samples}")
    print(f"Class indices: {train_generator.class_indices}")
    print(f"Batches per epoch (train): {len(train_generator)}")
    print(f"Batches per epoch (val): {len(val_generator)}")
    
    return train_generator, val_generator


def build_model(args):
    """Build CNN model architecture"""
    print("\n" + "="*60)
    print("BUILDING MODEL")
    print("="*60)
    
    input_shape = (args.image_size, args.image_size, 3)
    
    model = models.Sequential([
        # First convolutional block - 32 filters
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Second convolutional block - 64 filters
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Third convolutional block - 128 filters
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Fourth convolutional block - 256 filters
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Flatten and fully connected layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(args.dropout_rate),
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nModel Architecture:")
    model.summary()
    
    print(f"\nHyperparameters:")
    print(f"  - Learning rate: {args.learning_rate}")
    print(f"  - Dropout rate: {args.dropout_rate}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Epochs: {args.epochs}")
    
    return model


def create_callbacks(args):
    """Create training callbacks"""
    callbacks = []
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=args.early_stopping_patience,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stopping)
    
    # Learning rate reduction
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=args.reduce_lr_patience,
        min_lr=1e-7,
        verbose=1
    )
    callbacks.append(reduce_lr)
    
    # Model checkpoint
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, args.model_name)
    checkpoint = ModelCheckpoint(
        model_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    callbacks.append(checkpoint)
    
    print("\n" + "="*60)
    print("CALLBACKS CONFIGURED")
    print("="*60)
    print(f"Early stopping patience: {args.early_stopping_patience} epochs")
    print(f"Learning rate reduction patience: {args.reduce_lr_patience} epochs")
    print(f"Model checkpoint: {model_path}")
    
    return callbacks, model_path


def train_model(model, train_gen, val_gen, callbacks, args):
    """Train the model"""
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    history = model.fit(
        train_gen,
        epochs=args.epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    return history


def save_training_plots(history, args):
    """Save training history plots"""
    if not args.save_plots:
        return
    
    os.makedirs(args.plot_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(args.plot_dir, 'training_history.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Training plots saved to: {plot_path}")


def print_summary(history, model_path):
    """Print training summary"""
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    
    print(f"\n✓ Best model saved to: {model_path}")
    
    print(f"\nFinal Results:")
    print(f"  Training Accuracy:   {history.history['accuracy'][-1]:.4f}")
    print(f"  Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"  Training Loss:       {history.history['loss'][-1]:.4f}")
    print(f"  Validation Loss:     {history.history['val_loss'][-1]:.4f}")
    
    if len(history.history['val_accuracy']) > 1:
        best_val_acc = max(history.history['val_accuracy'])
        best_epoch = history.history['val_accuracy'].index(best_val_acc) + 1
        print(f"\nBest Validation Accuracy: {best_val_acc:.4f} (Epoch {best_epoch})")
    
    print("\n" + "="*60)


def main():
    """Main training function"""
    # Parse arguments
    args = parse_arguments()
    
    # Configure environment
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # Configure GPU
    configure_gpu(args.disable_gpu, args.gpu_memory_growth)
    
    # Create data generators
    train_gen, val_gen = create_data_generators(args)
    
    # Build model
    model = build_model(args)
    
    # Create callbacks
    callbacks, model_path = create_callbacks(args)
    
    # Train model
    history = train_model(model, train_gen, val_gen, callbacks, args)
    
    # Save plots
    save_training_plots(history, args)
    
    # Print summary
    print_summary(history, model_path)


if __name__ == '__main__':
    main()
