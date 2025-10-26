# Cat vs Dog Image Classification 🐱🐶

A deep learning project that classifies images as either cats or dogs using Convolutional Neural Networks (CNN) with TensorFlow/Keras.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Files Description](#files-description)

## 🎯 Project Overview

This project implements a binary image classifier to distinguish between cat and dog images. The model uses a custom CNN architecture built with TensorFlow/Keras, achieving high accuracy on the test dataset.

**Key Features:**
- Custom CNN architecture with batch normalization
- Data augmentation for better generalization
- GPU support for faster training
- Comprehensive benchmarking with visualizations
- Excel export with embedded images
- Model checkpointing and early stopping

## 📊 Dataset

### Training Dataset
**Source:** Kaggle Cats and Dogs Dataset  
**Location:** `data/1/kagglecatsanddogs_3367a/PetImages/`  
**Structure:**
```
PetImages/
├── Cat/        # ~12,500 cat images
└── Dog/        # ~12,500 dog images
```

**Total Training Images:** ~25,000 images  
**Format:** JPG  
**Original Source:** Microsoft Research

### Test Dataset
**Source:** [Kaggle Dogs vs Cats Competition](https://www.kaggle.com/c/dogs-vs-cats/data?select=sampleSubmission.csv)  
**Location:** `test_images/`  
**Total Test Images:** 12,500 unlabeled images  
**Format:** JPG  
**Naming:** Sequential numbering (1.jpg to 12500.jpg)

**Download Instructions:**
1. Visit the [Kaggle competition page](https://www.kaggle.com/c/dogs-vs-cats/data)
2. Download `test1.zip` (unlabeled test set)
3. Extract to `test_images/` folder

## 📁 Project Structure

```
cat_dog_prediction/
│
├── data/                           # Training data folder
│   └── 1/
│       └── kagglecatsanddogs_3367a/
│           └── PetImages/
│               ├── Cat/            # Cat training images
│               └── Dog/            # Dog training images
│
├── test_images/                    # Test images (12,500 files)
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
│
├── models/                         # Saved models (generated)
│   ├── model.keras                # Trained Keras model
│   ├── model.tflite               # TFLite standard model
│   └── model_quantized.tflite     # TFLite quantized model
│
├── assets/                         # Visualization outputs (generated)
│   ├── test_results_visualization.png
│   ├── sample_predictions.png
│   ├── model_size_comparison.png
│   ├── inference_benchmark.png
│   └── tflite_sample_predictions.png
│
├── main.ipynb                      # Main training notebook
├── test.ipynb                      # Benchmarking notebook
├── conversion.ipynb                # Model conversion to TFLite
├── download_dataset.ipynb          # Dataset download utilities
│
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore file
└── README.md                       # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.9 - 3.11
- Conda (recommended) or pip
- NVIDIA GPU with CUDA support (optional but recommended)

### Step 1: Clone or Download the Project

```bash
cd cat_dog_prediction
```

### Step 2: Create Conda Environment

```bash
# Create new environment
conda create -n cat_dog_env python=3.10 -y

# Activate environment
conda activate cat_dog_env
```

### Step 3: Install Dependencies

```bash
# Install all required packages from requirements.txt
pip install -r requirements.txt
```

**Note:** The `requirements.txt` includes TensorFlow 2.10.1 which automatically installs compatible CUDA and cuDNN libraries via pip. No separate cuDNN installation is needed.

**Alternative Manual Installation:**
```bash
# Install NumPy (specific version for TensorFlow compatibility)
conda install numpy=1.26.4 -y

# Install TensorFlow with GPU support (includes cuDNN)
pip install tensorflow[and-cuda]==2.10.0

# Install other dependencies
pip install jupyter matplotlib seaborn pandas pillow scipy xlsxwriter
```

### Step 4: Verify GPU Setup (Optional)

```python
import tensorflow as tf
print("GPUs Available:", tf.config.list_physical_devices('GPU'))
```

## ⚙️ Configuration

### Image Settings
```python
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
BATCH_SIZE = 32
```

### Training Configuration
```python
EPOCHS = 50
VALIDATION_SPLIT = 0.2  # 80% train, 20% validation
LEARNING_RATE = 0.001
OPTIMIZER = 'adam'
LOSS_FUNCTION = 'binary_crossentropy'
```

### Data Augmentation Settings
```python
rotation_range = 20
width_shift_range = 0.2
height_shift_range = 0.2
shear_range = 0.2
zoom_range = 0.2
horizontal_flip = True
```

### Callbacks
- **Early Stopping:** Patience = 10 epochs
- **Learning Rate Reduction:** Factor = 0.5, Patience = 5 epochs
- **Model Checkpoint:** Saves best model based on validation accuracy

## 🚀 Usage

### 1. Training the Model

Open `main.ipynb` and run all cells sequentially:

```bash
jupyter notebook main.ipynb
```

**Training Steps:**
1. Load and configure dataset
2. Create data generators with augmentation
3. Build CNN model
4. Compile model
5. Train with callbacks
6. Save trained model to `models/model.keras`
7. Visualize training history

**Expected Training Time:**
- With GPU: ~30-60 minutes
- With CPU: ~3-5 hours

### 2. Testing and Benchmarking

Open `test.ipynb` to benchmark the model:

```bash
jupyter notebook test.ipynb
```

**Benchmark Steps:**
1. Load trained model
2. Process 12,500 test images in batches
3. Generate predictions
4. Create comprehensive visualizations
5. Display sample predictions

**Output Files:**
- `assets/test_results_visualization.png` - 6 analysis charts
- `assets/sample_predictions.png` - Visual samples

## 🏗️ Model Architecture

### CNN Structure

```
Input (128x128x3)
    ↓
Conv2D (32 filters, 3x3) + ReLU + BatchNorm
Conv2D (32 filters, 3x3) + ReLU + BatchNorm
MaxPooling (2x2)
Dropout (0.25)
    ↓
Conv2D (64 filters, 3x3) + ReLU + BatchNorm
Conv2D (64 filters, 3x3) + ReLU + BatchNorm
MaxPooling (2x2)
Dropout (0.25)
    ↓
Conv2D (128 filters, 3x3) + ReLU + BatchNorm
Conv2D (128 filters, 3x3) + ReLU + BatchNorm
MaxPooling (2x2)
Dropout (0.25)
    ↓
Conv2D (256 filters, 3x3) + ReLU + BatchNorm
MaxPooling (2x2)
Dropout (0.25)
    ↓
Flatten
Dense (512) + ReLU + BatchNorm + Dropout (0.5)
Dense (256) + ReLU + BatchNorm + Dropout (0.5)
Dense (1) + Sigmoid
    ↓
Output (Binary: Cat=0, Dog=1)
```

**Total Parameters:** ~5-7 million (trainable)

### Key Features
- **Batch Normalization:** Stabilizes training
- **Dropout:** Prevents overfitting (0.25-0.5)
- **Progressive Filters:** 32 → 64 → 128 → 256
- **Data Augmentation:** Improves generalization
- **Binary Classification:** Sigmoid output with binary crossentropy loss

## 📈 Results

### Training Performance
- **Training Accuracy:** 96.88 %
- **Validation Accuracy:** 96.47 %
- **Training Loss:** 0.0914
- **Validation Loss:** 0.0908

### Test Set Performance (12,500 images)
- **Average Confidence:** ~85-90%
- **High Confidence Predictions (>90%):** ~70-80%
- **Low Confidence Predictions (<60%):** ~5-10%
- **Processing Time:** ~2-5 minutes (with GPU)

### Visualization Outputs

**Test Results Analysis Includes:**
1. Prediction distribution (Pie chart)
2. Confidence score distribution (Histogram)
3. Confidence by class (Box plot)
4. Dog probability distribution (Histogram)
5. Images by confidence range (Bar chart)
6. Summary statistics (Table)

![Test Results Visualization](assets/test_results_visualization.png)
*Comprehensive analysis of 12,500 test predictions including distribution charts, confidence metrics, and statistical summaries*

### Sample Predictions

![Sample Predictions](assets/sample_predictions.png)
*Sample predictions showing model confidence levels across different test images*

### Model Conversion Results

**TensorFlow Lite Conversion:**
- **Standard TFLite:** ~50-60% size reduction, 2-3x faster inference
- **Quantized TFLite:** ~75-80% size reduction, 3-4x faster inference

![Model Size Comparison](assets/model_size_comparison.png)
*Comparison of model sizes: Keras vs TFLite Standard vs TFLite Quantized*

![Inference Benchmark](assets/inference_benchmark.png)
*Inference speed comparison showing significant performance improvements with TFLite models*

![TFLite Sample Predictions](assets/tflite_sample_predictions.png)
*TFLite quantized model predictions demonstrating maintained accuracy after quantization*

## 📝 Files Description

### Notebooks

| File | Description |
|------|-------------|
| `main.ipynb` | Main training notebook with model building and training |
| `test.ipynb` | Benchmarking notebook for test set evaluation |
| `conversion.ipynb` | Convert Keras model to TensorFlow Lite format |
| `download_dataset.ipynb` | Utilities for downloading the dataset |

### Generated Files

| File | Description | Size |
|------|-------------|------|
| `models/model.keras` | Trained Keras model weights and architecture | ~80-100 MB |
| `models/model.tflite` | TensorFlow Lite standard model | ~30-50 MB |
| `models/model_quantized.tflite` | TensorFlow Lite quantized model | ~15-25 MB |
| `assets/test_results_visualization.png` | 6 comprehensive analysis charts | ~2 MB |
| `assets/sample_predictions.png` | 9 sample predictions with confidence levels | ~1 MB |
| `assets/model_size_comparison.png` | Model size comparison chart | ~500 KB |
| `assets/inference_benchmark.png` | Inference speed comparison charts | ~800 KB |
| `assets/tflite_sample_predictions.png` | TFLite model prediction samples | ~1 MB |

### Configuration Files

| File | Description |
|------|-------------|
| `requirements.txt` | Python package dependencies |
| `README.md` | Project documentation (this file) |

## 🔧 Troubleshooting

### Common Issues

**1. GPU Not Detected**
```bash
# Install TensorFlow with CUDA and cuDNN
pip uninstall tensorflow
pip install tensorflow[and-cuda]==2.10.1

# Restart kernel and check
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

**2. NumPy Version Error**
```bash
# Downgrade NumPy
conda install numpy=1.26.4 -y
```

**3. Out of Memory (GPU)**
- Reduce batch size to 16 or 8
- Close other GPU-intensive applications
- Enable memory growth in code

**4. Keras Import Error**
```bash
# Reinstall TensorFlow
pip uninstall tensorflow keras
pip install tensorflow[and-cuda]==2.10.1
```

## 📦 Dependencies

### Core Libraries
- `tensorflow==2.10.1` - Deep learning framework (includes CUDA and cuDNN via pip)
- `numpy==1.26.4` - Numerical computing
- `pandas==2.3.3` - Data manipulation
- `pillow==11.3.0` - Image processing
- `scipy==1.13.1` - Scientific computing

### Visualization
- `matplotlib==3.9.4` - Plotting
- `seaborn==0.13.2` - Statistical visualizations

### Excel Export
- `xlsxwriter==3.2.9` - Excel file creation with images

### Development
- `jupyter` - Interactive notebooks (ipykernel, jupyter-client, jupyter-core)
- `keras==2.10.0` - High-level neural networks API

**Full dependencies list:** See `requirements.txt` for complete package list with versions.

## 🎓 Model Training Tips

1. **Start with small epochs** to test the setup
2. **Monitor validation loss** to detect overfitting
3. **Use GPU** for faster training (30x speedup)
4. **Adjust learning rate** if loss plateaus
5. **Increase data augmentation** if overfitting occurs
6. **Save checkpoints** regularly during long training sessions

## 📊 Performance Optimization

### For Faster Training
- Use GPU with CUDA support
- Increase batch size (if memory allows)
- Use mixed precision training
- Reduce image size to 64x64 or 96x96

### For Better Accuracy
- Increase model depth or width
- Use transfer learning (VGG16, ResNet50)
- Add more data augmentation
- Train for more epochs
- Use ensemble methods

## 🤝 Contributing

This is an educational project. Feel free to:
- Experiment with different architectures
- Try transfer learning approaches
- Add more data augmentation techniques
- Improve the benchmarking visualizations

## 📄 License

This project uses the Microsoft Research Cats and Dogs Dataset, which is available for research purposes.

## 🙏 Acknowledgments

- **Dataset:** Microsoft Research and Kaggle
- **Framework:** TensorFlow/Keras team
- **Inspiration:** Kaggle Dogs vs Cats competition

## 📞 Support

For issues or questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review TensorFlow documentation
3. Check Kaggle competition discussions

---

**Last Updated:** October 26, 2025  
**Version:** 1.0.0  
**Python Version:** 3.10+  
**TensorFlow Version:** 2.10.1

🎉 **Happy Classifying!** 🐱🐶
