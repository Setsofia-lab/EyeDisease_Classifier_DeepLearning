# Eye Disease Classification using Deep Learning

## 🔬 Project Overview

This project develops a machine learning solution for automated classification of eye diseases using deep learning techniques. By leveraging transfer learning and state-of-the-art convolutional neural network architectures, we aim to create a robust diagnostic support tool for detecting key eye conditions.

### 🎯 Project Goals
- Develop an accurate multi-class image classification model for eye diseases
- Compare performance across different pre-trained neural network architectures
- Create a reproducible and scalable machine learning workflow

### 🩺 Supported Disease Categories
- Cataract
- Glaucoma
- Diabetic Retinopathy
- Normal (Healthy) Eye Condition

## 📊 Technical Approach

### Methodology
- **Transfer Learning**: Utilize pre-trained neural network architectures
- **Multi-class Classification**: Classify images into four distinct categories
- **Model Architectures**:
  - VGG16
  - ResNet50
  - MobileNetV2

### Performance Metrics
- Accuracy
- F1-Score
- Precision
- Recall
- Confusion Matrix

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- TensorFlow 2.x
- scikit-learn
- NumPy
- Matplotlib (for visualization)

### Installation

1. Clone the repository
```bash
git clone https://github.com/Setsofia-lab/EyeDisease_Classifier_DeepLearning.git
cd EyeDisease_Classifier_DeepLearning
```

2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install required packages
```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
eye-disease-classification/
│
├── dataset/                   # Raw image dataset
│   ├── cataract/
│   ├── glaucoma/
│   ├── diabetic_retinopathy/
│   └── normal/
│
├── output_directory/          # Processed dataset splits
│   ├── train/
│   ├── val/
│   └── test/
│
├── scripts/
│   ├── split_dataset.py       # Dataset splitting utility
│   ├── train_vgg16.py         # VGG16 training script
│   ├── train_resnet50.py      # ResNet50 training script
│   └── train_mobilenetv2.py   # MobileNetV2 training script
│
├── models/                    # Saved model weights
├── notebooks/                 # Jupyter notebooks for exploration
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## 🔧 Usage

### 1. Prepare Dataset
Organize your image dataset into the required folder structure:
```
dataset/
├── cataract/
├── glaucoma/
├── diabetic_retinopathy/
└── normal/
```

### 2. Split Dataset
```bash
python scripts/split_dataset.py
```

### 3. Train Models
```bash
python scripts/train_vgg16.py
python scripts/train_resnet50.py
python scripts/train_mobilenetv2.py
```

## 📈 Model Performance

### Comparative Results

| Model        | Accuracy | F1-Score | Precision | Recall |
|--------------|----------|----------|-----------|--------|
| VGG16        | 0.xx     | 0.xx     | 0.xx      | 0.xx   |
| ResNet50     | 0.xx     | 0.xx     | 0.xx      | 0.xx   |
| MobileNetV2  | 0.xx     | 0.xx     | 0.xx      | 0.xx   |


## 🧠 Transfer Learning Strategy
- Used pre-trained ImageNet weights
- Froze base model layers
- Added custom classification head
- Applied dropout for regularization


## 🤝 Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
