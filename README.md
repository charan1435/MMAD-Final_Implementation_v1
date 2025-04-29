# Adversarial MRI Defense System

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

A comprehensive system for detecting, analyzing, purifying, and generating adversarial examples in medical imaging - specifically focused on MRI scans.

## Overview

The Adversarial MRI Defense System addresses the emerging security challenge of adversarial attacks on medical imaging. This platform integrates state-of-the-art deep learning models to provide three key capabilities:

1. **Detection** - Identify whether an MRI image has been adversarially manipulated and classify the attack type
2. **Purification** - Remove adversarial perturbations while preserving important diagnostic features
3. **Attack Simulation** - Generate adversarial examples to test and improve defensive measures

## Key Features

- **Advanced Classification Model**: Uses a hybrid architecture combining CNN, Vision Transformer, and SNN components with attention-based fusion to detect FGSM, BIM, and PGD attacks
- **GAN-based Purifier**: U-Net architecture with self-attention mechanisms for superior adversarial noise removal
- **Multiple Attack Methods**: Supports FGSM, BIM, and PGD attack generation with configurable parameters
- **Interactive Web Interface**: Easy-to-use interface for uploading and processing MRI images
- **Detailed Reporting**: Generates comprehensive PDF reports for classification, purification, and attack generation
- **MRI Verification**: Validates that uploaded images are actual MRI scans using a feature-based detector

## Getting Started

### Prerequisites

- Python 3.8 or higher
- PyTorch 1.9 or higher
- CUDA-capable GPU (recommended for optimal performance)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/adversarial-mri-defense.git
   cd adversarial-mri-defense
   ```

2. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Create required directories:
   ```bash
   mkdir -p uploads results reports checkpoints/classifier checkpoints/purifier checkpoints/mri_validator
   ```

### Running the Application

Start the Flask web server:

```bash
python app.py
```

Access the web interface at [http://localhost:5000](http://localhost:5000)

## Usage Guide

### Image Classification

1. Navigate to the "Classify" page
2. Upload an MRI image
3. Click "Classify Image"
4. View the detection results, including:
   - Predicted attack type (Clean, FGSM, BIM, PGD)
   - Confidence scores for each class
   - Model contribution analysis
5. Download a detailed classification report

### Image Purification

1. Navigate to the "Purify" page
2. Upload an adversarial MRI image
3. Click "Purify Image"
4. View the purification results, including:
   - Side-by-side comparison of original and purified images
   - PSNR and SSIM metrics
5. Download the purified image and/or a detailed purification report

### Attack Generation

1. Navigate to the "Attack" page
2. Upload a clean MRI image
3. Select attack type (FGSM, BIM, or PGD)
4. Set epsilon (perturbation size)
5. Click "Generate Adversarial Example"
6. View the attack results, including:
   - Side-by-side comparison of original and adversarial images
   - L2 and L∞ distance metrics
   - Classification results for the adversarial example
7. Download the adversarial image and/or a detailed attack report

## Model Architecture

### Classification Model

The classifier uses a hybrid architecture that combines:

- **Vision Transformer**: For global feature extraction and context understanding
- **CNN**: For local feature extraction and pattern recognition
- **SNN**: For low-level feature processing
- **Attention-based fusion mechanism**: To dynamically weight the contribution of each component

### Purification Model

The purifier uses a U-Net based architecture with:

- Multiple downsample and upsample blocks with skip connections
- Self-attention mechanisms at multiple scales
- Dual attention (channel + spatial) in residual blocks
- Multi-scale feature fusion for preserving details
- Edge enhancement and local contrast enhancement

### Attack Generator

Supports multiple attack methods:

- **FGSM (Fast Gradient Sign Method)**: Single-step attack
- **BIM (Basic Iterative Method)**: Multi-step attack with fixed step size
- **PGD (Projected Gradient Descent)**: Multi-step attack with random initialization

## Project Structure

```
.
├── app.py                    # Main Flask application
├── config.py                 # Configuration settings
├── mri_validator.py          # MRI scan validation module
├── models/
│   ├── classifier.py         # Adversarial detection model
│   ├── purifier.py           # GAN-based purification model
│   └── attack_generator.py   # Attack generation methods
├── static/
│   ├── css/                  # Stylesheets
│   └── js/                   # JavaScript files
├── templates/                # HTML templates
├── utils/
│   ├── image_utils.py        # Image processing utilities
│   ├── model_debug.py        # Model debugging utilities
│   └── report_generator.py   # PDF report generation
├── tests/                    # Unit and integration tests
├── uploads/                  # Uploaded images (created at runtime)
├── results/                  # Generated results (created at runtime)
└── reports/                  # Generated reports (created at runtime)
```

## Future Work

- Support for additional attack methods (CW, DeepFool)
- Integration with DICOM medical imaging format
- Improved explainability of classification decisions
- Transfer learning for domain adaptation to other medical imaging modalities
- Ensemble methods for improved robustness

## Acknowledgements

During the development of this project:

- AI tools were utilized to support knowledge acquisition, assist in understanding complex problems, and aid the exploration of deep learning networks.
- AI assistance helped in identifying errors, suggesting improvements, and refining approaches.
- No code was directly copied without permission or attribution.
- Wherever external code, research, or examples were referenced or adapted, appropriate citations have been provided.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
