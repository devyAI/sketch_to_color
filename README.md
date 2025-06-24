# CIFAR-10 Colorization Project

This project implements a PyTorch-based model to convert grayscale images back to their original colored versions using the CIFAR-10 dataset.

## Project Structure

- `requirements.txt`: Contains project dependencies
- `load_data.py`: Handles data loading and preprocessing
- `model.py`: Contains the ColorizerNet model definition
- `train.py`: Main training script with visualization capabilities

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Training

To train the model:
```bash
python train.py
```

The training process will:
1. Download the CIFAR-10 dataset
2. Train the colorization model
3. Save checkpoints in the `checkpoints` directory
4. Show visualization of results after training

## Model Architecture

The model consists of:
- An encoder that reduces the image dimensions while extracting features
- A decoder that reconstructs the color channels from the grayscale input

The architecture is designed to learn the mapping from grayscale to colored images while preserving spatial information.
