# Automatic Disease Detection in Guava Fruits (as sample)

This project aims to develop an automatic disease detection system for guava fruits using machine learning techniques. The system is designed to identify diseases such as anthracnose and fruit flies, as well as to classify healthy fruits. This will help in early disease detection, protecting harvests, and reducing economic losses in guava production.

## Project Structure

- **data/**: Contains the image datasets.
  - **raw/**: Original images categorized into three classes:
    - **anthracnose/**: Images of guava fruits affected by anthracnose.
    - **fruit_flies/**: Images of guava fruits affected by fruit flies.
    - **healthy/**: Images of healthy guava fruits.
  - **processed/**: Preprocessed images after applying techniques like unsharp masking and CLAHE.
    - **train/**: Training dataset.
    - **val/**: Validation dataset.
    - **test/**: Test dataset.
- **notebooks/**: Jupyter notebooks for data preprocessing, model training, and evaluation.
  - **data_preprocessing.ipynb**: Notebook for data preprocessing.
  - **model_training.ipynb**: Notebook for model training.
  - **model_evaluation.ipynb**: Notebook for model evaluation.
- **src/**: Source code for data preprocessing, model definition, training, and evaluation.
  - **data_preprocessing.py**: Script for data preprocessing.
  - **model.py**: Script defining the model architecture.
  - **train.py**: Script for training the model.
  - **evaluate.py**: Script for evaluating the model.

## Environment

- Python 3.10
- CUDA 12.4
- PyTorch 2.5.1
- NumPy
- Pandas
- Matplotlib
- OpenCV
- scikit-learn
- Keras
- torchvision
- TensorFlow
- Jupyter

To install PyTorch with the specified CUDA version, use the following command:

```sh
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
