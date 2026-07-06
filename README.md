# Visual Classifier with Neural Networks

Interactive application that demonstrates how a neural network can interpret, process, and classify images from numerical data.

## 🚀 Live Demo

[Live Demo - Visual Classifier with Neural Networks](https://visual-classifier.streamlit.app)

## 📌 Features

- Interactive Fashion-MNIST image classification
- Comparison between multiple neural network architectures
- Dataset exploration and visual testing
- Upload and classify custom images
- Confidence visualization for predictions
- Confusion matrix and performance metrics
- English and Portuguese language support

## 🛠️ Tech Stack

- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Pillow
- Joblib

## 📈 Future Improvements

- Improved preprocessing for real-world images
- Additional neural network architectures
- Enhanced mobile and tablet responsiveness

## 🔁 Training New Models

This repository includes the original training pipeline used to generate the neural network models.

The `fashion-mnist_train.csv` file is not included in the repository because of its size. It is only required if you want to recreate the models from the training data.

To retrain a model, after installing and adding the training file, run:

```bash
python train_model.py
```

Generated models will be saved in the `models/` directory.
