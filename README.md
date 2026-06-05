# Visual Classifier with Neural Networks

An interactive Streamlit application for exploring image classification with MLP neural networks. The project uses the Fashion-MNIST dataset to compare different model architectures, visualize performance metrics, test dataset images, and classify external images uploaded by the user.

## 🚀 Features

- Selection between pre-trained MLP models.
- Visualization of training and test accuracy.
- Class-level performance analysis.
- Confusion matrix with chart and table.
- Visual tests using Fashion-MNIST images.
- Upload and preprocessing of external images.
- Technical checklist to evaluate the uploaded image quality.

## 🧠 Available Models

The application allows users to compare three MLP architectures:

| Model  |  Architecture | Description   |
| ------ | ------------: | ------------- |
| MLP 4  |  784 x 4 x 10 | Smaller model |
| MLP 8  |  784 x 8 x 10 | Base model    |
| MLP 32 | 784 x 32 x 10 | Larger model  |

Each model receives a 28x28 image as input, represented by 784 numerical values, and returns a prediction among 10 possible classes.

## 🛠️ Built With

- Python
- Streamlit
- pandas
- NumPy
- Matplotlib
- scikit-learn
- joblib
- Pillow

## ⚙️ How to Run

Clone the repository:

```bash
git clone <REPOSITORY_URL>
cd <PROJECT_FOLDER_NAME>
```

Create a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On Linux/macOS:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## 🔁 Model Retraining

The `train_model.py` file can be used to train the MLP models again.

To retrain the models, the following files must be inside the `data/` folder:

```text
data/fashion-mnist_train.csv
data/fashion-mnist_test.csv
```

The `fashion-mnist_train.csv` file is not included in the repository because of its size. It is only required if you want to recreate the models from the training data.

After installing and adding the training file, run:

```bash
python train_model.py
```

The trained models will be saved again in the `models/` folder.

## 🖼️ External Image Upload

The application allows users to upload PNG, JPG, or JPEG images.

Before prediction, uploaded images are converted to grayscale, processed, centered, resized to 28x28 pixels, and transformed into the numerical format expected by the model.

The app also displays a technical checklist with indicators such as proportion, contrast, object area, and centering.

## ⚠️ Limitations

The models were trained with Fashion-MNIST, a dataset made of simple, centered, grayscale images with a resolution of 28x28 pixels.

Because of that, real-world images may produce inaccurate predictions, especially when they include complex backgrounds, poor lighting, strong shadows, distorted angles, multiple objects, or excessive visual details.

The upload feature is meant to demonstrate how the model behaves with external images, not to serve as a definitive classifier for real-world photos.

## 📄 License

This project is available for study, demonstration, and experimentation purposes.
