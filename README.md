# CNN Handwritten Digit Recognition

A simple handwritten digit recognition project using a Convolutional Neural Network (CNN) and the MNIST dataset.

## Features

- Recognizes digits from 0 to 9
- Upload an image of a handwritten digit
- Shows the predicted digit
- Shows prediction confidence
- Streamlit web interface
- Docker support

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pillow
- Streamlit
- Docker

## Dataset

MNIST handwritten digit dataset.

- 60,000 training images
- 10,000 testing images
- Image size: 28 × 28 pixels
- 10 classes (0–9)

## CNN Model

The model contains:

- Convolutional layers
- Max Pooling layers
- Flatten layer
- Dense layers
- Softmax output layer

The model achieved approximately **98% test accuracy**.

## Project Structure

```text
cnn-handwritten-digit-recognition/
│
├── models/
│   └── mnist_cnn.keras
│
├── src/
│   ├── train.ipynb
│   └── predict.ipynb
│
├── app.py
├── Dockerfile
├── requirements.txt
├── .gitignore
├── .dockerignore
└── README.md

Run Locally
Install the required packages:
pip install -r requirements.txt

Run the application:
streamlit run app.py

Open:
http://localhost:8501

Run with Docker
Build the image:
docker build -t digit-recognizer .

Run the container:
docker run -p 8501:8501 digit-recognizer

Then open:
http://localhost:8501

Author
Suprakash Ghosh
B.Tech CSE (AI & ML)

This is better for your project because it's **short, readable, and professional** without unnecessary explanations.
