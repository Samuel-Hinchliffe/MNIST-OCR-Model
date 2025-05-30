<br />
<div align="center">
  <a >
    <img src="git/logo.png" alt="Logo" width="100" height="100">
  </a>

<h3 align="center">Neural Network Built from Scratch <br>MNIST Digit Classifier </h3>

  <p align="center">
    Achieving 95% Accuracy
    <br />
    <a href="https://github.com/Samuel-Hinchliffe/mnist-ocr-model"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://www.kaggle.com/datasets/oddrationale/mnist-in-csv">MNIST on Kaggle (CSV)</a>
    <br />
  </p>
</div>

# About

This project is your basic run of the mill MNIST digit classifier built from scratch using Python. It implements a simple feed forward neural network architecture to classify handwritten digits from the MNIST dataset. I did this project to give myself the excuse to build a neural network from scratch and to understand the inner workings a little bit better.

## ➕ How the Model Works

This project implements a neural network from scratch, without any external machine learning libraries. The architecture consists of:

- **Input Layer:** 784 nodes (one for each pixel).
- **Hidden Layer:** 128 nodes with ReLU activation.
- **Output Layer:** 10 nodes (digits 0-9) with softmax.

Core components include:

- **Node Class:** Represents the basic unit of the network
- **Edge Class:** Models weighted connections between nodes.
- **Layer Class:** Groups our nodes together, handling forward and backward passes.
- **Network Class:** Orchestration Class.

Training, inference and evaluation run entirely on the CPU, so performance is slow. The network's state (weights, biases, etc.) can be saved to and loaded from JSON files, this allows to inspect the model. JSON is of course, not ideal for production use, but it serves well for educational purposes. None of this project should be used in a production environment where speed matters, it's purely educational.

## 🔢 Classification Outside of the MNIST Dataset

The MINST dataset is a well-known dataset for handwritten digit classification. It's excellent for a starter project. To learn about Neural Networks and how they work.

Outside of it's own training data and testing data, the model is very poor at classifying digits. The dataset is very clean and lacks any noise that the real world has.

## Usage Instructions

To use this application, follow the steps below:

### 1. Clone the repository to your local machine.

### 2. Create a virtual environment:

```bash
python -m venv venv
```

### 3. Activate the virtual environment:

- On Windows:
  ```bash
  venv\Scripts\activate
  ```
- On macOS/Linux:
  ```bash
  source venv/bin/activate
  ```

### 4. Install the required dependencies:

```bash
pip install -r requirements.txt
```

### 5. Run the inference script

Use the following command to run inference on your image (Ideally a MNIST digit, it'll do poorly on anything else):

```bash
python inference.py --image-path path/to/your/image.png
```
