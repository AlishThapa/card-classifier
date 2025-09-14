PyTorch Playing Card Classifier 🃏
A deep learning model built with PyTorch to classify 53 types of playing cards with over 93% accuracy.

This project provides a complete walkthrough for training an EfficientNet model to identify playing cards from images. It's an ideal starting point for anyone looking to learn the fundamentals of computer vision and PyTorch.

## 🌟 Features
High Accuracy: Achieves over 93% accuracy on the test set.

Modern Architecture: Utilizes a pre-trained EfficientNet-B0 model from the timm library.

Data Augmentation: Employs transformations like random flips and rotations to create a more robust model.

End-to-End Workflow: Covers everything from data loading and preprocessing to training, validation, and evaluation.

Clear Visualizations: Includes code to plot training loss and visualize model predictions on new images.

## 💻 Technologies Used
Python 3.8+

PyTorch: The core deep learning framework.

torchvision: For image transformations and datasets.

timm: (PyTorch Image Models) For easily accessing pre-trained EfficientNet.

NumPy: For numerical operations.

Matplotlib: For plotting graphs and visualizing images.

Jupyter Notebook: For interactive development.

## 📂 Project Structure
For the code to run correctly, your project should follow this structure. The data directory is not included in this repository and must be downloaded separately.

pytorch-card-classifier/
├── data/
│   ├── train/
│   │   ├── ace of clubs/
│   │   ├── .../
│   ├── valid/
│   │   ├── ace of clubs/
│   │   ├── .../
│   └── test/
│       ├── ace of clubs/
│       └── .../
├── card_classifier.ipynb
├── requirements.txt
└── README.md
## 🚀 Getting Started
Follow these steps to get the project running on your local machine.

### 1. Prerequisites
Python 3.8 or newer: You can download it from python.org.

Git: For cloning the repository.

### 2. Installation & Setup
Clone the Repository

Bash

git clone [https://github.com/your-username/pytorch-card-classifier.git](https://github.com/AlishThapa/card-classifier.git)
cd pytorch-card-classifier
Download the Dataset

The dataset used is the "Cards Image Dataset Classification" on Kaggle.

Download it from here: [https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification/data)

Unzip the downloaded file and place the train, valid, and test folders inside the data/ directory as shown in the project structure above.

Create and Activate a Virtual Environment

This keeps your project dependencies isolated.

Bash

# Create the environment
python -m venv venv

# Activate on Windows (PowerShell)
.\venv\Scripts\activate

# Activate on Mac/Linux
source venv/bin/activate
Install Required Packages

A requirements.txt file is included for easy installation.

Bash

pip install -r requirements.txt
## ▶️ How to Run
Launch VS Code and open the project folder.

Open the card_classifier.ipynb notebook file.

Select the Kernel: In the top-right corner of the notebook, click to select a kernel. Be sure to choose the Python interpreter from your virtual environment (it should have ('venv': venv) in the name).

Run the Cells: You can run the notebook cells sequentially to preprocess the data, train the model, and evaluate its performance.

## 📊 Example Results
After training for 5 epochs, the model achieves high accuracy and demonstrates good generalization with a low loss value.

Validation Accuracy: 92.08%

Test Accuracy: 93.58%

The model performs well on unseen test images, correctly identifying the card and its suit.