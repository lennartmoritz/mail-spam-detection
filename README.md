# Mail Spam Detection using BERT

Welcome to the "Mail Spam Detection" project! This repository contains a Python project that uses PyTorch and the BERT model from Hugging Face to train a classifier for spam and non-spam (ham) email classification. In this README, you will find instructions on setting up the environment, downloading the Enron spam dataset, and running the code.
# Getting Started
## Prerequisites

Before you can run the code, make sure you have the following installed on your system:

    Anaconda
    Python (3.10 or later)
    PyTorch
    Transformers

## Clone the Repository

First, clone this repository to your local machine:


```
git clone https://github.com/lennartmoritz/mail-spam-detection.git
cd mail-spam_detection
```

# Set up Anaconda Environment

If you prefer to use Anaconda, you can create a dedicated environment using the provided environment.yml file. To do this, follow these steps:

1. Navigate to the project directory:
```
cd mail-spam_detection
```
2. Create a new Anaconda environment and activate it:
```
conda env create -f environment.yml
conda activate spam
```
# Creating the Environment Manually

If you prefer to create the Anaconda environment manually, you can use the following commands:

```
conda create --name spam python=3.10
conda activate spam
pip install torch
pip install transformers
```
These commands will create a new environment named "spam" and install the required packages.
This will create a new environment named "spam" with the necessary dependencies.
# Download the Enron Spam Dataset

In order to train and test the spam detection model, you need to download the Enron spam dataset. Place the dataset in the following directory:

```
./dataset/enron1/
```

You can obtain the dataset from the following source: [Enron Spam Dataset](http://nlp.cs.aueb.gr/software_and_datasets/Enron-Spam/index.html)
# Running the Code

With the environment set up and the dataset in place, you can now execute the code. Use the following command to run the spam detection model:
```
python bert_model.py
```

This script will preprocess the dataset, train the BERT-based classifier, and evaluate its performance.
# Conclusion

You've successfully set up the environment and run the mail spam detection code using BERT. Feel free to explore and modify the code to suit your needs. If you have any questions or encounter any issues, please don't hesitate to reach out.

Happy spam detection! 📧🚫