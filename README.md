# Twitter Sentiment Analysis

This project aims to analyze sentiments from tweets using a deep learning model built with PyTorch. The dataset consists of tweets labeled with sentiments (positive, negative, neutral), and the model is trained to classify the sentiment of new tweets.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Training the Model](#training-the-model)
- [Results](#results)
- [License](#license)

## Introduction

Sentiment analysis is a natural language processing (NLP) task that involves determining the sentiment expressed in a piece of text. In this project, we utilize tweets to build a model that classifies sentiments as positive, negative, or neutral. The model is built using an LSTM (Long Short-Term Memory) architecture and trained on a labeled dataset of tweets.

## Dataset

The dataset used in this project contains tweets with associated sentiment labels. The CSV files are structured as follows:

- **Training Dataset**: `twitter_training.csv`
- **Validation Dataset**: `twitter_validation.csv`

Each row in the dataset contains the following columns:
- **ID**: Unique identifier for each tweet.
- **Game**: Game-related information (if applicable).
- **Sentiment**: Sentiment label (e.g., Positive, Negative, Neutral).
- **Tweet**: The actual tweet text.

## Requirements

Make sure to install the following libraries before running the code:

- Python 3.x
- PyTorch
- NLTK
- scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- tqdm

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/twitter-sentiment-analysis.git
cd twitter-sentiment-analysis
```

Install the required libraries using the requirements.txt file:
```bash
pip install -r requirements.txt
```

Usage
To train the sentiment analysis model, run sentiment_analysis.ipynb


This will preprocess the data, build the vocabulary, train the model, and save the trained model as sentiment_model.pth.
