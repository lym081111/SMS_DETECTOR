# SMS Spam Classifier

This project implements a machine learning-based SMS spam classifier that can distinguish between spam and legitimate (ham) messages.

## Features
- Text preprocessing with advanced cleaning (URL removal, number removal, stemming, stopword removal).
- TF-IDF vectorization for converting text into numerical features.
- Training using Multinomial Naive Bayes model.
- Model evaluation with accuracy, confusion matrix, classification report, and cross-validation.
- Direct spam classification for messages containing suspicious links.
- Interactive command-line interface for real-time SMS classification.
- Model persistence using `joblib` for saving and loading the trained model and vectorizer.

## Getting Started

### Prerequisites
- Python 3.7 or higher
- Recommended to create and activate a virtual environment.

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/lym081111/sms-spam-classifier.git
   cd sms-spam-classifier
2. Install required packages:
   pip install -r requirements.txt
3. Download necessary NLTK data (if not already downloaded):
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')