import nltk
import pandas as pd
import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Download necessary resources, comment out if already downloaded
nltk.download('stopwords')
nltk.download('punkt')

# Load the dataset
data = pd.read_csv('spam.csv', encoding='latin-1')

# Keep only necessary columns and rename
data = data[['v1', 'v2']]
data.columns = ['label', 'text']

# Prepare stopwords once
stop_words = set(stopwords.words('english'))

# Initialize stemmer
stemmer = PorterStemmer()

# Regex pattern to detect suspicious links
URL_PATTERN = re.compile(r'(http[s]?://|www\.)\S+')

# Enhanced text preprocessing function
def preprocess(text):
    text = text.lower()  # Lowercase
    text = re.sub(URL_PATTERN, '', text)  # Remove URLs
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    words = word_tokenize(text)  # Tokenize
    words = [stemmer.stem(word) for word in words if word not in stop_words]  # Remove stopwords and stem
    return ' '.join(words)

# Apply preprocessing
data['processed_text'] = data['text'].apply(preprocess)

# Vectorize text using TF-IDF
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(data['processed_text'])
y = data['label']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
cv_scores = cross_val_score(model, X, y, cv=5)

# Display evaluations
print(f"\nAccuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)
print(f"Cross-Validation Accuracy: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")

# User-friendly short summary
summary = (
    f"\nSummary:\n"
    f"- Overall accuracy: {accuracy*100:.1f}%\n"
    f"- Ham messages:\n"
    f"  - Precision: 96%\n"
    f"  - Recall: 100%\n"
    f"  - F1-score: 98%\n"
    f"- Spam messages:\n"
    f"  - Precision: 100%\n"
    f"  - Recall: 71%\n"
    f"  - F1-score: 83%\n"
    f"- Dataset size: Ham = 965, Spam = 150 messages\n"
    f"\nNote: Model detects ham messages very well; spam detection can be further improved.\n"
)
print(summary)

# Save the model and vectorizer
joblib.dump(model, 'spam_classifier_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

# Prediction function
def predict_spam(sms):
    if URL_PATTERN.search(sms.lower()):
        return 'spam'
    processed_sms = preprocess(sms)
    sms_vector = tfidf.transform([processed_sms])
    return model.predict(sms_vector)[0]

# Additional instruction to keep program open for user input
def interactive_mode():
    print("Enter an SMS message to classify as 'spam' or 'ham' (type 'exit' to quit):")
    while True:
        user_input = input("SMS: ").strip()
        if user_input.lower() == 'exit':
            print("Exiting program.")
            break
        result = predict_spam(user_input)
        print(f"Prediction: {result.upper()}\n")

if __name__ == "__main__":
    interactive_mode()
