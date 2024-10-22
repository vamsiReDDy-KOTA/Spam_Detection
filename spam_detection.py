import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
data = pd.read_csv('data/email.csv', encoding='latin-1')
data = data[['Category', 'Message']]  # Select relevant columns
data.columns = ['label', 'message']  # Rename columnscls

# Preprocess the data
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

data['message'] = data['message'].apply(preprocess_text)

# Split the dataset into training and testing sets
X = data['message']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Bag-of-Words model
vectorizer = CountVectorizer()
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

# Train the classifier
model = MultinomialNB()
model.fit(X_train_bow, y_train)

# Test the model
y_pred = model.predict(X_test_bow)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Function to predict spam
def predict_spam(message):
    processed_message = preprocess_text(message)
    message_bow = vectorizer.transform([processed_message])
    prediction = model.predict(message_bow)
    return prediction[0]

# Test with a sample message
test_message = "Congratulations! You've won a $1,000 Walmart gift card. Click here to claim your prize."
print("Message:", test_message)
print("Prediction:", predict_spam(test_message))
