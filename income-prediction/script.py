from flask import Flask, request, render_template, jsonify
import joblib  # For loading the model
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer  # Or another vectorizer
import nltk
nltk.download('punkt')  # Ensure the necessary NLTK data is downloaded

app = Flask(__name__)

# Load the trained model and vectorizer (assuming you have these files)
model = joblib.load('xgb_model.pkl')  # Replace with your model's filename
vectorizer = joblib.load('tfidf_vectorizer.pkl')  # Replace with your vectorizer's filename

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        tweet = request.form['tweet']
        sentiment = predict_sentiment(tweet)
        return render_template('result.html', prediction=sentiment)

def predict_sentiment(text):
    # Preprocess the text if needed (e.g., vectorizing)
    processed_text = vectorizer.transform([text])  # Transform the text using the vectorizer

    # Predict the sentiment using the loaded model
    prediction = model.predict(processed_text)

    # Convert the numeric prediction to a human-readable label (assumes 0=negative, 1=positive)
    if prediction == 1:
        return "Positive 😊"
    elif prediction == 0:
        return "Negative ☹️"
    else:
        return "Neutral"

if __name__ == '__main__':
    app.run(debug=True)
