# Import required libraries
from flask import Flask, request, render_template
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load trained model
model = joblib.load('phishing_model.pkl')

# Define feature extraction function
def extract_features(url):
    return {
        'length_url': len(url),
        'num_digits': sum(c.isdigit() for c in url),
        'num_special_chars': sum(not c.isalnum() and c != '/' and c != ':' for c in url),
        'has_https': 1 if 'https://' in url else 0  # New feature
    }

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        url = request.form['url']

        # Extract features from the URL
        features = extract_features(url)

        # Convert features into a DataFrame for prediction
        features_df = pd.DataFrame([features])

        # Make prediction using the trained model
        prediction = model.predict(features_df)[0]

        # Determine the result
        result = "Phishing" if prediction == 1 else "Legitimate"

        return render_template('index.html', url=url, result=result)

# Run Flask application on a different port (5001)
if __name__ == '__main__':
    app.run(debug=True, port=5001)
