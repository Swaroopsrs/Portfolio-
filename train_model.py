import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
data = pd.read_csv('minimal_dataset_with_url.csv')

# Drop 'url' column — it's not numeric and can't be used directly
X = data.drop(columns=['url', 'label'])  # Features only
y = data['label']                        # Target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

# Save model
with open('phishing_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("🎉 Model trained and saved as phishing_model.pkl")
