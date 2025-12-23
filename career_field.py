import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import joblib

# Load the dataset
df = pd.read_csv('career_fields.csv')

# Extract features (questions) and target variable
X = df.iloc[:, :-1]  # Features (questions)
y = df['Predicted_Career_Field']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert features (questions) into a single string per row
X_train = X_train.apply(lambda row: ' '.join(row), axis=1)
X_test = X_test.apply(lambda row: ' '.join(row), axis=1)

# Vectorize the answers using CountVectorizer
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train a Multinomial Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate the model
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)
print('Train Accuracy:', train_accuracy)
print('Test Accuracy:', test_accuracy)

# Save the model and vocabulary
joblib.dump(model, 'career_field_prediction_model.pkl')
joblib.dump(vectorizer.vocabulary_, 'vocabulary.pkl')
