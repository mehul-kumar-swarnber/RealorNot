import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load datasets
fake_df = pd.read_csv("fake.csv")
true_df = pd.read_csv("true.csv")

# Label the data
fake_df['label'] = 0  # 0 = fake
true_df['label'] = 1  # 1 = real

# Combine the datasets
df = pd.concat([fake_df, true_df])

# Use the 'title' column for this example — or 'text' if available
df = df[['title', 'label']].dropna()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['title'], df['label'], test_size=0.2, random_state=42
)

# Vectorize the text data
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train the model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Evaluate the model
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the model and vectorizer
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
