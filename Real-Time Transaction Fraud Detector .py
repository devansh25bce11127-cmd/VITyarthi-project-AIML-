# Real-Time Transaction Fraud Detector
# CSA2001 - Fundamentals in AI & ML Project

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv("creditcard.csv")

# Features and target
X = data.drop("Class", axis=1)
y = data["Class"]

# Scaling data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Machine Learning Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# -----------------------------
# Real-time transaction checker
# -----------------------------

def check_transaction(transaction):

    transaction_scaled = scaler.transform([transaction])
    prediction = model.predict(transaction_scaled)

    if prediction[0] == 0:
        print("Transaction Status: APPROVED")
    else:
        print("Transaction Status: DECLINED (Possible Fraud)")


# Example transaction
example_transaction = X.iloc[0].values

check_transaction(example_transaction)