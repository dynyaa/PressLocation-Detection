import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib


train_features = pd.read_csv('train_features.csv')
val_features = pd.read_csv('val_features.csv')
test_features = pd.read_csv('test_features.csv')

X_train = train_features.drop(columns=['label'])
y_train = train_features['label']

X_val = val_features.drop(columns=['label'])
y_val = val_features['label']

X_test = test_features.drop(columns=['label'])
y_test = test_features['label']

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

y_val_pred = model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f'Validation Accuracy: {val_accuracy:.4f}')

print("\nValidation Classification Report:")
print(classification_report(y_val, y_val_pred))

y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Test Accuracy: {test_accuracy:.4f}')

print("\nTest Classification Report:")
print(classification_report(y_test, y_test_pred))
joblib.dump(model, 'logistic_regression_model.pkl')