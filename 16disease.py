# 🩺 Naïve Bayes From Scratch - Disease Diagnosis Prediction
# Author: Shravani Farkade

# ---------------------- IMPORT LIBRARIES ----------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------- STEP 1: LOAD DATA ----------------------
data = pd.read_csv("/content/disease_diagnosis_16_17.csv")

# ---------------------- STEP 2: ENCODE CATEGORICAL COLUMNS ----------------------
le = LabelEncoder()
for col in data.select_dtypes(include='object').columns:
    data[col] = le.fit_transform(data[col])

# Separate features and target
X = data.drop(columns=['Diagnosis']).values
y = data['Diagnosis'].values

# ---------------------- STEP 3: VISUALIZE CLASS DISTRIBUTION ----------------------
plt.figure(figsize=(6,4))
sns.countplot(x=y, palette='coolwarm')
plt.title("Disease Diagnosis Class Distribution")
plt.xlabel("Diagnosis Classes")
plt.ylabel("Count")
plt.show()

# ---------------------- STEP 4: TRAIN-TEST SPLIT ----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------- STEP 5: IMPLEMENT NAIVE BAYES FROM SCRATCH ----------------------
class NaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.class_priors = {}
        self.feature_probs = {}
        n_features = X.shape[1]

        for c in self.classes:
            X_c = X[y == c]
            # Prior probability P(C)
            self.class_priors[c] = X_c.shape[0] / X.shape[0]
            # Likelihood P(x_i | C) with Laplace smoothing
            self.feature_probs[c] = (X_c.sum(axis=0) + 1) / (X_c.sum() + n_features)

    def predict(self, X):
        y_pred = []
        for x in X:
            posteriors = []
            for c in self.classes:
                prior = np.log(self.class_priors[c])
                likelihood = np.sum(x * np.log(self.feature_probs[c]))
                posterior = prior + likelihood
                posteriors.append(posterior)
            y_pred.append(self.classes[np.argmax(posteriors)])
        return np.array(y_pred)

# ---------------------- STEP 6: TRAIN MODEL ----------------------
model = NaiveBayes()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ---------------------- STEP 7: EVALUATE MODEL ----------------------
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("📊 Naïve Bayes From Scratch - Disease Diagnosis Prediction")
print(f"✅ Accuracy: {acc * 100:.2f}%")
print("Confusion Matrix:\n", cm)

# ---------------------- STEP 8: VISUALIZE CONFUSION MATRIX ----------------------
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples')
plt.title("Confusion Matrix - Naïve Bayes (Disease Diagnosis)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
