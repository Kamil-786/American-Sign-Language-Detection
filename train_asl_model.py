import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 📥 Load dataset
df = pd.read_csv("augmented_landmarks.csv", skiprows=1, header=None)

# ✅ Drop rows with any NaN values
df = df.dropna()

print(f"✅ Loaded dataset with {len(df)} samples")

# 🏷️ Separate labels and features
labels = df.iloc[:, -1]         # Last column = label
features = df.iloc[:, :-1]      # All columns except last = landmarks

# 🎯 Encode labels (e.g., 'a', 'b' → 0, 1)
encoder = LabelEncoder()
y = encoder.fit_transform(labels)
X = features.values

# 🔀 Train-test split (no stratify due to small samples)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🧠 Define MLP model
model = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu',
                      solver='adam', max_iter=500, random_state=42)

# 🚀 Train
print("🧠 Training model...")
model.fit(X_train, y_train)
print("✅ Training complete.")

# 🧪 Evaluate
y_pred = model.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))
print(f"🎯 Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# 💾 Save model + label encoder
joblib.dump(model, "landmark_model.pkl")
joblib.dump(encoder, "label_encoder.pkl")
print("💾 Model and label encoder saved.")

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()