# American Sign Language Detection 🤟

This project detects A–Z hand gestures using a webcam and converts them into text in real time using:

- Mediapipe for hand tracking
- MLPClassifier (scikit-learn) for classification
- OpenCV for webcam interaction

## 📦 Files

- `collect_landmarks.py` – Capture and save labeled hand landmarks
- `augmentation.py` – Apply noise/rotation/scale augmentation to dataset
- `train_asl_model.py` – Train MLP model on augmented data
- `asl_predictor.py` – Run real-time webcam-based ASL prediction
- `requirements.txt` – Python dependencies

## 📊 Accuracy
- **Before Augmentation**: 94.26%
- **After Augmentation**: 98.16%

## 🚀 Run

```bash
pip install -r requirements.txt
python asl_predictor.py
