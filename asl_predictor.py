import cv2
import numpy as np
import mediapipe as mp
import joblib
from collections import deque

# Load trained model and label encoder
print("📦 Loading landmark-based ASL model...")
model = joblib.load("landmark_model.pkl")
encoder = joblib.load("label_encoder.pkl")
print("✅ Model loaded.")

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Webcam
print("🎥 Starting webcam...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Webcam not accessible.")
    exit()

# Track prediction stability
history = deque(maxlen=15)
last_confirmed = None
current_text = ""

print("🟢 Show your hand! Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to read from webcam.")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    label = None

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Flatten hand landmarks
            landmark_points = []
            for lm in hand_landmarks.landmark:
                landmark_points.extend([lm.x, lm.y, lm.z])

            input_data = np.array(landmark_points).reshape(1, -1)
            prediction = model.predict(input_data)[0]
            label = encoder.inverse_transform([prediction])[0]

            history.append(label)

            # Check if last N predictions are stable
            if len(history) == history.maxlen and all(p == label for p in history):
                if label != last_confirmed:
                    last_confirmed = label

                    # ✅ Space/Delete/Letter handling
                    if label.lower() in ["space", "spc"]:
                        current_text += " "
                    elif label.lower() in ["delete", "del"]:
                        current_text = current_text[:-1]
                    elif label.isalpha() and len(label) == 1:
                        current_text += label.upper()

    else:
        history.clear()
        last_confirmed = None
        cv2.putText(frame, "No hand detected", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)

    # Display current word
    cv2.putText(frame, f"Text: {current_text}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # Display live prediction
    if label:
        cv2.putText(frame, f"Prediction: {label}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    cv2.imshow("ASL Word Predictor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🧹 Webcam released. Program ended.")