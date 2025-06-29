import cv2
import mediapipe as mp
import numpy as np
import csv
import os

# File setup
output_file = "hand_landmarks.csv"
header_written = os.path.exists(output_file)

# ASL classes
classes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "nothing", "space"]

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Webcam not accessible.")
    exit()

print("📸 Press 'c' to capture hand landmarks.")
print("🔴 Press 'q' to quit.\n")

# Label to assign
current_label = input("🔤 Enter label for current capture (A-Z, del, nothing, space): ").strip().lower()

with open(output_file, mode="a", newline="") as f:
    writer = csv.writer(f)

    # Write header if not already
    if not header_written:
        header = ["label"] + [f"{axis}{i+1}" for i in range(21) for axis in ["x", "y", "z"]]
        writer.writerow(header)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to read from webcam.")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw landmarks on screen
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, f"Label: {current_label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow("ASL Landmark Capture", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('c') and result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                landmark_row = []
                for lm in hand_landmarks.landmark:
                    landmark_row.extend([lm.x, lm.y, lm.z])
                writer.writerow([current_label] + landmark_row)
                print("✅ Landmark saved.")

        elif key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("🧹 Webcam released. Program ended.")