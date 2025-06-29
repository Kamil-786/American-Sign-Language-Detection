import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import joblib

# Load model & label encoder
model = joblib.load("landmark_model.pkl")
encoder = joblib.load("label_encoder.pkl")

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Streamlit UI setup
st.set_page_config(layout="wide")
st.title("🧠 American Sign Language (ASL) - Real-Time Gesture Detection")
st.markdown("👉 Use your **webcam** to show a gesture and see predictions below.")

# Webcam start
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error("❌ Webcam not accessible.")
else:
    run = st.checkbox("▶ Start Gesture Recognition")

    # Initialize variables
    accumulated_text = st.session_state.get("accumulated_text", "")
    prediction_text = st.empty()
    frame_display = st.empty()

    # Add text area outside the loop (avoid multiple keys)
    text_area = st.text_area("📝 Accumulated Text", accumulated_text, height=150, key="live_text_area")

    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Couldn't read from webcam.")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        label = ""
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Predict
                flat = [val for lm in hand_landmarks.landmark for val in [lm.x, lm.y, lm.z]]
                input_data = np.array(flat).reshape(1, -1)
                pred = model.predict(input_data)[0]
                label = encoder.inverse_transform([pred])[0]

        # Update if label is new
        if label and not accumulated_text.endswith(label):
            accumulated_text += label
            st.session_state["accumulated_text"] = accumulated_text

        prediction_text.subheader(f"✏️ Prediction: {label if label else 'Detecting...'}")
        frame_display.image(frame, channels="RGB")

    cap.release()