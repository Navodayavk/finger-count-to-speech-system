import cv2
import mediapipe as mp
import numpy as np
import os

# --------- Setup ---------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)
mp_draw = mp.solutions.drawing_utils

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "airdraw_dataset")

os.makedirs(DATASET_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)

draw_points = []
canvas = None

# --------- Ask Letter ---------
letter = input("Enter letter to collect (A-Z): ").upper()
LETTER_DIR = os.path.join(DATASET_DIR, letter)
os.makedirs(LETTER_DIR, exist_ok=True)

count = len(os.listdir(LETTER_DIR))

print("Draw the letter in air using index finger")
print("Remove hand to save image")
print("Press ESC to stop")

# --------- Loop ---------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        lm = result.multi_hand_landmarks[0].landmark
        h, w, _ = frame.shape
        x, y = int(lm[8].x * w), int(lm[8].y * h)
        draw_points.append((x, y))
        mp_draw.draw_landmarks(frame, result.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

    else:
        if len(draw_points) > 50:
            count += 1
            img_path = os.path.join(LETTER_DIR, f"{count}.png")
            gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(img_path, gray)
            print(f"Saved {img_path}")

        draw_points.clear()
        canvas[:] = 0

    for i in range(1, len(draw_points)):
        cv2.line(canvas, draw_points[i-1], draw_points[i], (255,255,255), 8)

    cv2.imshow("Air Draw Collector", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()