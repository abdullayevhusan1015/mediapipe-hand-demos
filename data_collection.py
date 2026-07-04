import os
import csv
import cv2
import mediapipe as mp

SIGNS = {
    ord('0'): "salom",
    ord('1'): "rahmat",
    ord('2'): "yaxshi",
    ord('3'): "yordam",
    ord('4'): "oila",
    ord('5'): "ish",
    ord('6'): "ha",
    ord('7'): "yoq",
    ord('8'): "kechirasiz",
    ord('9'): "xayr",
}

DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "signs.csv")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

os.makedirs(DATA_DIR, exist_ok=True)
file_exists = os.path.isfile(CSV_PATH)

counts = {label: 0 for label in SIGNS.values()}
if file_exists:
    with open(CSV_PATH, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if row and row[0] in counts:
                counts[row[0]] += 1

csv_file = open(CSV_PATH, "a", newline="")
csv_writer = csv.writer(csv_file)
if not file_exists:
    header = ["label"] + [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]
    csv_writer.writerow(header)

cap = cv2.VideoCapture(0)
current_label = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    key = cv2.waitKey(1) & 0xFF

    if key in SIGNS:
        current_label = SIGNS[key]
    elif key == ord('q'):
        break

    hand_landmarks = None
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    is_recording = (key == 32) and current_label is not None and hand_landmarks is not None
    if is_recording:
        row = [current_label]
        for lm in hand_landmarks.landmark:
            row.extend([lm.x, lm.y, lm.z])
        csv_writer.writerow(row)
        counts[current_label] += 1

    label_text = current_label if current_label else "None (press 0-9)"
    cv2.putText(frame, f"Sign: {label_text}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    if current_label:
        cv2.putText(frame, f"Samples: {counts[current_label]}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    if is_recording:
        cv2.putText(frame, "RECORDING", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow("Data Collection", frame)

cap.release()
cv2.destroyAllWindows()
csv_file.close()

print("\nSummary of collected samples:")
for label, count in counts.items():
    print(f"  {label}: {count}")
