import cv2
import json
import numpy as np
import tensorflow as tf

# ---------------- PATHS ----------------
MODEL_PATH = "models/parking_cnn.h5"
SLOTS_PATH = "assets/slots.json"
IMAGE_PATH = "assets/sample_frame.jpg"

# ---------------- LOAD MODEL & DATA ----------------
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Loading parking slot annotations...")
with open(SLOTS_PATH, "r") as f:
    data = json.load(f)
    slots = data["slots"]  # access the actual slot list


# ---------------- PARAMETERS ----------------
IMG_SIZE = (128, 128)
COLOR_OCCUPIED = (0, 0, 255)      # Red
COLOR_EMPTY = (0, 255, 0)         # Green
COLOR_EV = (255, 255, 0)          # Cyan
COLOR_UNCERTAIN = (0, 255, 255)   # Yellow

# ---------------- HELPER FUNCTION ----------------
def crop_slot(frame, points):
    pts = np.array(points, dtype=np.int32)
    rect = cv2.boundingRect(pts)
    x, y, w, h = rect
    crop = frame[y:y+h, x:x+w]
    crop = cv2.resize(crop, IMG_SIZE)
    crop = crop.astype("float32") / 255.0
    crop = np.expand_dims(crop, axis=0)
    return crop

# ---------------- RUN ON IMAGE ----------------
frame = cv2.imread(IMAGE_PATH)
if frame is None:
    raise FileNotFoundError(f"Could not find image: {IMAGE_PATH}")

total_slots = len(slots)
occupied_slots = 0
ev_slots = 0
correct = 0
uncertain = 0

for i, slot in enumerate(slots):
    crop = crop_slot(frame, slot["points"])
    prob = model.predict(crop, verbose=0)[0][0]
    is_ev = slot.get("type", "") == "EV"

    # Determine predicted label
    if prob > 0.7:
        pred_label = "Occupied"
        color = COLOR_OCCUPIED
        occupied_slots += 1
    elif prob < 0.3:
        pred_label = "Empty"
        color = COLOR_EV if is_ev else COLOR_EMPTY
    else:
        pred_label = "Uncertain"
        color = COLOR_UNCERTAIN
        uncertain += 1

    # Optional: read true label if available
    true_label = slot.get("true_label", None)
    if true_label:
        if true_label.lower() == pred_label.lower():
            correct += 1

    # Draw polygon & label
    pts = np.array(slot["points"], np.int32)
    cv2.polylines(frame, [pts], True, color, 2)
    cv2.putText(frame, f"{i+1}:{pred_label} ({prob*100:.1f}%)",
                (pts[0][0], pts[0][1]-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if is_ev:
        ev_slots += 1

# ---------------- ACCURACY & SUMMARY ----------------
available_slots = total_slots - occupied_slots

print("\n Summary:")
print(f"Total Slots: {total_slots}")
print(f"Occupied: {occupied_slots}")
print(f"Available: {available_slots}")
print(f"EV Slots: {ev_slots}")
print(f"Uncertain: {uncertain}")

if any("true_label" in s for s in slots):
    accuracy = (correct / total_slots) * 100
    print(f"Slot-level accuracy: {accuracy:.2f}%")
else:
    print("No ground-truth labels ('true_label') found in slots.json")

# ---------------- DASHBOARD OVERLAY ----------------
cv2.rectangle(frame, (0, 0), (380, 140), (0, 0, 0), -1)
cv2.putText(frame, f"Total Slots: {total_slots}", (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
cv2.putText(frame, f"Occupied: {occupied_slots}", (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_OCCUPIED, 2)
cv2.putText(frame, f"Available: {available_slots}", (10, 85),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_EMPTY, 2)
cv2.putText(frame, f"EV Slots: {ev_slots}", (200, 55),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_EV, 2)
cv2.putText(frame, f"Uncertain: {uncertain}", (200, 85),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_UNCERTAIN, 2)

# ---------------- SAVE & SHOW ----------------
output_path = "assets/static_output_accuracy.jpg"
cv2.imwrite(output_path, frame)
print(f"\nOutput saved to {output_path}")

cv2.imshow("Static Parking Lot Analysis", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
