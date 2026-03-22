import cv2
import json
import os

# Paths
IMG_PATH = os.path.join("assets", "sample_frame.jpg")
OUT_PATH = os.path.join("assets", "slots.json")

points = []
slots = []
mode = "normal"  # default mode

color_map = {
    "normal": (0, 255, 0),     # green
    "ev": (255, 255, 0),       # cyan
    "entry": (255, 0, 0),      # blue
    "exit": (0, 0, 255)        # red
}

def click_event(event, x, y, flags, param):
    global points, slots, mode
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(img, (x, y), 5, color_map[mode], -1)

        if len(points) == 4:
            slots.append({
                "id": len(slots) + 1,
                "points": points.copy(),
                "type": mode
            })
            cv2.polylines(img, [np.array(points)], True, color_map[mode], 2)
            points = []
            print(f"Added {mode} slot #{len(slots)}")

def save_slots():
    with open(OUT_PATH, "w") as f:
        json.dump({"slots": slots}, f, indent=4)
    print(f"\n Saved {len(slots)} slots to {OUT_PATH}")

if __name__ == "__main__":
    import numpy as np

    if not os.path.exists(IMG_PATH):
        print("sample_frame.jpg not found in assets/")
        exit()

    global img
    img = cv2.imread(IMG_PATH)
    clone = img.copy()
    cv2.namedWindow("Slot Annotator")
    cv2.setMouseCallback("Slot Annotator", click_event)

    print("""
     Smart Parking Annotator

Instructions:
 - Click 4 points per slot (corners)
 - Press [1] for Normal slot
 - Press [2] for EV slot
 - Press [3] for Entry gate
 - Press [4] for Exit gate
 - Press [S] to Save
 - Press [Q] to Quit
""")

    while True:
        cv2.imshow("Slot Annotator", img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('1'):
            mode = "normal"
            print("Mode → Normal Slot")
        elif key == ord('2'):
            mode = "ev"
            print("Mode → EV Slot")
        elif key == ord('3'):
            mode = "entry"
            print("Mode → Entry Gate")
        elif key == ord('4'):
            mode = "exit"
            print("Mode → Exit Gate")
        elif key == ord('s'):
            save_slots()
        elif key == ord('q'):
            break
        elif key == ord('r'):
            img = clone.copy()
            slots.clear()
            print(" Reset all annotations")

    cv2.destroyAllWindows()
