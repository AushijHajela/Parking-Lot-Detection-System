import cv2
import numpy as np
import os

# ----------- Paths -----------
INPUT_PATH = os.path.join("assets", "sample_frame.jpg")
OUTPUT_PATH = os.path.join("assets", "sample_frame_clean.jpg")

# ----------- Load Image -----------
img = cv2.imread(INPUT_PATH)
if img is None:
    raise FileNotFoundError(" sample_frame.jpg not found in assets/ folder!")

# ----------- Convert to LAB color space (better light normalization) -----------
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)

# ----------- Equalize lighting to reduce shadows -----------
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
l_eq = clahe.apply(l)

lab_eq = cv2.merge((l_eq, a, b))
shadow_fixed = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

# ----------- Optional: slight blur to smooth edges -----------
shadow_fixed = cv2.GaussianBlur(shadow_fixed, (3, 3), 0)

# ----------- Save and Show Results -----------
cv2.imwrite(OUTPUT_PATH, shadow_fixed)
print(f" Shadow-reduced image saved to: {OUTPUT_PATH}")

cv2.imshow("Original", img)
cv2.imshow("Shadow Reduced", shadow_fixed)
cv2.waitKey(0)
cv2.destroyAllWindows()
