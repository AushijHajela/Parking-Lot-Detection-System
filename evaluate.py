import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator #type: ignore

# ----------------- Paths -----------------
TEST_DIR = r"D:\Parking Lot Detection System\data\test"
TEST_CSV = r"D:\Parking Lot Detection System\data\test\_classes.csv"
MODEL_PATH = os.path.join("models", "parking_cnn_v2.h5")

IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# ----------------- Load Model -----------------
model = tf.keras.models.load_model(MODEL_PATH)

# ----------------- Load Test Data -----------------
datagen = ImageDataGenerator(rescale=1./255)

test_df = pd.read_csv(TEST_CSV)
test_df["filename"] = test_df["filename"].apply(lambda x: os.path.join(TEST_DIR, x))
test_df["label"] = test_df[" space-occupied"]

test_data = datagen.flow_from_dataframe(
    test_df,
    x_col="filename",
    y_col="label",
    target_size=IMG_SIZE,
    class_mode="raw",
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ----------------- Evaluate -----------------
loss, acc = model.evaluate(test_data)
print(f"\nTest Accuracy: {acc*100:.2f}%")

# ----------------- Predictions -----------------
y_true = test_df["label"].values
y_pred_probs = model.predict(test_data)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

# ----------------- Confusion Matrix -----------------
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Empty", "Occupied"],
            yticklabels=["Empty", "Occupied"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ----------------- Classification Report -----------------
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=["Empty","Occupied"]))



'''
Classification Report:

              precision    recall  f1-score   support

       Empty       0.96      1.00      0.98       251
    Occupied       1.00      0.99      0.99       991

    accuracy                           0.99      1242
   macro avg       0.98      0.99      0.99      1242
weighted avg       0.99      0.99      0.99      1242

Test Accuracy: 99.11%
'''