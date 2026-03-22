import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator #type: ignore
from model import build_cnn

# ----------------- Paths -----------------
TRAIN_DIR = r"D:\Parking Lot Detection System\data\train"
TEST_DIR = r"D:\Parking Lot Detection System\data\test"
TRAIN_CSV = r"D:\Parking Lot Detection System\data\train\_classes.csv"
TEST_CSV = r"D:\Parking Lot Detection System\data\test\_classes.csv"
MODEL_PATH = os.path.join("models", "parking_cnn_v2.h5")

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 5

# ----------------- Load Data -----------------
df = pd.read_csv(TRAIN_CSV)
df["filename"] = df["filename"].apply(lambda x: os.path.join(TRAIN_DIR, x))

# 🔹 Automatically detect your label format
if "label" not in df.columns:
    if " space-empty" in df.columns and " space-occupied" in df.columns:
        df["label"] = df[" space-occupied"]  # Use 1 for occupied, 0 for empty
    else:
        raise ValueError("❌ Could not find label columns. Expected either 'label' or [' space-empty', ' space-occupied'].")

# Convert labels to integers if they are strings
df["label"] = df["label"].astype(int)

# ----------------- Split -----------------
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# ----------------- Data Augmentation -----------------
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.6, 1.4],
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_data = datagen.flow_from_dataframe(
    train_df,
    x_col="filename",
    y_col="label",
    target_size=IMG_SIZE,
    class_mode="raw",
    batch_size=BATCH_SIZE
)

val_data = datagen.flow_from_dataframe(
    val_df,
    x_col="filename",
    y_col="label",
    target_size=IMG_SIZE,
    class_mode="raw",
    batch_size=BATCH_SIZE
)

# ----------------- Model -----------------
model = build_cnn(input_shape=IMG_SIZE + (3,))
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss="binary_crossentropy",
              metrics=["accuracy"])

# ----------------- Train -----------------
history = model.fit(train_data, validation_data=val_data, epochs=EPOCHS)

# ----------------- Save -----------------
os.makedirs("models", exist_ok=True)
model.save(MODEL_PATH)
print(f"\nModel saved to {MODEL_PATH}")

# ----------------- Plot -----------------
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.title("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Loss")
plt.legend()
plt.tight_layout()
plt.show()



'''
Epoch 1/5
218/218 ━━━━━━━━━━━━━━━━━━━━ 213s 966ms/step - accuracy: 0.8324 - loss: 0.3841 - val_accuracy: 0.2156 - val_loss: 5.5284
Epoch 2/5
218/218 ━━━━━━━━━━━━━━━━━━━━ 93s 426ms/step - accuracy: 0.8625 - loss: 0.2738 - val_accuracy: 0.7177 - val_loss: 0.5677
Epoch 3/5
218/218 ━━━━━━━━━━━━━━━━━━━━ 94s 431ms/step - accuracy: 0.8783 - loss: 0.2438 - val_accuracy: 0.8781 - val_loss: 0.2401
Epoch 4/5
218/218 ━━━━━━━━━━━━━━━━━━━━ 93s 426ms/step - accuracy: 0.8977 - loss: 0.2178 - val_accuracy: 0.8873 - val_loss: 0.2019
Epoch 5/5
218/218 ━━━━━━━━━━━━━━━━━━━━ 94s 430ms/step - accuracy: 0.9121 - loss: 0.1913 - val_accuracy: 0.8971 - val_loss: 0.2106
'''