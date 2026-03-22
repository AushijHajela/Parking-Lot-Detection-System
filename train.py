import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator #type: ignore
from model import build_cnn

# ----------------- Paths -----------------
TRAIN_DIR = r"D:\Parking Lot Detection System\data\train"
TEST_DIR = r"D:\Parking Lot Detection System\data\test"
TRAIN_CSV = r"D:\Parking Lot Detection System\data\train\_classes.csv"
TEST_CSV = r"D:\Parking Lot Detection System\data\test\_classes.csv"

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 5
MODEL_PATH = os.path.join("models", "parking_cnn.h5")

# ----------------- Load Train Data -----------------
df = pd.read_csv(TRAIN_CSV)
df["filename"] = df["filename"].apply(lambda x: os.path.join(TRAIN_DIR, x))
df["label"] = df[" space-occupied"]

train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)

# ----------------- Data Generators -----------------
datagen = ImageDataGenerator(rescale=1./255)

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
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer,loss="binary_crossentropy",metrics=["accuracy"])


# ----------------- Train -----------------
history = model.fit(train_data, validation_data=val_data, epochs=EPOCHS)

# Save model
os.makedirs("models", exist_ok=True)
model.save(MODEL_PATH)

# ----------------- Plot Graphs -----------------
plt.figure(figsize=(10,4))

# Accuracy
plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.title("Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

# Loss
plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()

# ----------------- Evaluate on Test Set -----------------
if os.path.exists(TEST_CSV):
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

    test_loss, test_acc = model.evaluate(test_data)
    print(f"\n✅ Test Accuracy: {test_acc*100:.2f}%")




'''
Epoch 1/5
218/218 ━━━━━━━━━━━━━━━━━━━━ 76s 340ms/step - accuracy: 0.9226 - loss: 0.2022 - val_accuracy: 0.2156 - val_loss: 4.2828
Epoch 2/5
218/218 ━━━━━━━━━━━━━━━━━━━━ 73s 336ms/step - accuracy: 0.9668 - loss: 0.0875 - val_accuracy: 0.7729 - val_loss: 0.5453
Epoch 3/5
218/218 ━━━━━━━━━━━━━━━━━━━━ 74s 339ms/step - accuracy: 0.9804 - loss: 0.0538 - val_accuracy: 0.9799 - val_loss: 0.0666
Epoch 4/5
218/218 ━━━━━━━━━━━━━━━━━━━━ 74s 340ms/step - accuracy: 0.9875 - loss: 0.0369 - val_accuracy: 0.9862 - val_loss: 0.0395
Epoch 5/5
218/218 ━━━━━━━━━━━━━━━━━━━━ 197s 908ms/step - accuracy: 0.9895 - loss: 0.0307 - val_accuracy: 0.9885 - val_loss: 0.0338

Test Accuracy: 99.11%'''