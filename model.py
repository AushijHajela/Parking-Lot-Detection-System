import tensorflow as tf
from tensorflow.keras import layers, models #type: ignore

def build_cnn(input_shape=(128,128,3)):
    model = models.Sequential([
        layers.Conv2D(32,(3,3),activation='relu',input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)), 
        
        layers.Conv2D(64,(3,3),activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        
        layers.Conv2D(128,(3,3),activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        
        layers.Flatten(),
        layers.Dense(128,activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1,activation='sigmoid')  
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model

if __name__ == "__main__":
    model = build_cnn()
    model.summary()
    
        

