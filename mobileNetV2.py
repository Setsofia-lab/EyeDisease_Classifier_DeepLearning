import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

def train_mobilenetv2_model(train_ds, val_ds):
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False

    model_mobilenetv2 = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4, activation='softmax')
    ])

    model_mobilenetv2.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history_mobilenetv2 = model_mobilenetv2.fit(train_ds, validation_data=val_ds, epochs=10)
    
    return model_mobilenetv2, history_mobilenetv2