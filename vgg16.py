import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models

def train_vgg16_model(train_ds, val_ds):
    base_model = VGG16(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False

    model_vgg16 = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4, activation='softmax')  # 4 classes
    ])

    model_vgg16.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history_vgg16 = model_vgg16.fit(train_ds, validation_data=val_ds, epochs=10)
    
    return model_vgg16, history_vgg16