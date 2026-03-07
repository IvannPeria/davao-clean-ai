import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import os

# 1. Setup Data Loading
# We use 'rescale' to normalize pixel values (0-255 becomes 0-1)
# 'validation_split' keeps 20% of images for testing the AI's accuracy
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

dataset_path = 'dataset'

train_gen = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# 2. Build the Model using Transfer Learning
# We use MobileNetV2 because it's fast and lightweight
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False 

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(train_gen.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 3. Start Training
print(f"Detected classes: {train_gen.class_indices}")
print("Training started... this may take a few minutes.")

model.fit(train_gen, epochs=5) 

# 4. Save the "Brain"
model.save('garbage_model.h5')
print("✅ Training Complete! 'garbage_model.h5' has been updated.")