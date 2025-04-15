import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

# Define paths
TRAIN_PATH = r"C:\Users\PA Lab Dell 3\Downloads\dataset_1\train"

# Define constants
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 1  # Binary classification
NUM_EPOCHS = 10  # Increased epochs, but we'll use early stopping
CLIENT_ID = 0  # Change this for each client (0, 1, 2, 3, 4)

# Preprocess training data
def preprocess_data(train_path):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=25,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        channel_shift_range=0.2,
        validation_split=0.2  # Add validation split
    )
    
    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training'  # Specify this is the training set
    )
    
    validation_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation'  # Specify this is the validation set
    )
    
    return train_generator, validation_generator

# Create model using MobileNetV2
def create_model():
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation='relu')(x)  # Reduced from 512 to 64
    predictions = Dense(NUM_CLASSES, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Freeze only the first 100 layers of the base model
    for layer in base_model.layers[:100]:
        layer.trainable = False
    for layer in base_model.layers[100:]:
        layer.trainable = True
    
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Preprocess training data
train_generator, validation_generator = preprocess_data(TRAIN_PATH)

# Check class distribution
print("Class indices:", train_generator.class_indices)
print("Class distribution:", np.bincount(train_generator.classes))

# Create model
model = create_model()

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train model
print(f"Training Client {CLIENT_ID} for up to {NUM_EPOCHS} epochs")
history = model.fit(
    train_generator, 
    epochs=NUM_EPOCHS,
    steps_per_epoch=len(train_generator),
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate the model
evaluation = model.evaluate(validation_generator)
print(f"Validation Loss: {evaluation[0]}, Validation Accuracy: {evaluation[1]}")

# Save the entire model (including weights) in H5 format
model.save(f'client_{CLIENT_ID}_model.h5')
print(f"Model saved for client {CLIENT_ID}")

print(f"Client {CLIENT_ID} training complete.")

# Plot training history
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig(f'client_{CLIENT_ID}_training_history.png')
plt.close()

print(f"Training history plot saved for client {CLIENT_ID}")