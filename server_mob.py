import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define constants
NUM_CLIENTS = 6
CLIENT_MODEL_PREFIX = 'client_'
CLIENT_MODEL_SUFFIX = '_model.h5'
GLOBAL_MODEL_FILENAME = 'GLOBAL_MODEL_MOB.h5'
IMG_SIZE = 224
BATCH_SIZE = 32
VAL_PATH = r"C:\Users\PA Lab Dell 3\brain_tumer_dataset\val"

def load_client_weights(client_id):
    model_path = f"{CLIENT_MODEL_PREFIX}{client_id}{CLIENT_MODEL_SUFFIX}"
    if not os.path.exists(model_path):
        return None
    model = load_model(model_path)
    return model.get_weights()

def custom_aggregate_weights(weights_list):
    aggregated_weights = []
    for weights_per_layer in zip(*weights_list):
        layer_weights = []
        for i, weights in enumerate(weights_per_layer):
            client_weight = 0.1 / (i + 0.1)
            layer_weights.append(weights * client_weight)
        aggregated_weights.append(np.sum(layer_weights, axis=0) / sum(0.1 / (i + 0.1) for i in range(len(weights_per_layer))))
    return aggregated_weights

client_weights = [weight for weight in (load_client_weights(i) for i in range(NUM_CLIENTS)) if weight is not None]

if not client_weights:
    raise ValueError("No client models could be loaded. Please check the file paths and client numbers.")

aggregated_weights = custom_aggregate_weights(client_weights)
base_model = load_model(f"{CLIENT_MODEL_PREFIX}0{CLIENT_MODEL_SUFFIX}")
base_model.set_weights(aggregated_weights)
base_model.save(GLOBAL_MODEL_FILENAME)

def prepare_validation_data(val_path):
    if not os.path.exists(val_path):
        raise ValueError(f"Validation data directory not found: {val_path}")

    val_datagen = ImageDataGenerator(rescale=1./255)
    val_generator = val_datagen.flow_from_directory(
        val_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    return val_generator

val_generator = prepare_validation_data(VAL_PATH)
loss, accuracy = base_model.evaluate(val_generator)

print(f"Global Model Validation Loss: {loss:.4f}")
print(f"Global Model Validation Accuracy: {accuracy:.4f}")
