# 🧠 Federated Learning for Brain Tumor Classification using MobileNetV2

This project demonstrates the use of **Federated Learning (FL)** for binary classification of brain MRI images (tumor vs. no tumor) using **MobileNetV2** architecture. It enables multiple clients to train models locally and privately on their own datasets, and later aggregates their trained weights on a central server for a global model update.

---

## 📚 Project Overview

- **Domain**: Medical Image Analysis (Brain Tumor Detection)
- **Architecture**: Client-Server
- **Base Model**: MobileNetV2 (pre-trained on ImageNet)
- **Goal**: Train a collaborative model while preserving patient data privacy
- **Approach**: Federated Learning using TensorFlow and Keras

---

## 🧩 Client-Side Workflow

Each client performs the following steps:

1. Loads and preprocesses local MRI scan images.
2. Fine-tunes a MobileNetV2-based binary classifier using data augmentation.
3. Trains the model using early stopping.
4. Saves the trained model and training history plot.

### ✅ Client Configuration

- Image size: `224x224`
- Batch size: `32`
- Epochs: `10` (with early stopping)
- Loss Function: `BinaryCrossentropy`
- Optimizer: `Adam` (lr = `0.0001`)
- Final layer: `Sigmoid` for binary classification

---

## 🌐 Server-Side Workflow

The server is responsible for:

- Loading trained `.h5` models from all clients
- Aggregating model weights using a custom **FedAvg-inspired** formula
- Creating and saving a global model
- Evaluating it using a held-out validation dataset

### 📦 Aggregation Function (FedAvg Inspired)

```python
def custom_aggregate_weights(weights_list):
    aggregated_weights = []
    for layer_weights in zip(*weights_list):
        weighted_sum = sum((w * (0.1 / (i + 0.1))) for i, w in enumerate(layer_weights))
        avg_weight = weighted_sum / sum(0.1 / (i + 0.1) for i in range(len(layer_weights)))
        aggregated_weights.append(avg_weight)
    return aggregated_weights
```

---

## 📊 Results & Performance

| Metric                   | Value      |
|--------------------------|------------|
| Global Accuracy (Max)    | **83.12%** |
| Global Validation Loss   | **0.672**  |
| Average Client Accuracy  | **~99%**   |
| Global Epoch (Peak)      | **70**     |
| Model Used               | `MobileNetV2` |

- All clients achieved high training accuracy despite diverse data distributions.
- The final aggregated model generalized well across the validation dataset.
- Training plots (`accuracy`, `loss`) for each client are saved as images (e.g., `client_X_training_history.png`).

---


## 🧪 Environment

- Python ≥ 3.8  
- TensorFlow 2.x  
- NumPy  
- Matplotlib


