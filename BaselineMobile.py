import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, accuracy_score,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize

# =====================================================
# 1. CONFIGURATION
# =====================================================
DATA_DIR = r"C:\Users\noor_\Downloads\Zahiamar\OSISO"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS_PHASE1 = 5
EPOCHS_PHASE2 = 25
SEED = 123

# =====================================================
# 2. DATA LOADING (70/10/20)
# =====================================================
full_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True
)

class_names = full_ds.class_names
num_classes = len(class_names)

ds_size = len(full_ds)
train_size = int(0.7 * ds_size)
val_size = int(0.1 * ds_size)

train_ds = full_ds.take(train_size)
remaining = full_ds.skip(train_size)
val_ds = remaining.take(val_size)
test_ds = remaining.skip(val_size)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)
test_ds = test_ds.cache().prefetch(AUTOTUNE)

# =====================================================
# 3. BASELINE MODEL (MobileNetV2)
# =====================================================
base_model = MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_shape=(128, 128, 3)
)
base_model.trainable = False

inputs = layers.Input(shape=(128, 128, 3))
x = layers.RandomFlip("horizontal")(inputs)
x = layers.RandomRotation(0.1)(x)

x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

baseline_model = models.Model(inputs, outputs)

# =====================================================
# 4. PHASE 1 – WARM-UP
# =====================================================
baseline_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

baseline_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_PHASE1
)

# =====================================================
# 5. PHASE 2 – FINE-TUNING
# =====================================================
base_model.trainable = True

baseline_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True
    )
]

baseline_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_PHASE2,
    callbacks=callbacks
)

# =====================================================
# 6. FINAL EVALUATION + ADVANCED METRICS
# =====================================================
print("\n--- BASELINE FINAL EVALUATION ---")

y_true, y_pred, y_prob = [], [], []

for imgs, labels in test_ds:
    probs = baseline_model.predict(imgs, verbose=0)
    y_prob.extend(probs)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(probs, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_prob = np.array(y_prob)

accuracy = accuracy_score(y_true, y_pred)
print(f"Baseline Test Accuracy: {accuracy*100:.2f}%")

# =====================================================
# 7. CONFUSION MATRIX
# =====================================================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="mako",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.title(f"Baseline MobileNetV2 (Acc: {accuracy*100:.2f}%)")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

# =====================================================
# 8. FULL METRICS TABLE (WORD-READY)
# =====================================================
y_true_bin = label_binarize(y_true, classes=range(num_classes))

rows = []

for i, cls in enumerate(class_names):
    TP = cm[i, i]
    FP = cm[:, i].sum() - TP
    FN = cm[i, :].sum() - TP
    TN = cm.sum() - (TP + FP + FN)

    precision = precision_score(y_true, y_pred, average=None)[i]
    recall = recall_score(y_true, y_pred, average=None)[i]
    f1 = f1_score(y_true, y_pred, average=None)[i]

    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    npv = TN / (TN + FN) if (TN + FN) > 0 else 0

    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)

    pr_auc = average_precision_score(y_true_bin[:, i], y_prob[:, i])

    rows.append([
        cls,
        precision * 100,
        recall * 100,
        f1 * 100,
        specificity * 100,
        accuracy * 100,
        roc_auc,
        npv * 100,
        pr_auc
    ])

metrics_table = pd.DataFrame(rows, columns=[
    "Class",
    "Precision (%)",
    "Recall / Sensitivity (%)",
    "F1-score (%)",
    "Specificity (%)",
    "Accuracy (%)",
    "AUC",
    "NPV (%)",
    "PR-AUC"
])

# Macro-average row
macro = metrics_table.mean(numeric_only=True)
macro["Class"] = "Macro Average"
metrics_table = pd.concat(
    [metrics_table, pd.DataFrame([macro])], ignore_index=True
)

print("\n=== BASELINE PERFORMANCE TABLE ===")
print(metrics_table.round(4).to_string(index=False))

# Save for Word / Excel
metrics_table.to_csv("Baseline_MobileNetV2_Metrics.csv", index=False)
# =====================================================
# 9. TRAINING CURVES: ACCURACY & LOSS
# =====================================================
plt.figure(figsize=(12, 5))

# Accuracy Curve
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Training & Validation Accuracy (Baseline MobileNetV2)")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(alpha=0.3)

# Loss Curve
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training & Validation Loss (Baseline MobileNetV2)")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()
# =====================================================
# 10. ROC–AUC CURVES (ONE-vs-REST)
# =====================================================
plt.figure(figsize=(8, 7))

for i, cls in enumerate(class_names):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(
        fpr, tpr,
        label=f"{cls} (AUC = {roc_auc:.3f})",
        linewidth=2
    )

# Random classifier line
plt.plot([0, 1], [0, 1], "k--", alpha=0.6)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC–AUC Curves (Baseline MobileNetV2)")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()