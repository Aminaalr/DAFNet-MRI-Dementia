# ======================================================
# 1. IMPORTS
# ======================================================
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
    roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize

# ======================================================
# 2. CONFIGURATION
# ======================================================
DATA_DIR = r"C:\Users\noor_\Downloads\Zahiamar\OSISO"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS_PHASE1 = 5
EPOCHS_PHASE2 = 25
SEED = 123

# ======================================================
# 3. DATA LOADING
# ======================================================
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

# ======================================================
# 4. ATTENTION & POOLING LAYERS
# ======================================================
def coordination_attention(x, reduction=32):
    h, w, c = x.shape[1], x.shape[2], x.shape[3]
    x_h = layers.Lambda(lambda t: tf.reduce_mean(t, axis=2, keepdims=True))(x)
    x_w = layers.Lambda(lambda t: tf.reduce_mean(t, axis=1, keepdims=True))(x)
    x_w = layers.Permute((2, 1, 3))(x_w)
    y = layers.Concatenate(axis=1)([x_h, x_w])
    mip = max(8, c // reduction)
    y = layers.Conv2D(mip, 1, activation="relu", padding="same")(y)
    y = layers.BatchNormalization()(y)
    x_h, x_w = layers.Lambda(lambda t: tf.split(t, [h, w], axis=1))(y)
    x_w = layers.Permute((2, 1, 3))(x_w)
    a_h = layers.Conv2D(c, 1, activation="sigmoid")(x_h)
    a_w = layers.Conv2D(c, 1, activation="sigmoid")(x_w)
    return layers.Multiply()([x, a_h, a_w])

def eca_block(x, gamma=2, b=1):
    channels = x.shape[-1]
    k = int(abs((np.log2(channels) + b) / gamma))
    k = k if k % 2 else k + 1
    y = layers.GlobalAveragePooling2D()(x)
    y = layers.Reshape((channels, 1))(y)
    y = layers.Conv1D(1, k, padding="same", use_bias=False)(y)
    y = layers.Activation("sigmoid")(y)
    y = layers.Reshape((1, 1, channels))(y)
    return layers.Multiply()([x, y])

class GeMPooling(layers.Layer):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = tf.Variable(p, trainable=True)
        self.eps = eps

    def call(self, x):
        x = tf.clip_by_value(x, self.eps, tf.reduce_max(x))
        return tf.pow(tf.reduce_mean(tf.pow(x, self.p), axis=[1, 2]), 1. / self.p)

# ======================================================
# 5. BUILD MODEL
# ======================================================
base_model = MobileNetV2(include_top=False, weights="imagenet",
                         input_shape=(128, 128, 3))
base_model.trainable = False

f1 = base_model.get_layer("block_6_expand_relu").output
f2 = base_model.get_layer("block_13_expand_relu").output
f3 = base_model.get_layer("out_relu").output

f1 = layers.AveragePooling2D(4)(f1)
f2 = layers.AveragePooling2D(2)(f2)

x = layers.Concatenate()([f1, f2, f3])
x = coordination_attention(x)
x = eca_block(x)
x = GeMPooling()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(256, activation="swish")(x)
x = layers.Dropout(0.45)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(base_model.input, outputs)

# ======================================================
# 6. TRAINING
# ======================================================
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=["accuracy"]
)
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_PHASE1)

for layer in base_model.layers[-40:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(5e-5),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=["accuracy"]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True
    )
]

model.fit(train_ds, validation_data=val_ds,
          epochs=EPOCHS_PHASE2, callbacks=callbacks)

# ======================================================
# 7. EVALUATION + ALL METRICS
# ======================================================
y_true, y_prob = [], []

for imgs, labels in test_ds:
    y_prob.extend(model.predict(imgs, verbose=0))
    y_true.extend(np.argmax(labels.numpy(), axis=1))

y_true = np.array(y_true)
y_prob = np.array(y_prob)
y_pred = np.argmax(y_prob, axis=1)
y_true_bin = label_binarize(y_true, classes=range(num_classes))

cm = confusion_matrix(y_true, y_pred)

rows = []
overall_acc = accuracy_score(y_true, y_pred)

for i, cls in enumerate(class_names):
    TP = cm[i, i]
    FP = cm[:, i].sum() - TP
    FN = cm[i, :].sum() - TP
    TN = cm.sum() - (TP + FP + FN)

    precision = precision_score(y_true, y_pred, average=None)[i]
    recall = recall_score(y_true, y_pred, average=None)[i]
    f1 = f1_score(y_true, y_pred, average=None)[i]

    specificity = TN / (TN + FP)
    npv = TN / (TN + FN)

    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    pr_auc = average_precision_score(y_true_bin[:, i], y_prob[:, i])

    rows.append([
        cls,
        precision * 100,
        recall * 100,
        f1 * 100,
        specificity * 100,
        overall_acc * 100,
        roc_auc,
        npv * 100,
        pr_auc
    ])

metrics_table = pd.DataFrame(rows, columns=[
    "Class", "Precision (%)", "Recall (%)", "F1-score (%)",
    "Specificity (%)", "Accuracy (%)", "AUC", "NPV (%)", "PR-AUC"
])

macro = metrics_table.mean(numeric_only=True)
macro["Class"] = "Macro Average"
metrics_table = pd.concat(
    [metrics_table, pd.DataFrame([macro])], ignore_index=True
)

print("\n=== FINAL PERFORMANCE TABLE ===")
print(metrics_table.round(4).to_string(index=False))

metrics_table.to_csv("DAFNet_Full_Metrics.csv", index=False)

# ======================================================
# 8. ROC & PR CURVES
# ======================================================
plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
for i in range(num_classes):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
    plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={auc(fpr, tpr):.3f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curves")
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
for i in range(num_classes):
    p, r, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
    plt.plot(r, p, label=f"{class_names[i]}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("PR Curves")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()
