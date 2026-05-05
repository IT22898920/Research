"""
======================================================================
 Coconut vs Not-Coconut Binary Classifier (v1)
 ----------------------------------------------------------------------
 Goal:  distinguish regular green coconut (class 'coconut') from
        yellow/orange thambili-style fruits (class 'not_coconut').
        The discriminating cue is dominantly colour + shape.

 Architecture:  MobileNetV2 (ImageNet) + custom head
 Strategy   :   2-phase transfer learning
                Phase 1  - frozen backbone, train head
                Phase 2  - fine-tune top 40 layers with low LR
 Output     :   ml/models/coconut_vs_notcoconut_v1/
======================================================================
"""

import os, json, time, shutil, random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_fscore_support)

# ----------------------------------------------------------------------
# 0. Reproducibility
# ----------------------------------------------------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# ----------------------------------------------------------------------
# 1. Paths
# ----------------------------------------------------------------------
BASE     = r"D:\SLIIT\Reaserch Project\CoconutHealthMonitor\Research"
SRC_DIR  = os.path.join(BASE, "ml", "data", "raw", "Coconut_NotCoconut")
SPLIT    = os.path.join(BASE, "ml", "data", "processed", "coconut_vs_notcoconut_v1")
MODEL_D  = os.path.join(BASE, "ml", "models", "coconut_vs_notcoconut_v1")
os.makedirs(MODEL_D, exist_ok=True)

CLS_DIRS = {"coconut": "coconut", "not_coconut": "not coconut"}

# ----------------------------------------------------------------------
# 2. Train / Val / Test split (70 / 15 / 15)
# ----------------------------------------------------------------------
def build_splits():
    if os.path.exists(SPLIT):
        shutil.rmtree(SPLIT)
    for split in ("train", "val", "test"):
        for cls in CLS_DIRS:
            os.makedirs(os.path.join(SPLIT, split, cls), exist_ok=True)

    counts = {}
    for cls, folder in CLS_DIRS.items():
        src = os.path.join(SRC_DIR, folder)
        files = [f for f in os.listdir(src)
                 if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
        random.shuffle(files)
        n = len(files)
        n_tr = int(n * 0.70)
        n_va = int(n * 0.15)
        for i, f in enumerate(files):
            if   i < n_tr:           dst = "train"
            elif i < n_tr + n_va:    dst = "val"
            else:                    dst = "test"
            shutil.copy2(os.path.join(src, f),
                         os.path.join(SPLIT, dst, cls, f))
        counts[cls] = {"total": n,
                       "train": n_tr,
                       "val":   n_va,
                       "test":  n - n_tr - n_va}
    return counts

print("Building dataset splits ...")
counts = build_splits()
for k, v in counts.items():
    print(f"  {k:<12}  total={v['total']:>4}  "
          f"train={v['train']:>4}  val={v['val']:>3}  test={v['test']:>3}")

# ----------------------------------------------------------------------
# 3. Data generators
# ----------------------------------------------------------------------
IMG_SIZE = 224
BATCH    = 32

# IMPORTANT - do NOT use heavy colour jitter; colour is the main cue.
train_gen_aug = ImageDataGenerator(
    preprocessing_function = applications.mobilenet_v2.preprocess_input,
    rotation_range      = 25,
    width_shift_range   = 0.15,
    height_shift_range  = 0.15,
    zoom_range          = 0.15,
    horizontal_flip     = True,
    brightness_range    = (0.85, 1.15),  # mild brightness only
    fill_mode           = "reflect"
)
plain_gen = ImageDataGenerator(
    preprocessing_function = applications.mobilenet_v2.preprocess_input
)

train_ds = train_gen_aug.flow_from_directory(
    os.path.join(SPLIT, "train"),
    target_size = (IMG_SIZE, IMG_SIZE),
    batch_size  = BATCH,
    class_mode  = "binary",
    shuffle     = True,
    seed        = SEED
)
val_ds = plain_gen.flow_from_directory(
    os.path.join(SPLIT, "val"),
    target_size = (IMG_SIZE, IMG_SIZE),
    batch_size  = BATCH,
    class_mode  = "binary",
    shuffle     = False
)
test_ds = plain_gen.flow_from_directory(
    os.path.join(SPLIT, "test"),
    target_size = (IMG_SIZE, IMG_SIZE),
    batch_size  = BATCH,
    class_mode  = "binary",
    shuffle     = False
)
class_names = list(train_ds.class_indices.keys())
print(f"\nClass indices: {train_ds.class_indices}")

# ----------------------------------------------------------------------
# 4. Model
# ----------------------------------------------------------------------
def build_model(trainable_backbone=False):
    backbone = applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
    backbone.trainable = trainable_backbone
    inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x   = backbone(inp, training=trainable_backbone)
    x   = layers.GlobalAveragePooling2D()(x)
    x   = layers.Dropout(0.30)(x)
    x   = layers.Dense(128, activation="relu")(x)
    x   = layers.Dropout(0.20)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return models.Model(inp, out), backbone

print("\nBuilding model ...")
model, backbone = build_model(trainable_backbone=False)
model.compile(
    optimizer = optimizers.Adam(1e-3),
    loss      = "binary_crossentropy",
    metrics   = ["accuracy",
                 tf.keras.metrics.Precision(name="prec"),
                 tf.keras.metrics.Recall(name="rec")]
)
model.summary()

# ----------------------------------------------------------------------
# 5. Phase 1 - frozen backbone
# ----------------------------------------------------------------------
PH1_CKPT = os.path.join(MODEL_D, "phase1_best.keras")
ph1_cbs = [
    callbacks.ModelCheckpoint(PH1_CKPT, monitor="val_accuracy",
                              save_best_only=True, mode="max"),
    callbacks.EarlyStopping(monitor="val_accuracy", patience=4,
                            restore_best_weights=True, mode="max"),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                patience=2, min_lr=1e-6)
]
print("\n========== PHASE 1 - FROZEN BACKBONE ==========")
t0 = time.time()
hist1 = model.fit(train_ds, validation_data=val_ds,
                  epochs=10, callbacks=ph1_cbs, verbose=1)
ph1_t = (time.time() - t0) / 60.0

# ----------------------------------------------------------------------
# 6. Phase 2 - fine-tune top 40 backbone layers
# ----------------------------------------------------------------------
backbone.trainable = True
for l in backbone.layers[:-40]:
    l.trainable = False

model.compile(
    optimizer = optimizers.Adam(1e-5),
    loss      = "binary_crossentropy",
    metrics   = ["accuracy",
                 tf.keras.metrics.Precision(name="prec"),
                 tf.keras.metrics.Recall(name="rec")]
)
BEST_CKPT = os.path.join(MODEL_D, "best_model.keras")
ph2_cbs = [
    callbacks.ModelCheckpoint(BEST_CKPT, monitor="val_accuracy",
                              save_best_only=True, mode="max"),
    callbacks.EarlyStopping(monitor="val_accuracy", patience=5,
                            restore_best_weights=True, mode="max"),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                patience=2, min_lr=1e-7)
]
print("\n========== PHASE 2 - FINE-TUNING ==========")
t0 = time.time()
hist2 = model.fit(train_ds, validation_data=val_ds,
                  epochs=15, callbacks=ph2_cbs, verbose=1)
ph2_t = (time.time() - t0) / 60.0

# ----------------------------------------------------------------------
# 7. Evaluate on test set
# ----------------------------------------------------------------------
print("\n========== TEST EVALUATION ==========")
best_model = models.load_model(BEST_CKPT)
test_loss, test_acc, test_prec, test_rec = best_model.evaluate(
    test_ds, verbose=1)

probs = best_model.predict(test_ds, verbose=0).flatten()
y_pred = (probs >= 0.5).astype(int)
y_true = test_ds.classes
report = classification_report(y_true, y_pred,
                               target_names=class_names,
                               digits=4, output_dict=True)
cm = confusion_matrix(y_true, y_pred).tolist()
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
print("Confusion matrix:")
print(np.array(cm))

# ----------------------------------------------------------------------
# 8. Plots
# ----------------------------------------------------------------------
def merge_hist(h1, h2):
    out = {}
    for k in h1.history:
        out[k] = h1.history[k] + h2.history.get(k, [])
    return out
H = merge_hist(hist1, hist2)

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].plot(H["accuracy"],     label="train")
ax[0].plot(H["val_accuracy"], label="val")
ax[0].axvline(len(hist1.history["accuracy"]) - 0.5,
              color="red", ls="--", alpha=0.4, label="phase 2 start")
ax[0].set_title("Accuracy")
ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Accuracy")
ax[0].legend(); ax[0].grid(alpha=0.3)

ax[1].plot(H["loss"],     label="train")
ax[1].plot(H["val_loss"], label="val")
ax[1].axvline(len(hist1.history["loss"]) - 0.5,
              color="red", ls="--", alpha=0.4)
ax[1].set_title("Loss")
ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Loss")
ax[1].legend(); ax[1].grid(alpha=0.3)

plt.suptitle("Coconut vs Not-Coconut v1 - Training History",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(MODEL_D, "training_history.png"),
            dpi=130, bbox_inches="tight")
plt.close()

plt.figure(figsize=(5, 4.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=class_names, yticklabels=class_names,
            cbar=False, annot_kws={"size": 14, "weight": "bold"})
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.title(f"Confusion Matrix - Test Acc {test_acc*100:.2f}%",
          fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(MODEL_D, "confusion_matrix.png"),
            dpi=130, bbox_inches="tight")
plt.close()

# Per-class bar plot
classes = class_names
prec_arr = [report[c]["precision"] for c in classes]
rec_arr  = [report[c]["recall"]    for c in classes]
f1_arr   = [report[c]["f1-score"]  for c in classes]
x = np.arange(len(classes)); w = 0.27
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.bar(x - w, prec_arr, w, label="Precision", color="#4C9AFF")
ax.bar(x,     rec_arr,  w, label="Recall",    color="#36B37E")
ax.bar(x + w, f1_arr,   w, label="F1",        color="#FF8B00")
for i, (p, r, f) in enumerate(zip(prec_arr, rec_arr, f1_arr)):
    ax.text(i - w, p + 0.01, f"{p:.3f}", ha="center", fontsize=8)
    ax.text(i,     r + 0.01, f"{r:.3f}", ha="center", fontsize=8)
    ax.text(i + w, f + 0.01, f"{f:.3f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(classes)
ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
ax.set_title("Per-Class Metrics", fontweight="bold")
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_D, "per_class_metrics.png"),
            dpi=130, bbox_inches="tight")
plt.close()

# Sample test predictions
def sample_grid():
    paths, labels = [], []
    for cls in classes:
        d = os.path.join(SPLIT, "test", cls)
        for f in os.listdir(d)[:6]:
            paths.append(os.path.join(d, f))
            labels.append(cls)
    fig, ax = plt.subplots(2, 6, figsize=(15, 5.5))
    for i, (p, lab) in enumerate(zip(paths, labels)):
        img = tf.keras.preprocessing.image.load_img(p, target_size=(IMG_SIZE, IMG_SIZE))
        arr = tf.keras.preprocessing.image.img_to_array(img)
        pre = applications.mobilenet_v2.preprocess_input(arr)
        prob = float(best_model.predict(pre[None, ...], verbose=0)[0][0])
        pred_cls = classes[int(prob >= 0.5)]
        ok = pred_cls == lab
        r, c = i // 6, i % 6
        ax[r, c].imshow(img); ax[r, c].axis("off")
        col = "green" if ok else "red"
        ax[r, c].set_title(f"True: {lab}\nPred: {pred_cls} ({prob:.2f})",
                           color=col, fontsize=9)
    plt.suptitle("Sample Test Predictions", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_D, "sample_predictions.png"),
                dpi=130, bbox_inches="tight")
    plt.close()
sample_grid()

# ----------------------------------------------------------------------
# 9. Persist model_info.json
# ----------------------------------------------------------------------
info = {
    "model_name":   "coconut_vs_notcoconut_v1",
    "version":      "v1",
    "architecture": "MobileNetV2 (Transfer Learning + Fine-tune top 40)",
    "task":         "binary classification - coconut vs not_coconut",
    "num_classes":  2,
    "classes":      class_names,
    "class_indices": train_ds.class_indices,
    "input_size":   [IMG_SIZE, IMG_SIZE, 3],
    "preprocessing": "mobilenet_v2.preprocess_input (scales to [-1, 1])",
    "dataset": {
        "source": "ml/data/raw/Coconut_NotCoconut",
        "split":  "70 / 15 / 15  (train / val / test)",
        "counts": counts
    },
    "augmentation": {
        "rotation_range":     25,
        "width_shift_range":  0.15,
        "height_shift_range": 0.15,
        "zoom_range":         0.15,
        "horizontal_flip":    True,
        "brightness_range":   [0.85, 1.15],
        "note": "no aggressive colour jitter - colour is the discriminating cue"
    },
    "training": {
        "phase1_epochs_run":   len(hist1.history["accuracy"]),
        "phase2_epochs_run":   len(hist2.history["accuracy"]),
        "phase1_minutes":      round(ph1_t, 2),
        "phase2_minutes":      round(ph2_t, 2),
        "total_minutes":       round(ph1_t + ph2_t, 2),
        "phase1_optimizer":    "Adam(1e-3)",
        "phase2_optimizer":    "Adam(1e-5)",
        "loss":                "binary_crossentropy",
        "fine_tuned_layers":   "top 40 of MobileNetV2"
    },
    "performance": {
        "test_loss":      float(test_loss),
        "test_accuracy":  float(test_acc),
        "test_precision": float(test_prec),
        "test_recall":    float(test_rec),
        "macro_f1":       float(report["macro avg"]["f1-score"]),
        "weighted_f1":    float(report["weighted avg"]["f1-score"]),
        "per_class": [
            {
                "class":     c,
                "precision": float(report[c]["precision"]),
                "recall":    float(report[c]["recall"]),
                "f1":        float(report[c]["f1-score"]),
                "support":   int(report[c]["support"])
            } for c in class_names
        ],
        "confusion_matrix": cm
    },
    "files": {
        "best_model":       "best_model.keras",
        "phase1_checkpoint":"phase1_best.keras",
        "training_history": "training_history.png",
        "confusion_matrix": "confusion_matrix.png",
        "per_class_metrics":"per_class_metrics.png",
        "sample_predictions":"sample_predictions.png"
    }
}
with open(os.path.join(MODEL_D, "model_info.json"), "w") as f:
    json.dump(info, f, indent=2)

print("\n======================================================================")
print(f"  TEST ACCURACY : {test_acc*100:.2f}%")
print(f"  MACRO F1      : {report['macro avg']['f1-score']*100:.2f}%")
print(f"  TOTAL TIME    : {ph1_t + ph2_t:.2f} min")
print(f"  Saved to       : {MODEL_D}")
print("======================================================================")
