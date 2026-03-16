"""
Leaf Health Detection Model - v4  (Final Fixed)
=================================================
Fixes applied:
  1. MobileNetV2 + correct preprocess_input  (EfficientNetB0 had range bug)
  2. 60/20/20 split on originals -> 20 test images (was only 16, too small)
  3. CrossEntropy + label smoothing (simpler, more stable than Focal Loss)
  4. Val = augmented from val-originals (gives more val samples)
  5. Test = original images only (true unseen evaluation)

Dataset:
  Originals  : Healthy_co (55) + Unhealthy_Co (45) = 100 images
  Augmented  : healthy_augmented (1100) + unhealthy_augmented (900) = 2000

Split by original image (no leakage):
  60% train originals -> use their aug versions for training
  20% val originals   -> use their aug versions for validation
  20% test originals  -> use ORIGINAL images for test evaluation

Arch  : MobileNetV2 (ImageNet pretrained)
Loss  : Categorical CrossEntropy + Label Smoothing 0.05
Output: ml/models/leaf_health_v4/
"""

import os, json, time, random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
import warnings
warnings.filterwarnings('ignore')

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DATA      = r'C:\Users\USER\Documents\GizmoraGit\Research\ml\data\raw\new_data'
HEALTHY_ORIG   = os.path.join(BASE_DATA, 'Healthy_co')
UNHEALTHY_ORIG = os.path.join(BASE_DATA, 'Unhealthy_Co')
HEALTHY_AUG    = os.path.join(BASE_DATA, 'healthy_augmented')
UNHEALTHY_AUG  = os.path.join(BASE_DATA, 'unhealthy_augmented')
OUTPUT_DIR     = r'C:\Users\USER\Documents\GizmoraGit\Research\ml\models\leaf_health_v4'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────────────────────
IMG_SIZE      = 224
BATCH_SIZE    = 32
PHASE1_EPOCHS = 40
PHASE2_EPOCHS = 30
LR_PHASE1     = 1e-3
LR_PHASE2     = 3e-5
DROPOUT       = 0.5
L2_REG        = 1e-4
FINE_TUNE_AT  = 50
LABEL_SMOOTH  = 0.05
CLASSES       = ['healthy', 'unhealthy']
NUM_CLASSES   = 2

print("=" * 65)
print("  Leaf Health v4  —  Training  (MobileNetV2 + Fixed Split)")
print("=" * 65)
print(f"  GPU : {tf.config.list_physical_devices('GPU')}")
print(f"  TF  : {tf.__version__}")

# ══════════════════════════════════════════════════════════════════════════════
# 1.  Collect original stems
# ══════════════════════════════════════════════════════════════════════════════
EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def get_stems(folder):
    return sorted([os.path.splitext(f)[0]
                   for f in os.listdir(folder)
                   if os.path.splitext(f)[1].lower() in EXT])

def find_orig(folder, stem):
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
        p = os.path.join(folder, stem + ext)
        if os.path.exists(p):
            return p
    return None

def find_aug(aug_folder, stem):
    return [os.path.join(aug_folder, f)
            for f in os.listdir(aug_folder)
            if os.path.splitext(f)[0].startswith(stem + '_aug')
            and os.path.splitext(f)[1].lower() in EXT]

healthy_stems   = get_stems(HEALTHY_ORIG)
unhealthy_stems = get_stems(UNHEALTHY_ORIG)
print(f"\n[Originals]  healthy={len(healthy_stems)}  unhealthy={len(unhealthy_stems)}")

# ── Split originals  60 / 20 / 20 ─────────────────────────────────────────────
h_train, h_tmp = train_test_split(healthy_stems,   test_size=0.40, random_state=SEED)
h_val,  h_test = train_test_split(h_tmp,            test_size=0.50, random_state=SEED)

u_train, u_tmp = train_test_split(unhealthy_stems,  test_size=0.40, random_state=SEED)
u_val,  u_test = train_test_split(u_tmp,             test_size=0.50, random_state=SEED)

print(f"[Split 60/20/20]")
print(f"  healthy   train={len(h_train)}  val={len(h_val)}  test={len(h_test)}")
print(f"  unhealthy train={len(u_train)}  val={len(u_val)}  test={len(u_test)}")

# ── Collect paths ──────────────────────────────────────────────────────────────
def collect_aug(h_stems, u_stems, h_aug_dir, u_aug_dir):
    paths, labels = [], []
    for s in h_stems:
        for p in find_aug(h_aug_dir, s): paths.append(p); labels.append(0)
    for s in u_stems:
        for p in find_aug(u_aug_dir, s): paths.append(p); labels.append(1)
    return paths, labels

def collect_orig(h_stems, u_stems):
    paths, labels = [], []
    for s in h_stems:
        p = find_orig(HEALTHY_ORIG, s)
        if p: paths.append(p); labels.append(0)
    for s in u_stems:
        p = find_orig(UNHEALTHY_ORIG, s)
        if p: paths.append(p); labels.append(1)
    return paths, labels

# Train & Val -> augmented;  Test -> originals
train_p, train_l = collect_aug(h_train, u_train, HEALTHY_AUG, UNHEALTHY_AUG)
val_p,   val_l   = collect_aug(h_val,   u_val,   HEALTHY_AUG, UNHEALTHY_AUG)
test_p,  test_l  = collect_orig(h_test, u_test)

print(f"\n[Dataset]")
print(f"  Train (aug) : {len(train_p)}  -> healthy={train_l.count(0)}  unhealthy={train_l.count(1)}")
print(f"  Val   (aug) : {len(val_p)}    -> healthy={val_l.count(0)}   unhealthy={val_l.count(1)}")
print(f"  Test  (orig): {len(test_p)}   -> healthy={test_l.count(0)}   unhealthy={test_l.count(1)}")

# ── Class weights ──────────────────────────────────────────────────────────────
total = len(train_l); c0 = train_l.count(0); c1 = train_l.count(1)
class_weight = {0: total/(2*c0), 1: total/(2*c1)}
print(f"[Weights] healthy={class_weight[0]:.4f}  unhealthy={class_weight[1]:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 2.  tf.data pipeline — MobileNetV2 preprocess_input scales to [-1, 1]
# ══════════════════════════════════════════════════════════════════════════════
def parse_image(path, label):
    raw = tf.io.read_file(path)
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)          # scales [0,255] -> [-1, 1]
    return img, label

def augment_fn(img, label):
    img_255 = (img + 1.0) * 127.5       # back to [0, 255]
    img_255 = tf.image.random_flip_left_right(img_255)
    img_255 = tf.image.random_flip_up_down(img_255)
    img_255 = tf.image.random_brightness(img_255, max_delta=20)
    img_255 = tf.clip_by_value(img_255, 0.0, 255.0)
    img = preprocess_input(img_255)     # back to [-1, 1]
    return img, label

AUTOTUNE = tf.data.AUTOTUNE

def make_ds(paths, labels, training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(parse_image, num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.map(augment_fn, num_parallel_calls=AUTOTUNE)
        ds = ds.shuffle(len(paths), seed=SEED)
    return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

train_ds = make_ds(train_p, train_l, training=True)
val_ds   = make_ds(val_p,   val_l)
test_ds  = make_ds(test_p,  test_l)

# ══════════════════════════════════════════════════════════════════════════════
# 3.  Model — MobileNetV2
# ══════════════════════════════════════════════════════════════════════════════
def build_model():
    base = MobileNetV2(
        include_top=False, weights='imagenet',
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base.trainable = False

    inp = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x   = base(inp, training=False)
    x   = layers.GlobalAveragePooling2D()(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(DROPOUT)(x)
    x   = layers.Dense(256, activation='relu',
                       kernel_regularizer=regularizers.l2(L2_REG))(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(DROPOUT * 0.6)(x)
    x   = layers.Dense(128, activation='relu',
                       kernel_regularizer=regularizers.l2(L2_REG))(x)
    x   = layers.Dropout(DROPOUT * 0.4)(x)
    out = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    return keras.Model(inp, out), base

model, base_model = build_model()
print(f"\n[Model] MobileNetV2  params={model.count_params():,}")

# ══════════════════════════════════════════════════════════════════════════════
# 4.  Phase 1  —  Head training
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  PHASE 1  —  Head training  (base frozen)")
print("=" * 65)

model.compile(
    optimizer=keras.optimizers.Adam(LR_PHASE1),
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTH),
    metrics=['accuracy']
)

def to_onehot(ds):
    return ds.map(lambda x, y: (x, tf.one_hot(y, NUM_CLASSES)))

train_oh = to_onehot(train_ds)
val_oh   = to_onehot(val_ds)
test_oh  = to_onehot(test_ds)

cb1 = [
    keras.callbacks.ModelCheckpoint(
        os.path.join(OUTPUT_DIR, 'phase1_best.keras'),
        monitor='val_accuracy', save_best_only=True, verbose=1),
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=10,
        restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
]

t0 = time.time()
h1 = model.fit(train_oh, epochs=PHASE1_EPOCHS, validation_data=val_oh,
               class_weight=class_weight, callbacks=cb1, verbose=1)
p1_time = (time.time() - t0) / 60
print(f"\nPhase 1 done — best val_accuracy: {max(h1.history['val_accuracy'])*100:.2f}%")

# ══════════════════════════════════════════════════════════════════════════════
# 5.  Phase 2  —  Fine-tune
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print(f"  PHASE 2  —  Fine-tune last {FINE_TUNE_AT} layers")
print("=" * 65)

base_model.trainable = True
for layer in base_model.layers[:-FINE_TUNE_AT]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(LR_PHASE2, clipnorm=1.0),
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTH),
    metrics=['accuracy']
)

cb2 = [
    keras.callbacks.ModelCheckpoint(
        os.path.join(OUTPUT_DIR, 'best_model.keras'),
        monitor='val_accuracy', save_best_only=True, verbose=1),
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=12,
        restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=4, min_lr=1e-7, verbose=1)
]

t1 = time.time()
h2 = model.fit(train_oh, epochs=PHASE2_EPOCHS, validation_data=val_oh,
               class_weight=class_weight, callbacks=cb2, verbose=1)
p2_time = (time.time() - t1) / 60
total_t  = p1_time + p2_time
print(f"\nPhase 2 done — best val_accuracy: {max(h2.history['val_accuracy'])*100:.2f}%")

# ══════════════════════════════════════════════════════════════════════════════
# 6.  Test evaluation
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  TEST EVALUATION  (original images)")
print("=" * 65)

probs  = model.predict(test_ds, verbose=0)
y_pred = np.argmax(probs, axis=1)
y_true = np.array(test_l)

acc  = np.mean(y_pred == y_true)
prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
rec  = recall_score(y_true,  y_pred, average='macro', zero_division=0)
f1   = f1_score(y_true, y_pred, average='macro', zero_division=0)

print(f"\n  Accuracy       : {acc*100:.2f}%")
print(f"  Macro Precision: {prec*100:.2f}%")
print(f"  Macro Recall   : {rec*100:.2f}%")
print(f"  Macro F1       : {f1*100:.2f}%")
print(f"  Training time  : {total_t:.1f} min")
print()
print(classification_report(y_true, y_pred, target_names=CLASSES))

rpt   = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True)
h_p, h_r, h_f = rpt['healthy']['precision'], rpt['healthy']['recall'], rpt['healthy']['f1-score']
u_p, u_r, u_f = rpt['unhealthy']['precision'], rpt['unhealthy']['recall'], rpt['unhealthy']['f1-score']

# ══════════════════════════════════════════════════════════════════════════════
# 7.  Plots
# ══════════════════════════════════════════════════════════════════════════════
acc_h   = h1.history['accuracy']     + h2.history['accuracy']
vacc_h  = h1.history['val_accuracy'] + h2.history['val_accuracy']
loss_h  = h1.history['loss']         + h2.history['loss']
vloss_h = h1.history['val_loss']     + h2.history['val_loss']
ep      = range(1, len(acc_h) + 1)
p1_end  = len(h1.history['accuracy'])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Leaf Health v4  —  Training History', fontsize=14)
for ax, (tr, vl, ttl) in zip(axes,
    [(acc_h, vacc_h, 'Accuracy'), (loss_h, vloss_h, 'Loss')]):
    ax.plot(ep, tr, label='Train', color='#2196F3')
    ax.plot(ep, vl, label='Val',   color='#4CAF50')
    ax.axvline(p1_end, color='gray', ls='--', alpha=0.6, label='Fine-tune')
    ax.set_title(ttl); ax.legend(); ax.set_xlabel('Epoch')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'training_history.png'), dpi=150); plt.close()

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASSES, yticklabels=CLASSES)
plt.title(f'Confusion Matrix  —  Acc {acc*100:.2f}%')
plt.ylabel('True'); plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=150); plt.close()

plt.figure(figsize=(8, 4))
plt.hist(probs[y_true==0, 0], bins=15, alpha=0.7, label='Healthy -> P(healthy)',    color='#4CAF50')
plt.hist(probs[y_true==1, 1], bins=15, alpha=0.7, label='Unhealthy -> P(unhealthy)', color='#F44336')
plt.title('Confidence Distribution'); plt.xlabel('Confidence'); plt.ylabel('Count')
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confidence_distribution.png'), dpi=150); plt.close()
print("[Saved] plots")

# ══════════════════════════════════════════════════════════════════════════════
# 8.  model_info.json
# ══════════════════════════════════════════════════════════════════════════════
model_info = {
    "model_name": "leaf_health_v4",
    "architecture": "MobileNetV2",
    "version": "v4",
    "classes": CLASSES,
    "num_classes": NUM_CLASSES,
    "input_shape": [IMG_SIZE, IMG_SIZE, 3],
    "preprocessing": "MobileNetV2 preprocess_input  ->  [-1, 1]",
    "split_strategy": "60/20/20 on originals; train+val=aug, test=original images",
    "training": {
        "phase1_epochs": len(h1.history['accuracy']),
        "phase2_epochs": len(h2.history['accuracy']),
        "batch_size": BATCH_SIZE,
        "lr_phase1": LR_PHASE1,
        "lr_phase2": LR_PHASE2,
        "loss_function": f"CategoricalCrossentropy + LabelSmooth {LABEL_SMOOTH}",
        "optimizer": "Adam (clipnorm=1.0 phase2)",
        "fine_tune_layers": FINE_TUNE_AT,
        "dropout": DROPOUT,
        "l2_reg": L2_REG,
        "training_time_minutes": round(total_t, 1),
        "best_val_accuracy": round(float(max(h1.history['val_accuracy'] + h2.history['val_accuracy'])), 6),
        "final_train_accuracy": float(acc_h[-1]),
        "final_val_accuracy":   float(vacc_h[-1])
    },
    "data": {
        "healthy_originals":   len(healthy_stems),
        "unhealthy_originals": len(unhealthy_stems),
        "train_aug": len(train_p),
        "val_aug":   len(val_p),
        "test_orig": len(test_p),
        "split": "60% train (aug) / 20% val (aug) / 20% test (orig)"
    },
    "test_performance": {
        "accuracy":            round(float(acc),  6),
        "macro_precision":     round(float(prec), 6),
        "macro_recall":        round(float(rec),  6),
        "macro_f1":            round(float(f1),   6),
        "healthy_precision":   round(float(h_p),  6),
        "healthy_recall":      round(float(h_r),  6),
        "healthy_f1":          round(float(h_f),  6),
        "unhealthy_precision": round(float(u_p),  6),
        "unhealthy_recall":    round(float(u_r),  6),
        "unhealthy_f1":        round(float(u_f),  6)
    },
    "class_weights": {str(k): round(v, 6) for k, v in class_weight.items()},
    "api_endpoint": "/predict/leaf-health"
}
with open(os.path.join(OUTPUT_DIR, 'model_info.json'), 'w') as f:
    json.dump(model_info, f, indent=2)
print("[Saved] model_info.json")

# ══════════════════════════════════════════════════════════════════════════════
# 9.  Final Summary
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  COMPLETE — leaf_health_v4")
print("=" * 65)
print(f"  Test Accuracy   : {acc*100:.2f}%")
print(f"  Macro F1        : {f1*100:.2f}%")
print(f"  healthy   F1    : {h_f*100:.2f}%  (P={h_p*100:.1f}%  R={h_r*100:.1f}%)")
print(f"  unhealthy F1    : {u_f*100:.2f}%  (P={u_p*100:.1f}%  R={u_r*100:.1f}%)")
print(f"  Best Val Acc    : {max(h1.history['val_accuracy']+h2.history['val_accuracy'])*100:.2f}%")
print(f"  Training time   : {total_t:.1f} min")
print(f"  Saved to        : {OUTPUT_DIR}")
print("=" * 65)
