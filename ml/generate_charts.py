import os
import json
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing import image

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'coconut_white_fly_v1')
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'white_fly')
TRAIN_DIR = os.path.join(DATA_DIR, 'Training')
TEST_DIR = os.path.join(DATA_DIR, 'test')

class_names = ['healthy', 'not_coconut', 'white_fly']
IMG_SIZE = 224

# Load results
with open(os.path.join(MODEL_DIR, 'evaluation_results.json'), 'r') as f:
    results = json.load(f)

with open(os.path.join(MODEL_DIR, 'model_info.json'), 'r') as f:
    model_info = json.load(f)

print('Generating charts...')

# 1. Dataset Distribution Chart
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
splits = [('Training', results['train_counts']), ('Validation', results['val_counts']), ('Test', results['test_counts'])]
colors = ['#2ecc71', '#e74c3c', '#3498db']

for idx, (split_name, counts) in enumerate(splits):
    classes = list(counts.keys())
    values = list(counts.values())
    bars = axes[idx].bar(classes, values, color=colors)
    axes[idx].set_title(f'{split_name} Set', fontweight='bold', fontsize=12)
    axes[idx].set_ylabel('Number of Images')
    for bar, val in zip(bars, values):
        axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, str(val), ha='center', fontweight='bold')
    axes[idx].tick_params(axis='x', rotation=45)

plt.suptitle('Dataset Distribution by Class', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'dataset_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print('1. Dataset distribution saved')

# 2. Sample Images
fig, axes = plt.subplots(3, 5, figsize=(15, 10))
fig.suptitle('Sample Images from Training Set', fontsize=16, fontweight='bold')
random.seed(42)

for row, cls in enumerate(class_names):
    cls_dir = os.path.join(TRAIN_DIR, cls)
    if os.path.exists(cls_dir):
        images_list = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        sample_imgs = random.sample(images_list, min(5, len(images_list)))
        for col, img_name in enumerate(sample_imgs):
            img_path = os.path.join(cls_dir, img_name)
            img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
            axes[row, col].imshow(img)
            axes[row, col].axis('off')
            if col == 0:
                axes[row, col].set_title(cls.replace('_', ' ').title(), fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'sample_images.png'), dpi=150, bbox_inches='tight')
plt.close()
print('2. Sample images saved')

# 3. Confusion Matrix
cm = np.array(results['confusion_matrix'])
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=axes[0], annot_kws={'size': 14, 'fontweight': 'bold'})
axes[0].set_title('Confusion Matrix (Counts)', fontweight='bold', fontsize=12)
axes[0].set_xlabel('Predicted', fontsize=11)
axes[0].set_ylabel('True', fontsize=11)

cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=axes[1], annot_kws={'size': 14, 'fontweight': 'bold'})
axes[1].set_title('Confusion Matrix (%)', fontweight='bold', fontsize=12)
axes[1].set_xlabel('Predicted', fontsize=11)
axes[1].set_ylabel('True', fontsize=11)

plt.suptitle('Confusion Matrix - White Fly Model v1', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.close()
print('3. Confusion matrix saved')

# 4. Per-Class Metrics Bar Chart
precision = np.array(results['precision'])
recall = np.array(results['recall'])
f1 = np.array(results['f1'])

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(class_names))
width = 0.25

bars1 = ax.bar(x - width, precision * 100, width, label='Precision', color='#3498db')
bars2 = ax.bar(x, recall * 100, width, label='Recall', color='#2ecc71')
bars3 = ax.bar(x + width, f1 * 100, width, label='F1-Score', color='#e74c3c')

ax.set_xlabel('Class', fontsize=12)
ax.set_ylabel('Score (%)', fontsize=12)
ax.set_title('Per-Class Metrics - White Fly Model v1', fontweight='bold', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels([c.replace('_', ' ').title() for c in class_names])
ax.legend()
ax.set_ylim([80, 105])
ax.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'per_class_metrics.png'), dpi=150, bbox_inches='tight')
plt.close()
print('4. Per-class metrics saved')

# 5. Correct Predictions
y_true = np.array(results['y_true'])
y_pred = np.array(results['y_pred'])
predictions = np.array(results['predictions'])
filenames = results['filenames']

correct_idx = [i for i in range(len(y_true)) if y_true[i] == y_pred[i]]
wrong_idx = [i for i in range(len(y_true)) if y_true[i] != y_pred[i]]

random.seed(42)
fig, axes = plt.subplots(2, 5, figsize=(15, 7))
fig.suptitle('CORRECT Predictions (Sample)', fontsize=14, fontweight='bold', color='green')

sample_correct = random.sample(correct_idx, min(10, len(correct_idx)))
for idx, i in enumerate(sample_correct):
    row, col = idx // 5, idx % 5
    img_path = os.path.join(TEST_DIR, filenames[i])
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    axes[row, col].imshow(img)
    axes[row, col].axis('off')
    pred_label = class_names[y_pred[i]]
    conf = predictions[i][y_pred[i]] * 100
    axes[row, col].set_title(f'{pred_label}\n{conf:.1f}%', fontsize=10, color='green')

plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'correct_predictions.png'), dpi=150, bbox_inches='tight')
plt.close()
print('5. Correct predictions saved')

# 6. Wrong Predictions
if len(wrong_idx) > 0:
    n_wrong = min(10, len(wrong_idx))
    rows = (n_wrong + 4) // 5
    fig, axes = plt.subplots(rows, 5, figsize=(15, 3.5*rows))
    fig.suptitle('WRONG Predictions', fontsize=14, fontweight='bold', color='red')

    if rows == 1:
        axes = axes.reshape(1, -1)

    for idx, i in enumerate(wrong_idx[:n_wrong]):
        row, col = idx // 5, idx % 5
        img_path = os.path.join(TEST_DIR, filenames[i])
        img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
        axes[row, col].imshow(img)
        axes[row, col].axis('off')
        true_label = class_names[y_true[i]]
        pred_label = class_names[y_pred[i]]
        axes[row, col].set_title(f'True: {true_label}\nPred: {pred_label}', fontsize=9, color='red')

    for idx in range(n_wrong, rows * 5):
        row, col = idx // 5, idx % 5
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'wrong_predictions.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('6. Wrong predictions saved')

print('All charts generated successfully!')
