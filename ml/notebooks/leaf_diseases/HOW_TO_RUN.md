# How to Run Stage 1 Training Cell-by-Cell

## Quick Start

### 1. Start Jupyter Lab
```powershell
cd "C:\Users\Tharindu Nandun\Desktop\Research\Research\ml"
python -m jupyterlab
```

### 2. Open Notebook
- Navigate to: `notebooks/leaf_diseases/01_stage1_coconut_detection.ipynb`
- Double-click to open

### 3. Run Cells
**Keyboard Shortcuts:**
- `Shift + Enter` - Run current cell and move to next
- `Ctrl + Enter` - Run current cell and stay
- `Alt + Enter` - Run cell and insert new cell below

**Mouse Method:**
- Click on a cell
- Click the ▶️ (Run) button in toolbar
- Or menu: Run → Run Selected Cell

## Cell-by-Cell Guide

### Section 1: Setup (Cells 1-2)
**What it does:** Import libraries and set configuration

**Expected output:**
```
TensorFlow version: 2.20.0
GPU Available: []
CONFIGURATION SUMMARY:
  Dataset Directory: ../../data/raw/stage_1
  Input Shape: (224, 224, 3)
  Batch Size: 32
  ...
```

**Time:** ~5 seconds

---

### Section 2: Dataset Analysis (Cells 3-4)
**What it does:** Count images and create distribution charts

**Expected output:**
```
TRAINING SET:
  cocount      : 9971 images (49.93%)
  not_cocount  : 10000 images (50.07%)
  TOTAL        : 19971 images
```

**Output files:**
- `../../models/coconut_leaf_detector_stage1_v1/dataset_distribution.png`

**Time:** ~10 seconds

---

### Section 3: Data Generators (Cells 5-6)
**What it does:** Create data loaders with augmentation

**Expected output:**
```
Found 20000 images belonging to 2 classes.
Found 855 images belonging to 2 classes.
✅ Data generators created
```

**Time:** ~5 seconds

---

### Section 4: Model Building (Cells 7-9)
**What it does:** Build EfficientNetB0 model

**Expected output:**
```
Model: "functional"
_________________________________________________________________
Layer (type)                Output Shape              Param #
=================================================================
Total params: 4,410,661 (16.83 MB)
Trainable params: 361,090 (1.38 MB)
✅ Model built successfully
```

**Time:** ~10 seconds

---

### Section 5: PHASE 1 TRAINING (Cell 10) ⏰
**What it does:** Train for 25 epochs (base model frozen)

**Expected output:**
```
Epoch 1/25
625/625 ━━━━━━━━━━━━━━━━━━━━ 234s 374ms/step
  loss: 0.2345 - accuracy: 0.9012 - val_accuracy: 0.9123

Epoch 2/25
625/625 ━━━━━━━━━━━━━━━━━━━━ 228s 365ms/step
  loss: 0.1234 - accuracy: 0.9534 - val_accuracy: 0.9578
...

Epoch 25/25
625/625 ━━━━━━━━━━━━━━━━━━━━ 230s 368ms/step
  loss: 0.0567 - accuracy: 0.9823 - val_accuracy: 0.9756
```

**⚠️ IMPORTANT:**
- This cell takes **1-2 HOURS** to run!
- You'll see progress bar for each epoch
- Accuracy starts low (~50%) and improves each epoch
- By Epoch 10: should be ~90%+
- By Epoch 25: should be ~95%+

**What to watch:**
- `accuracy` should increase each epoch
- `val_accuracy` should be close to `accuracy` (gap < 10%)
- Loss should decrease

**Time:** ~60-120 minutes (CPU), ~20-30 minutes (GPU)

---

### Section 6: PHASE 2 TRAINING (Cell 11) ⏰
**What it does:** Fine-tune entire model for 25 more epochs

**Expected output:**
```
Unfreezing base model for fine-tuning
Trainable parameters: 4,049,571

Epoch 1/25
625/625 ━━━━━━━━━━━━━━━━━━━━ 245s 392ms/step
  loss: 0.0456 - accuracy: 0.9856 - val_accuracy: 0.9801
...
```

**Time:** ~60-120 minutes (CPU)

---

### Section 7: Results & Visualization (Cells 12-17)
**What it does:** Create charts and evaluate model

**Expected outputs:**

**Cell 13 - Training History Chart:**
- Shows accuracy, loss, precision, recall, AUC over 50 epochs
- Check: Training and validation lines should be close

**Cell 14 - Test Results:**
```
Test Accuracy:  0.9578 (95.78%)
Test Precision: 0.9534
Test Recall:    0.9623
Test F1-Score:  0.9578
```

**Cell 15 - Classification Report:**
```
              precision    recall  f1-score   support
     cocount     0.9534    0.9623    0.9578       425
 not_cocount     0.9612    0.9534    0.9573       430
```

**Cell 16 - Per-Class Metrics Chart:**
- Bar chart comparing Precision, Recall, F1 for each class
- Should be very similar (balanced)

**Cell 17 - Confusion Matrix:**
- Two heatmaps showing classification results

**Time:** ~2-3 minutes total

---

### Section 8: Save Results (Cells 18-20)
**What it does:** Save model info and summary

**Output files:**
- `best_model.keras` - Your trained model
- `model_info.json` - Complete results for professor
- `classification_report.txt` - Detailed metrics
- All PNG charts

**Time:** ~30 seconds

---

## Keyboard Shortcuts Reference

| Action | Windows/Linux | Mac |
|--------|--------------|-----|
| Run cell | `Shift + Enter` | `Shift + Return` |
| Run cell (stay) | `Ctrl + Enter` | `Cmd + Return` |
| Insert cell below | `B` | `B` |
| Insert cell above | `A` | `A` |
| Delete cell | `D D` (twice) | `D D` |
| Undo delete | `Z` | `Z` |
| Save notebook | `Ctrl + S` | `Cmd + S` |
| Change to Markdown | `M` | `M` |
| Change to Code | `Y` | `Y` |

## Tips

### 🎯 Best Practice:
1. **Run cells 1-9 first** (takes ~1 minute total)
2. **Check outputs** - make sure no errors
3. **Then run cell 10** (Phase 1 training)
4. **Go have lunch** ☕ - takes 1-2 hours
5. **Come back**, check results, run remaining cells

### ⚠️ If Training Stops:
- Just click on the cell and press `Shift + Enter` again
- Model checkpoints are saved, so you won't lose progress

### 💾 Save Often:
- Press `Ctrl + S` to save notebook
- Outputs are saved in the notebook file

### 📊 Watch These Metrics:
- **Accuracy:** Should reach 90%+ by Epoch 10
- **Val_accuracy:** Should be close to accuracy (within 5%)
- **Loss:** Should decrease steadily

### ✅ Success Criteria:
- Test Accuracy > 90%
- Precision ≈ Recall ≈ F1 (difference < 5%)
- Both classes have similar metrics
- Training-validation gap < 10%

## Output Files Location

All outputs saved to:
```
ml/models/coconut_leaf_detector_stage1_v1/
├── best_model.keras
├── model_info.json
├── classification_report.txt
├── per_class_metrics.csv
├── dataset_distribution.png
├── training_history.png
├── per_class_metrics.png
└── confusion_matrix.png
```

## Troubleshooting

### "Kernel is busy"
- Wait for current cell to finish
- Or click: Kernel → Interrupt Kernel

### "Out of memory"
- Reduce BATCH_SIZE in Cell 2 (try 16 instead of 32)
- Restart kernel: Kernel → Restart Kernel

### "File not found"
- Check you're in the correct directory
- Paths are relative to notebook location

### Training too slow
- Normal on CPU! GPU is 5-10x faster
- Each epoch takes 3-5 minutes on CPU
- Consider running overnight

## Next Steps

After training completes:
1. Review all visualizations
2. Check `model_info.json` for metrics
3. Share results with professor
4. Move to Stage 2: Disease Classification

## Questions?

- Check outputs in each cell
- Read the markdown explanations
- Model should achieve 95%+ accuracy
- All professor requirements will be met automatically
