# Relation Network

Implementation of **"Learning to Compare: Relation Network for Few-Shot Learning"** (CVPR 2018), with a cross-attention extension replacing the original depth-wise concatenation.

---

## Overview

The Relation Network learns to classify images in a few-shot setting by learning both an embedding and a similarity metric end-to-end. This implementation adds a **cross-attention module** between the embedding and relation scoring steps, where the query and support features attend to each other before comparison.

---

## Configuration

All hyperparameters live in the `Config` class at the top of `main.py`:

| Parameter | Default | Description |
|---|---|---|
| `N_WAY` | 5 | Number of classes per episode |
| `K_SHOT` | 1 | Support examples per class |
| `N_QUERY` | 15 | Query images per class per episode |
| `N_EPISODES` | 300,000 | Total training episodes |
| `LR` | 1e-3 | Initial learning rate (halved every `LR_STEP`) |
| `LR_STEP` | 100,000 | LR decay step |
| `GRAD_CLIP` | 0.5 | Gradient clipping norm |
| `EVAL_EVERY` | 1,000 | Evaluate every N episodes |
| `EVAL_EPS` | 1,000 | Episodes per evaluation run |
| `DATASET` | `"omniglot"` | `"omniglot"` or `"miniimagenet"` |
| `SEED` | 42 | Random seed |
| `SAVE_PATH` | `relation_net_best_working.pth` | Checkpoint save path |

To switch to miniImageNet, change `DATASET = "miniimagenet"` in `Config`.

---

## Setup
Training logs (loss, accuracy) are tracked via [Weights & Biases](https://wandb.ai). Set your API key:
```bash
export WANDB_API_KEY=your_key_here
```

We install all the dependencies from requirements.txt
```bash
python -m pip install -r requirements.txt
```

## Training

```bash
python main.py
```

The best model checkpoint is saved to `relation_net_best.pth` whenever validation accuracy improves.

---



## Inference UI

A Gradio-based demo is available in `ui.py`:

```bash
python ui.py
```

Upload one image per class as the support set (5 classes), optionally name each class, upload a query image, and click **Classify**. The model outputs a relation score for each class and highlights the predicted one.

---

## Model Architecture

### Embedding module (shared weights)
Four convolutional blocks (Conv → BatchNorm → ReLU). The first two include 2×2 max-pooling; the last two preserve spatial dimensions for the relation module.

### Cross-attention module *(extension)*
Support and query feature maps are flattened into token sequences and passed through 4-head multi-head attention — query attends to support, and support attends to query — with residual connections and LayerNorm. The enriched maps are concatenated depth-wise into a `[128, H, W]` tensor.

### Relation module
Two more convolutional blocks (with pooling), followed by two fully connected layers (ReLU: 64→8, Sigmoid: 8→1), producing a relation score `s ∈ [0, 1]`.

### Loss
Mean Squared Error against one-hot targets — matched pairs target 1, mismatched pairs target 0.

---

## Results (Omniglot, 5-way 1-shot)

| Model | Eval Accuracy |
|---|---|
| Relation Network (original) | 98.2% |
| + Cross-attention | 98.51% |

---

## Files

| File                            | Description |
|---------------------------------|---|
| `main.py`                       | Model definitions, training loop, evaluation |
| `ui.py`                         | Gradio inference UI |
|`requirements.txt`               | requirements file to install libraries|
| `relation_net_best_working.pth` | Saved checkpoint (after training) |