# Few-Shot Learning on Omniglot

A PyTorch implementation of **Matching Networks** (Vinyals et al., NeurIPS 2016) for few-shot image classification on the Omniglot dataset, with an optional **Prototypical Networks** (Snell et al., NeurIPS 2017) baseline included for comparison.

Both models are benchmarked on the standard **5-way 1-shot** task, targeting the scores reported in Table 1 of [Sung et al. (arXiv 1711.06025)](https://arxiv.org/abs/1711.06025).

---

## Models

### Matching Networks (default)
Classifies query images via cosine-softmax attention over support embeddings (Section 2.1.1 of the paper). This is the simple, non-FCE variant — no bidirectional LSTM context embedding.

### Prototypical Networks
Classifies by nearest class prototype under squared Euclidean distance (Eq. 1–2 of the paper). Snell et al. show Euclidean distance substantially outperforms cosine for ProtoNets.

Both models share the same 4-block CNN backbone:
`Conv(64, 3×3) → BN → ReLU → MaxPool(2)` × 4 → 64-d embedding

---

## Dataset

The notebook uses `torchvision`'s built-in Omniglot dataset with two paper-standard preprocessing steps:

- **Image inversion** — pixels flipped so foreground is white
- **Rotation augmentation** — each character class is replicated at 0°/90°/180°/270°, creating 4× more virtual classes (964 × 4 = **3856** training classes)

---

## Requirements

```bash
pip install torch torchvision wandb numpy Pillow
```

GPU recommended but not required (the code auto-detects CUDA).

---

## Configuration

All hyperparameters are set at the top of the notebook (Section 3):

| Parameter | Default | Description |
|---|---|---|
| `MODEL` | `"matching"` | `"matching"` or `"proto"` |
| `N_WAY` | 5 | Number of classes per episode |
| `K_SHOT` | 1 | Support examples per class |
| `N_QUERY` | 15 | Query examples per class |
| `EPOCHS` | 200 | Training epochs |
| `TRAIN_EPISODES` | 1000 | Episodes per epoch |
| `LR` | 1e-3 | Initial learning rate |
| `LR_STEP_EPISODES` | 2000 | Halve LR every N episodes |
| `HIDDEN_SIZE` | 64 | Conv filters in backbone |


## Training

Run all cells in order. The training loop (Section 9):
- Trains for `EPOCHS × TRAIN_EPISODES` total episodes
- Halves the learning rate every `LR_STEP_EPISODES` episodes
- Evaluates on the test set every `EVAL_FREQ` epochs
- Saves the best checkpoint to `./checkpoints/best_{model}.pt`