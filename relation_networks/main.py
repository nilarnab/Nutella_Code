import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from tqdm import tqdm
import wandb

wandb.login(key=os.environ["WANDB_API_KEY"])

class Config:
    # Episode settings
    N_WAY      = 5
    K_SHOT     = 1
    N_QUERY    = 15

    # Training
    N_EPISODES = 300_000
    LR         = 1e-3
    LR_STEP    = 100_000
    GRAD_CLIP  = 0.5

    # Eval
    EVAL_EVERY = 1000
    EVAL_EPS   = 1000
    N_VAL_WAY  = 5
    N_VAL_SHOT = 1

    # Dataset: 'omniglot' or 'miniimagenet'
    # DATASET    = 'miniimagenet'
    DATASET = "omniglot"

    # Misc
    SEED       = 42
    DEVICE     = (
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )
    SAVE_PATH  = 'relation_net_best_working.pth'


cfg = Config()

random.seed(cfg.SEED)
np.random.seed(cfg.SEED)
torch.manual_seed(cfg.SEED)


class FewShotDataset(Dataset):
    """
    Thin wrapper around any (image, label) dataset that pre-builds a
    class → [indices] index for O(1) episode sampling.

    Args:
        base_dataset : any torchvision-style dataset
        class_ids    : optional list of class labels to keep (for splits)
    """
    def __init__(self, base_dataset, class_ids=None):
        self.dataset = base_dataset

        # build class → sample indices mapping
        self.class_to_indices: dict[int, list[int]] = defaultdict(list)
        for i in range(len(base_dataset)):
            _, lbl = base_dataset[i]
            lbl = int(lbl)
            self.class_to_indices[lbl].append(i)

        if class_ids is not None:
            keep = set(class_ids)
            self.class_to_indices = {
                k: v for k, v in self.class_to_indices.items() if k in keep
            }

        self.classes = sorted(self.class_to_indices.keys())

    def __len__(self):
        return sum(len(v) for v in self.class_to_indices.values())

    def __getitem__(self, idx):
        return self.dataset[idx]


class EpisodeSampler:
    """
    Samples N-way K-shot episodes from a FewShotDataset.
    """
    def __init__(self, few_shot_dataset: FewShotDataset, device: str):
        self.ds     = few_shot_dataset
        self.device = device

    def sample(self, n_way: int, k_shot: int, n_query: int):
        """
        Returns:
            support  : (n_way * k_shot, C, H, W)
            s_labels : (n_way * k_shot,)  local class indices 0..n_way-1
            queries  : (n_way * n_query, C, H, W)
            q_labels : (n_way * n_query,) local class indices 0..n_way-1
        """
        chosen = random.sample(self.ds.classes, n_way)

        support_imgs,  support_labels = [], []
        query_imgs,    query_labels   = [], []

        for local_idx, cls in enumerate(chosen):
            pool    = self.ds.class_to_indices[cls]
            sampled = random.sample(pool, k_shot + n_query)

            for idx in sampled[:k_shot]:
                img, _ = self.ds.dataset[idx]
                support_imgs.append(img)
                support_labels.append(local_idx)

            for idx in sampled[k_shot:]:
                img, _ = self.ds.dataset[idx]
                query_imgs.append(img)
                query_labels.append(local_idx)

        # shuffle queries
        perm     = torch.randperm(len(query_imgs))
        support  = torch.stack(support_imgs).to(self.device)
        queries  = torch.stack([query_imgs[i] for i in perm]).to(self.device)
        s_labels = torch.tensor(support_labels, dtype=torch.long).to(self.device)
        q_labels = torch.tensor(
            [query_labels[i] for i in perm], dtype=torch.long
        ).to(self.device)

        return support, s_labels, queries, q_labels



def get_datasets(cfg: Config):
    """
    Returns (train_sampler, val_sampler, in_channels, relation_input_size).
    """
    root = os.path.expanduser('~/data')

    if cfg.DATASET == 'miniimagenet':
        # Expects ~/data/miniimagenet/train/ and ~/data/miniimagenet/validation/
        # each containing per-class sub-folders of JPEG images.
        tf = transforms.Compose([
            transforms.Resize(84),
            transforms.CenterCrop(84),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        mini_root = os.path.join(root, 'miniimagenet')
        train_base = datasets.ImageFolder(
            os.path.join(mini_root, 'train'), transform=tf)
        val_base   = datasets.ImageFolder(
            os.path.join(mini_root, 'validation'), transform=tf)

        train_fs = FewShotDataset(train_base)
        val_fs   = FewShotDataset(val_base)

        in_channels  = 3
        input_size   = 576   # 64 * 3 * 3  (after two max-pools on 84×84)

    elif cfg.DATASET == 'omniglot':
        # torchvision downloads Omniglot automatically.
        tf = transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor(),
        ])
        # background=True  → 964  classes (training alphabets)
        # background=False → 659  classes (evaluation alphabets)
        train_base = datasets.Omniglot(
            root=root, background=True,  transform=tf, download=True)
        val_base   = datasets.Omniglot(
            root=root, background=False, transform=tf, download=True)

        train_fs = FewShotDataset(train_base)
        val_fs   = FewShotDataset(val_base)

        in_channels = 1
        input_size  = 64   # 64 * 1 * 1  (after two max-pools on 28×28)

    else:
        raise ValueError(f"Unknown dataset: {cfg.DATASET!r}")

    print(f"Train classes: {len(train_fs.classes)} | "
          f"Val classes: {len(val_fs.classes)}")

    train_sampler = EpisodeSampler(train_fs, cfg.DEVICE)
    val_sampler   = EpisodeSampler(val_fs,   cfg.DEVICE)

    return train_sampler, val_sampler, in_channels, input_size


def conv_block(in_ch: int, out_ch: int, pool: bool = True) -> nn.Sequential:
    """Conv → BN → ReLU → (MaxPool)"""
    layers = [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class EmbeddingModule(nn.Module):
    """
    4-block CNN. First 2 blocks pool; last 2 preserve spatial dims
    so the RelationModule can apply its own convolutions.
    """
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.block1 = conv_block(in_channels, 64, pool=True)
        self.block2 = conv_block(64, 64, pool=True)
        self.block3 = conv_block(64, 64, pool=False)
        self.block4 = conv_block(64, 64, pool=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block4(self.block3(self.block2(self.block1(x))))

class CrossAttention(nn.Module):
    """
    Query attends to Support features before relation scoring.
    Uses multi-head attention on flattened spatial features.
    """
    def __init__(self, channels: int = 64, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

    def forward(self, support_feat: torch.Tensor, query_feat: torch.Tensor):
        """
        support_feat : (B, 64, H, W)
        query_feat   : (B, 64, H, W)
        returns      : (B, 128, H, W)  — same shape as before so rest of code unchanged
        """
        B, C, H, W = support_feat.shape
        # print("Attention called", B, C, H, W)

        # flatten spatial dims → sequence of tokens
        # (B, C, H, W) → (B, H*W, C)
        s = support_feat.view(B, C, -1).permute(0, 2, 1)  # (B, HW, C)
        q = query_feat.view(B, C, -1).permute(0, 2, 1)    # (B, HW, C)

        # query attends to support
        q_attended, _ = self.attention(
            query=q,   # what we want to enrich
            key=s,     # what we attend over
            value=s
        )
        q_attended = self.norm1(q + q_attended)  # residual + norm

        # support attends to query
        s_attended, _ = self.attention(
            query=s,
            key=q,
            value=q
        )
        s_attended = self.norm2(s + s_attended)  # residual + norm

        # reshape back to spatial
        # (B, HW, C) → (B, C, H, W)
        # q_out = q_attended.permute(0, 2, 1).view(B, C, H, W)
        # s_out = s_attended.permute(0, 2, 1).view(B, C, H, W)
        q_out = q_attended.permute(0, 2, 1).contiguous().view(B, C, H, W)
        s_out = s_attended.permute(0, 2, 1).contiguous().view(B, C, H, W)

        # concat just like before — drop-in replacement
        return torch.cat([s_out, q_out], dim=1)  # (B, 128, H, W)

class RelationModule(nn.Module):
    def __init__(self, input_size: int = 1600, num_heads: int = 4):
        super().__init__()
        # cross attention replaces simple concatenation
        self.cross_attention = CrossAttention(channels=64, num_heads=num_heads)

        # rest stays exactly the same as before
        self.conv1 = conv_block(128, 64, pool=True)
        self.conv2 = conv_block(64,  64, pool=True)
        self.fc1   = nn.Linear(input_size, 8)
        self.fc2   = nn.Linear(8, 1)

    def forward(self, support_feat: torch.Tensor, query_feat: torch.Tensor) -> torch.Tensor:
        # cross attention instead of raw concat
        x = self.cross_attention(support_feat, query_feat)  # (B, 128, H, W)
        x = self.conv2(self.conv1(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))


def compute_relation_scores(
    embed:    EmbeddingModule,
    relate:   RelationModule,
    support:  torch.Tensor,
    s_labels: torch.Tensor,
    queries:  torch.Tensor,
    n_way:    int,
    k_shot:   int,
) -> torch.Tensor:
    s_feats = embed(support)   # (n_way*k_shot, 64, h, w)
    q_feats = embed(queries)   # (n_q,          64, h, w)
    _, C, H, W = s_feats.shape

    # mean-pool K shots → prototypes (same as before)
    prototypes = torch.stack([
        s_feats[s_labels == cls].mean(dim=0) for cls in range(n_way)
    ])                         # (n_way, 64, h, w)

    n_q = q_feats.size(0)
    proto_exp = prototypes.unsqueeze(0).expand(n_q, -1, -1, -1, -1)
    query_exp = q_feats.unsqueeze(1).expand(-1, n_way, -1, -1, -1)

    # reshape to (n_q * n_way, 64, h, w) for batch processing
    proto_flat = proto_exp.contiguous().view(n_q * n_way, C, H, W)
    query_flat = query_exp.contiguous().view(n_q * n_way, C, H, W)

    # pass separately — attention handles the combination now
    scores = relate(proto_flat, query_flat).view(n_q, n_way)
    return scores


def make_targets(q_labels: torch.Tensor, n_way: int) -> torch.Tensor:
    """One-hot float targets for MSE loss."""
    targets = torch.zeros(q_labels.size(0), n_way, device=q_labels.device)
    targets.scatter_(1, q_labels.unsqueeze(1), 1.0)
    return targets


@torch.no_grad()
def evaluate(
    embed:      EmbeddingModule,
    relate:     RelationModule,
    sampler:    EpisodeSampler,
    cfg:        Config,
    n_episodes: int = 600,
) -> tuple[float, float]:
    total_loss = 0.0
    mse_loss = nn.MSELoss()
    embed.eval(); relate.eval()
    correct = total = 0

    for _ in range(n_episodes):
        support, s_labels, queries, q_labels = sampler.sample(
            cfg.N_VAL_WAY, cfg.N_VAL_SHOT, cfg.N_QUERY)

        scores = compute_relation_scores(
            embed, relate, support, s_labels,
            queries, cfg.N_VAL_WAY, cfg.N_VAL_SHOT)

        targets = make_targets(q_labels, cfg.N_VAL_WAY)
        loss = mse_loss(scores, targets)
        total_loss += loss.item()

        correct += (scores.argmax(dim=1) == q_labels).sum().item()
        total   += q_labels.size(0)

    embed.train(); relate.train()
    avg_loss = total_loss / n_episodes
    return 100.0 * correct / total, avg_loss


def train():
    print(f"Device : {cfg.DEVICE}")
    print(f"Dataset: {cfg.DATASET}")
    print(f"Task   : {cfg.N_WAY}-way {cfg.K_SHOT}-shot")

    run_name = f"{cfg.DATASET}-{cfg.N_WAY}-way{cfg.K_SHOT}-shot"
    wandb.init(
        project=f"IntroToCVProject",
        name=run_name,
        config={
            "model": "transformer",
        }
    )

    train_sampler, val_sampler, in_ch, input_size = get_datasets(cfg)

    embed  = EmbeddingModule(in_channels=in_ch).to(cfg.DEVICE)
    relate = RelationModule(input_size=input_size).to(cfg.DEVICE)

    print("\nRunning initial evaluation (before training)...")

    init_acc, init_loss = evaluate(
        embed, relate, val_sampler, cfg, cfg.EVAL_EPS
    )

    print(f"[Init] Val acc: {init_acc:.2f}% | Val loss: {init_loss:.4f}")

    wandb.log({
        "eval/accuracy": init_acc,
        "eval/loss": init_loss
    }, step=0)

    opt_e = optim.Adam(embed.parameters(),  lr=cfg.LR)
    opt_r = optim.Adam(relate.parameters(), lr=cfg.LR)
    sch_e = optim.lr_scheduler.StepLR(opt_e, step_size=cfg.LR_STEP, gamma=0.5)
    sch_r = optim.lr_scheduler.StepLR(opt_r, step_size=cfg.LR_STEP, gamma=0.5)

    mse_loss     = nn.MSELoss()
    best_acc = init_acc
    running_loss = 0.0

    pbar = tqdm(range(1, cfg.N_EPISODES + 1), desc='Training')

    for episode in pbar:
        support, s_labels, queries, q_labels = train_sampler.sample(
            cfg.N_WAY, cfg.K_SHOT, cfg.N_QUERY)

        scores  = compute_relation_scores(
            embed, relate, support, s_labels,
            queries, cfg.N_WAY, cfg.K_SHOT)
        targets = make_targets(q_labels, cfg.N_WAY)
        loss    = mse_loss(scores, targets)
        preds = scores.argmax(dim=1)
        train_acc = (preds == q_labels).float().mean().item() * 100

        opt_e.zero_grad(); opt_r.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(embed.parameters(),  cfg.GRAD_CLIP)
        nn.utils.clip_grad_norm_(relate.parameters(), cfg.GRAD_CLIP)
        opt_e.step(); opt_r.step()
        sch_e.step(); sch_r.step()

        running_loss += loss.item()

        if episode % 100 == 0:
            avg_loss     = running_loss / 100
            running_loss = 0.0
            pbar.set_postfix({'loss': f'{avg_loss:.4f}',
                              'best': f'{best_acc:.2f}%'})

            wandb.log({
                "train/loss": avg_loss,
                "train/accuracy": train_acc
            }, step=episode)

        if episode % cfg.EVAL_EVERY == 0:
            acc, val_loss = evaluate(embed, relate, val_sampler, cfg, cfg.EVAL_EPS)
            print(f"\n[Episode {episode:>7}] Val acc: {acc:.2f}%  "
                  f"(best: {best_acc:.2f}%)")

            wandb.log({
                "eval/accuracy": acc,
                "eval/loss": val_loss
            }, step=episode)

            if acc > best_acc:
                best_acc = acc
                torch.save({
                    'episode': episode,
                    'embed':   embed.state_dict(),
                    'relate':  relate.state_dict(),
                    'val_acc': acc,
                    'config':  cfg.__dict__,
                }, cfg.SAVE_PATH)
                print(f"  ✓ Saved best model → {cfg.SAVE_PATH}")

    print(f"\nTraining done. Best val accuracy: {best_acc:.2f}%")


def load_and_predict(
    checkpoint_path: str,
    support_imgs:    torch.Tensor,   # (n_way*k_shot, C, H, W)
    support_labels:  torch.Tensor,   # (n_way*k_shot,)
    query_img:       torch.Tensor,   # (C, H, W)
    n_way:           int,
    k_shot:          int,
    device:          str = 'cpu',
):
    """
    Load a saved checkpoint and classify one query image.

    Returns:
        predicted class index (int)
        relation scores per class (tensor of length n_way)
    """
    ckpt = torch.load(checkpoint_path, map_location=device)

    in_ch      = support_imgs.shape[1]
    # infer input_size from spatial dim: 84px → 576, 28px → 64
    spatial    = support_imgs.shape[-1]
    input_size = 576 if spatial == 84 else 64

    embed  = EmbeddingModule(in_channels=in_ch).to(device)
    relate = RelationModule(input_size=input_size).to(device)
    embed.load_state_dict(ckpt['embed'])
    relate.load_state_dict(ckpt['relate'])
    embed.eval(); relate.eval()

    query          = query_img.unsqueeze(0).to(device)
    support_imgs   = support_imgs.to(device)
    support_labels = support_labels.to(device)

    with torch.no_grad():
        scores = compute_relation_scores(
            embed, relate, support_imgs, support_labels,
            query, n_way, k_shot).squeeze(0)

    return scores.argmax().item(), scores


if __name__ == '__main__':
    train()