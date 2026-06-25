import re
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

# =================================================
# Config
# =================================================
IN_DIR = Path("../../../data/labram_classification_1s")
FILE_GLOB = "*.npz"

IN_DIM = 200            # 200 LaBraM features (γίνε 216 αν προσθέσεις τα 16 handcrafted)
HIDDEN = (256, 128)
OUT_DIM = 64
DROPOUT = 0.3

LOSS = "supcon"         # "supcon" ή "triplet"
TEMPERATURE = 0.1       # για supcon
MARGIN = 0.3            # για triplet

EPOCHS = 80
BATCH_SIZE = 128        # μισά-μισά pos/neg
LR = 1e-3
WEIGHT_DECAY = 1e-4

N_SHOT_RANGE = [1, 5, 10, 20]
N_SUPPORT_DRAWS = 20    # πόσα random support draws ανά (patient, n_shot) -> mean ± std
EVAL_THRESHOLD = 0.5

FEATURE_KEYS = ("embeddings", "X", "features", "emb")
LABEL_KEYS = ("labels", "y", "label")

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CSV_OUTPUT = Path("results_mlp_encoder_fewshot.csv")


# =================================================
# Data loading
# =================================================
def patient_from_filename(name: str):
    m = re.match(r"^(P\d+)_", name, flags=re.IGNORECASE)
    return m.group(1).upper() if m else None


def _first_present(candidates, available):
    for k in candidates:
        if k in available:
            return k
    return None


def load_npz(path: Path):
    z = np.load(path, allow_pickle=True)
    keys = set(z.files)
    xk = _first_present(FEATURE_KEYS, keys)
    yk = _first_present(LABEL_KEYS, keys)
    if xk is None or yk is None:
        raise KeyError(f"{path.name}: no feature/label keys. Found {sorted(keys)}")
    X = np.asarray(z[xk]).astype(np.float32)
    y = np.asarray(z[yk]).astype(np.int64).ravel()
    return X, y


def build_patient_data(in_dir: Path):
    files = sorted(in_dir.glob(FILE_GLOB))
    if not files:
        raise RuntimeError(f"No {FILE_GLOB} in {in_dir}")

    per_patient = {}
    for f in files:
        pid = patient_from_filename(f.name)
        if pid is None:
            print(f"Skipping (no patient id): {f.name}")
            continue
        X, y = load_npz(f)
        per_patient.setdefault(pid, []).append((X, y))

    data = {}
    for pid, chunks in per_patient.items():
        X = np.concatenate([c[0] for c in chunks], axis=0)
        y = np.concatenate([c[1] for c in chunks], axis=0)
        data[pid] = (X, y)
    return data


# =================================================
# Encoder
# =================================================
class MLPEncoder(nn.Module):
    def __init__(self, in_dim=IN_DIM, hidden=HIDDEN, out_dim=OUT_DIM, dropout=DROPOUT):
        super().__init__()
        dims = [in_dim, *hidden]
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [
                nn.Linear(a, b),
                nn.BatchNorm1d(b),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(dims[-1], out_dim)

    def forward(self, x):
        z = self.head(self.backbone(x))
        return F.normalize(z, dim=1)   # L2-normalized embeddings


# =================================================
# Losses (το "δημιουργεί τα ζευγάρια" γίνεται μέσα στο batch, μέσω του label mask)
# =================================================
def supcon_loss(emb, labels, temperature=TEMPERATURE):
    """Supervised contrastive loss. Κάθε anchor έλκει όλα τα same-class
    δείγματα του batch και απωθεί τα υπόλοιπα (implicit pairs)."""
    device = emb.device
    b = emb.shape[0]
    sim = (emb @ emb.T) / temperature
    sim = sim - sim.max(dim=1, keepdim=True)[0].detach()   # stability

    labels = labels.view(-1, 1)
    pos_mask = (labels == labels.T).float()
    self_mask = torch.eye(b, device=device)
    pos_mask = pos_mask - self_mask                         # βγάλε το self

    exp_sim = torch.exp(sim) * (1 - self_mask)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

    pos_count = pos_mask.sum(dim=1)
    mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / pos_count.clamp(min=1.0)

    has_pos = pos_count > 0
    loss = -(mean_log_prob_pos * has_pos).sum() / has_pos.sum().clamp(min=1.0)
    return loss


def batch_hard_triplet_loss(emb, labels, margin=MARGIN):
    """Batch-hard triplet: για κάθε anchor, hardest positive vs hardest negative."""
    device = emb.device
    dist = torch.cdist(emb, emb)                            # (B,B) euclidean
    labels = labels.view(-1, 1)
    pos_mask = (labels == labels.T)
    eye = torch.eye(len(emb), dtype=torch.bool, device=device)
    pos_mask_noself = pos_mask & ~eye
    neg_mask = ~pos_mask

    hardest_pos = dist.masked_fill(~pos_mask_noself, float("-inf")).max(dim=1)[0]
    hardest_neg = dist.masked_fill(~neg_mask, float("inf")).min(dim=1)[0]

    valid = torch.isfinite(hardest_pos) & torch.isfinite(hardest_neg)
    loss = torch.clamp(hardest_pos - hardest_neg + margin, min=0.0)[valid]
    if loss.numel() == 0:
        return torch.zeros((), device=device, requires_grad=True)
    return loss.mean()


def compute_loss(emb, labels):
    if LOSS == "supcon":
        return supcon_loss(emb, labels)
    elif LOSS == "triplet":
        return batch_hard_triplet_loss(emb, labels)
    raise ValueError(f"Unknown LOSS: {LOSS}")


# =================================================
# Balanced batches (αυτό "φτιάχνει τα ζευγάρια": εξασφαλίζει pos & neg σε κάθε batch)
# =================================================
def balanced_batches(y, batch_size, n_batches, rng):
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    half = batch_size // 2
    for _ in range(n_batches):
        p = rng.choice(pos, size=half, replace=len(pos) < half)
        n = rng.choice(neg, size=half, replace=len(neg) < half)
        idx = np.concatenate([p, n])
        rng.shuffle(idx)
        yield idx


# =================================================
# Train / encode
# =================================================
def train_encoder(X_train, y_train, seed=SEED):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    scaler = StandardScaler().fit(X_train)
    Xs = scaler.transform(X_train).astype(np.float32)
    Xs_t = torch.from_numpy(Xs).to(DEVICE)
    y_t = torch.from_numpy(y_train.astype(np.int64)).to(DEVICE)

    enc = MLPEncoder().to(DEVICE)
    opt = torch.optim.AdamW(enc.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    n_batches = max(1, len(y_train) // BATCH_SIZE)
    enc.train()
    for epoch in range(EPOCHS):
        ep_loss = 0.0
        for idx in balanced_batches(y_train, BATCH_SIZE, n_batches, rng):
            idx_t = torch.from_numpy(idx).to(DEVICE)
            emb = enc(Xs_t[idx_t])
            loss = compute_loss(emb, y_t[idx_t])
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        if (epoch + 1) % 20 == 0:
            print(f"    epoch {epoch+1:3d}/{EPOCHS}  loss={ep_loss/n_batches:.4f}")

    return enc, scaler


@torch.no_grad()
def encode(enc, scaler, X):
    enc.eval()
    Xs = scaler.transform(X).astype(np.float32)
    z = enc(torch.from_numpy(Xs).to(DEVICE))
    return z.cpu().numpy()


# =================================================
# Prototypical few-shot eval (στον χώρο του encoder)
# =================================================
def prototypical_predict(Z_support, y_support, Z_query):
    proto_pos = Z_support[y_support == 1].mean(axis=0)
    proto_neg = Z_support[y_support == 0].mean(axis=0)
    d_pos = np.linalg.norm(Z_query - proto_pos, axis=1)
    d_neg = np.linalg.norm(Z_query - proto_neg, axis=1)
    m = np.maximum(-d_pos, -d_neg)
    e_pos = np.exp(-d_pos - m)
    e_neg = np.exp(-d_neg - m)
    return e_pos / (e_pos + e_neg)


def sample_support(y, n_shot, rng):
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    if len(pos) < n_shot or len(neg) < n_shot:
        return None
    sup = np.concatenate([
        rng.choice(pos, size=n_shot, replace=False),
        rng.choice(neg, size=n_shot, replace=False),
    ])
    qry = np.setdiff1d(np.arange(len(y)), sup)
    return sup, qry


def safe_auc(fn, y, p):
    return float(fn(y, p)) if len(np.unique(y)) == 2 else float("nan")


def evaluate(y_true, y_proba, thr=EVAL_THRESHOLD):
    y_pred = (y_proba >= thr).astype(np.int64)
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "bacc": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "roc_auc": safe_auc(roc_auc_score, y_true, y_proba),
        "pr_auc": safe_auc(average_precision_score, y_true, y_proba),
    }


# =================================================
# LOPO
# =================================================
def main():
    data = build_patient_data(IN_DIR)
    patients = sorted(data.keys())
    print(f"Patients: {patients} (n={len(patients)})")
    print(f"Loss={LOSS}  in_dim={IN_DIM}  out_dim={OUT_DIM}  device={DEVICE}\n")

    metric_keys = ["acc", "bacc", "f1", "precision", "recall", "roc_auc", "pr_auc"]
    # per_shot[n_shot][metric] = list of per-patient means
    per_shot = {ns: {k: [] for k in metric_keys} for ns in N_SHOT_RANGE}

    for test_pid in patients:
        print(f"[fold] held-out = {test_pid}")
        train_pids = [p for p in patients if p != test_pid]
        X_train = np.concatenate([data[p][0] for p in train_pids], axis=0)
        y_train = np.concatenate([data[p][1] for p in train_pids], axis=0)

        enc, scaler = train_encoder(X_train, y_train)

        X_test, y_test = data[test_pid]
        Z_test = encode(enc, scaler, X_test)

        rng = np.random.default_rng(SEED)
        for ns in N_SHOT_RANGE:
            draws = []
            for _ in range(N_SUPPORT_DRAWS):
                s = sample_support(y_test, ns, rng)
                if s is None:
                    break
                sup, qry = s
                if len(np.unique(y_test[qry])) < 2:
                    continue
                p = prototypical_predict(Z_test[sup], y_test[sup], Z_test[qry])
                draws.append(evaluate(y_test[qry], p))
            if draws:
                for k in metric_keys:
                    per_shot[ns][k].append(float(np.nanmean([d[k] for d in draws])))

    # ---- summary ----
    print("\n=== Few-shot LOPO summary (mean ± std across patients) ===")
    rows = []
    for ns in N_SHOT_RANGE:
        row = {"n_shot": ns, "n_patients": len(per_shot[ns]["acc"])}
        line = [f"n_shot={ns:2d}", f"n_pat={row['n_patients']:2d}"]
        for k in metric_keys:
            vals = per_shot[ns][k]
            mean = float(np.nanmean(vals)) if vals else float("nan")
            std = float(np.nanstd(vals)) if vals else float("nan")
            row[k] = mean
            row[f"{k}_std"] = std
            if k in ("acc", "bacc", "f1", "roc_auc"):
                line.append(f"{k}={mean:.3f}±{std:.3f}")
        rows.append(row)
        print("  " + " | ".join(line))

    with open(CSV_OUTPUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {CSV_OUTPUT}")


if __name__ == "__main__":
    main()