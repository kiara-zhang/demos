"""
Workflow:

1.
- Load TP53_var.xlsx (dataset from https://doi.org/10.1016/j.molcel.2018.06.012)
- Split into train (80%) / val (10%) / test (10%) sets.
2.
- Extract Evo2 embeddings for all variant sequences (144nt).
- Average embeddings across tokens to use as feature vectors.
3.
- Train MLP for DMS fitness score prediction (architecture and training parameters adjusted from Evo2 paper).
4.
- Evaluate on different mutation types.
- Plot metrics (Spearman correlation).
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import contextlib
import logging, sys
import matplotlib.pyplot as plt
import umap.umap_ as umap
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
from evo2 import Evo2
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("run.log"), logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

DATA = "./TP53_var.xlsx"
LAYER_NAME = "blocks.28.mlp.l3"
EVO2_MODEL_NAME = "evo2_7b"
OUT_DIR = "./evo2_pr2"
os.makedirs(OUT_DIR, exist_ok=True)

# batch_size_embed = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Training hyperparameters from the paper
batch_size_train = 128
LR = 3e-4
MAX_EPOCHS = 500
PATIENCE = 100
LR_FACTOR = 0.5
MIN_LR = 1e-6
GRAD_CLIP = 1.0


# -----------------------------------------------------
# 1. Load data
# -----------------------------------------------------

df = pd.read_excel(DATA)
needed_cols = ["Var_seq", "RFS_H1299", "Mut_type", "Silent"]
missing = [c for c in needed_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in Excel: {missing}")
df = df.dropna(subset=["Var_seq", "RFS_H1299"]).reset_index(drop=True)
print("Total rows loaded:", len(df))
N = len(df)
REF_RANGES = [
    (1,    1663, 'A'),
    (1664, 3363, 'B'),
    (3364, 6292, 'C'),
    (6293, 9516, 'D'),
]

def _row_to_refkey(i1):
    for lo, hi, key in REF_RANGES:
        if lo <= i1 <= hi:
            return key
    return None
df["row_1based"] = np.arange(1, N+1, dtype=int)
df["RefKey"] = df["row_1based"].apply(_row_to_refkey)
# Map A/B/C/D -> 0/1/2/3 for fast indexing
REF_KEY_TO_IDX = {"A": 0, "B": 1, "C": 2, "D": 3}
df["RefIdx"] = df["RefKey"].map(REF_KEY_TO_IDX).astype("Int64")
assert df["RefKey"].notna().all(), "Some rows didn't get a RefKey—check row ranges."
print(df["RefKey"].value_counts(dropna=False).sort_index())

# Split df into train/val/test
rand_seed = 42
temp, df_test = train_test_split(df, test_size=0.10, random_state=rand_seed, shuffle=True)
df_train, df_val  = train_test_split(temp, test_size=0.1111, random_state=rand_seed, shuffle=True)
print("Split sizes:", len(df_train), len(df_val), len(df_test))

# Extract relevant columns
train_seqs = df_train["Var_seq"].astype(str).tolist()
train_y = df_train["RFS_H1299"].astype(float).values
val_seqs = df_val["Var_seq"].astype(str).tolist()
val_y = df_val["RFS_H1299"].astype(float).values
test_seqs = df_test["Var_seq"].astype(str).tolist()
test_y = df_test["RFS_H1299"].astype(float).values

# Group by mutation type (for test set only)
def _is_true(x):
    if pd.isna(x): return False
    s = str(x).strip().lower()
    return s in {"true"}
def map_test_group(row):
    # Silent TRUE => "Silent"
    if _is_true(row["Silent"]):
        return "Silent"
    # Silent FALSE => group by Mut_type
    m = str(row["Mut_type"]).strip()
    if m == "Sub":
        return "SNV"
    elif m in {"AASub", "Del2AA", "DelAA", "Sub2bp", "HSComb", "HSSupp"}:
        return "Non-SNV"
    elif m in {"Ins", "Del"}:
        return "Frameshift"
    elif m == "STOP":
        return "Nonsense"
    else:
        return None
df_test = df_test.copy()
df_test["Group"] = df_test.apply(map_test_group, axis=1)
df_test = df_test[df_test["Group"].notna()].reset_index(drop=True)
print("Mutation groups in test set:")
print(df_test["Group"].value_counts())
test_seqs = df_test["Var_seq"].astype(str).tolist()
test_y = df_test["RFS_H1299"].astype(float).values
test_group = df_test["Group"].astype(str).tolist()


# -----------------------------------------------------
# 2. Extract Evo2 embeddings
# -----------------------------------------------------

# Load Evo2
# Monkeypatch Transformer Engine FP8 to a no-op
try:
    import transformer_engine.pytorch.fp8 as te_fp8  # type: ignore
    @contextlib.contextmanager
    def _no_op_fp8_autocast(*args, **kwargs):
        # ignore enabled flag and yield a simple context
        yield
    # Replace fp8_autocast with the no-op
    te_fp8.fp8_autocast = _no_op_fp8_autocast
    # Also patch FP8GlobalStateManager methods to be safe no-ops if present
    if hasattr(te_fp8, "FP8GlobalStateManager"):
        te_fp8.FP8GlobalStateManager.fp8_autocast_enter = staticmethod(lambda *a, **k: None)
        te_fp8.FP8GlobalStateManager.fp8_autocast_exit = staticmethod(lambda *a, **k: None)
    print("[info] Patched transformer_engine.pytorch.fp8.fp8_autocast -> no-op")
except Exception as e:
    # If transformer_engine isn't installed, nothing to patch (layers won't call it).
    print("[info] Could not patch transformer_engine.pytorch.fp8 (may not be installed):", e)
print("Loading Evo2 model:", EVO2_MODEL_NAME)
evo2_model = Evo2(EVO2_MODEL_NAME)
evo2_model = evo2_model.to(device) if hasattr(evo2_model, "to") else evo2_model

# Extract averaged embedding vectors
def extract_avg_embeddings(seqs, layer_name):
    """
    seqs: list[str]
    returns: numpy array of shape (len(seqs), D) where D is the embedding dim
    """
    vecs = []
    with torch.no_grad():
        for s in seqs:
            # Tokenize one sequence (no more batching due to differing length...)
            tok = evo2_model.tokenizer.tokenize(str(s))
            input_ids = torch.tensor(tok, dtype=torch.long, device=device).unsqueeze(0)
            # Forward pass!
            _, embeddings = evo2_model(input_ids, return_embeddings=True, layer_names=[layer_name])
            emb = embeddings[layer_name]
            if emb.dim() == 3:       # (1, L, D)
                avg = emb.mean(dim=1).squeeze(0)     # (D,)
            elif emb.dim() == 2:     # (L, D)
                avg = emb.mean(dim=0)                # (D,)
            else:
                raise RuntimeError(f"Unexpected embedding shape: {emb.shape}")
            vecs.append(avg.detach().cpu().float().numpy())
    if not vecs:
        return np.zeros((0, 0), dtype=np.float32)
    return np.stack(vecs, axis=0)  # (N, D)

REF_SEQS = {
    "A": "ACCTACCAGGGCAGCTACGGTTTCCGTCTGGGCTTCTTGCATTCTGGGACAGCCAAGTCTGTGACTTGCACGTACTCCCCTGCCCTCAACAAGATGTTTTGCCAACTGGCCAAGACCTGCCCTGTGCAGCTGTGGGTTGATTCC",
    "B": "ACACCCCCGCCCGGCACCCGCGTCCGCGCCATGGCCATCTACAAGCAGTCACAGCACATGACGGAGGTTGTGAGGCGGTGCCCCCACCATGAGCGCTGCTCAAATAGCGATGGTCTGGCCCCTCCTCAGCATCTTATCCGAGTG",
    "C": "GAAGGAAATTTGCGTGTGGAGTATTTGGATGACAGAAACACTTTTCGGCATAGTGTGGTGGTGCCCTATGAGCCGCCTGAGGTTGGCTCTGACTGTACCACCATCCACTACAACTACATGTGTAACAGTTCCTGCATGGGCGGC",
    "D": "ATGAACCGGAGGCCCATCCTCACCATCATCACACTGGAAGACTCCAGTGGTAATCTACTGGGACGGAACAGCTTTGAGGTGCGTGTTTGTGCCTGTCCTGGGAGAGACCGGCGCACAGAGGAAGAGAATCTCCGCAAGAAA",
}
ref_seq_list = [REF_SEQS[k] for k in ("A","B","C","D")]
ref_embs = extract_avg_embeddings(ref_seq_list, LAYER_NAME)
print("ref_embs:", ref_embs.shape)

# Extract embeddings for variant sequences in each split
print("Extracting embeddings for variant sequences (train)...")
var_train_embs = extract_avg_embeddings(train_seqs, LAYER_NAME)
print("train var embeddings:", var_train_embs.shape)
print(f"First few embedding values:\n{var_train_embs[:2, :10]}")
print("Extracting embeddings for variant sequences (val)...")
var_val_embs = extract_avg_embeddings(val_seqs, LAYER_NAME)
print("val var embeddings:", var_val_embs.shape)
print("Extracting embeddings for variant sequences (test)...")
var_test_embs = extract_avg_embeddings(test_seqs, LAYER_NAME)
print("test var embeddings:", var_test_embs.shape)

def build_features_concat(df_split, var_embs_split, y_col="RFS_H1299"):
    if len(df_split) != len(var_embs_split):
        raise ValueError(f"len(df_split)={len(df_split)} != len(var_embs_split)={len(var_embs_split)}")

    if "RefIdx" not in df_split.columns:
        raise KeyError("df_split must contain 'RefIdx' column produced before splitting.")

    ref_idx = df_split["RefIdx"].to_numpy(dtype=int)
    ref_part = ref_embs[ref_idx, :]                          
    X = np.concatenate([ref_part, var_embs_split], axis=1)
    y = df_split[y_col].to_numpy(dtype=np.float32)
    return X.astype(np.float32), y
X_train, y_train = build_features_concat(df_train, var_train_embs, y_col="RFS_H1299")
X_val,   y_val   = build_features_concat(df_val,   var_val_embs,   y_col="RFS_H1299")
X_test,  y_test  = build_features_concat(df_test,  var_test_embs,  y_col="RFS_H1299")
print("Feature dims:", X_train.shape, X_val.shape, X_test.shape)
print("Expect 8192:", X_train.shape[1])

# -----------------------------------------------------
# 3. Train the MLP according to Evo2 paper on BCRA1 VEP
# -----------------------------------------------------

D = X_train.shape[1]
print("Regressor input dim:", D)

class Regressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.out = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.fc1(x); x = self.relu(x); x = self.bn1(x); x = self.dropout(x)
        x = self.fc2(x); x = self.relu(x); x = self.bn2(x); x = self.dropout(x)
        x = self.fc3(x); x = self.relu(x); x = self.bn3(x)
        x = self.out(x)
        return x.squeeze(1)

class NumpyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(NumpyDataset(X_train, y_train), batch_size=batch_size_train, shuffle=True)
val_loader   = DataLoader(NumpyDataset(X_val, y_val), batch_size=batch_size_train, shuffle=False)

model_reg = Regressor(D).to(device)
optimizer = torch.optim.Adam(model_reg.parameters(), lr=LR)
criterion = nn.MSELoss()

best_val_spear = -np.inf
patience_counter = 0
history = {"train_loss": [], "val_loss": [], "val_spear": []}

for epoch in range(1, MAX_EPOCHS + 1):
    model_reg.train()
    total_loss = 0.0
    for xb, yb in train_loader:
        xb = xb.to(device); yb = yb.to(device)
        optimizer.zero_grad()
        preds = model_reg(xb)                 # regression output
        loss = criterion(preds, yb)           # regression loss
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model_reg.parameters(), GRAD_CLIP)
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    train_loss = total_loss / len(train_loader.dataset)

    model_reg.eval()
    val_loss = 0.0
    v_preds, v_truths = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device); yb = yb.to(device)
            preds = model_reg(xb)
            loss = criterion(preds, yb)
            val_loss += loss.item() * xb.size(0)
            v_preds.append(preds.detach().cpu().numpy().ravel())
            v_truths.append(yb.detach().cpu().numpy().ravel())
    val_loss = val_loss / len(val_loader.dataset)
    v_preds  = np.concatenate(v_preds, axis=0)
    v_truths = np.concatenate(v_truths, axis=0)
    val_spear, _ = spearmanr(v_truths, v_preds)
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_spear"].append(val_spear)
    print(f"Epoch {epoch:03d} TrainLoss={train_loss:.4f} ValLoss={val_loss:.4f} ValSpearman={val_spear:.4f}")

    # Early stopping on Spearman
    improved = (val_spear > best_val_spear + 1e-6) or (np.isnan(best_val_spear) and not np.isnan(val_spear))
    if improved:
        best_val_spear = val_spear
        patience_counter = 0
        torch.save(model_reg.state_dict(), os.path.join(OUT_DIR, "best_reg.pt"))
    else:
        patience_counter += 1
    if patience_counter >= PATIENCE:
        for g in optimizer.param_groups:
            g["lr"] = max(g["lr"] * LR_FACTOR, MIN_LR)
        print(f"Patience reached: reducing lr. New lr: {optimizer.param_groups[0]['lr']:.1e}")
        patience_counter = 0
        if optimizer.param_groups[0]["lr"] <= MIN_LR + 1e-12:
            break
# Reload best model
model_reg.load_state_dict(torch.load(os.path.join(OUT_DIR, "best_reg.pt"), map_location=device))
model_reg.eval()
def eval_model_reg(X, y, batch_size=256):
    ds = DataLoader(NumpyDataset(X, y), batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for xb, _ in ds:
            xb = xb.to(device)
            out = model_reg(xb).detach().cpu().numpy().ravel()
            preds.append(out)
    preds = np.concatenate(preds, axis=0)
    r, p = spearmanr(y, preds)
    mse = float(np.mean((preds - y) ** 2))
    return {"spearman": float(r), "p_value": float(p), "mse": mse, "preds": preds}


# -----------------------------------------------------
# 4. Evaluate on validation and test set (by mutation group)
# -----------------------------------------------------

print("\nEvaluating on validation set...")
val_stats = eval_model_reg(X_val, y_val)
print(f"Val Spearman: {val_stats['spearman']:.4f} (p={val_stats['p_value']:.2e}) | MSE: {val_stats['mse']:.4f}")

print("\nEvaluating on test set...")
test_stats = eval_model_reg(X_test, y_test)
print(f"Test Spearman: {test_stats['spearman']:.4f} (p={test_stats['p_value']:.2e}) | MSE: {test_stats['mse']:.4f}")

# Add predictions back to df_test for grouping of mutation types
df_test_eval = df_test.copy()
df_test_eval["y_pred"] = test_stats["preds"]
# Spearman by group
rows = []
for g, sub in df_test_eval.groupby("Group"):
    r, p = spearmanr(sub["RFS_H1299"].values, sub["y_pred"].values)
    rows.append({"group": g, "n": len(sub), "spearman": float(r), "p_value": float(p)})
group_summary = pd.DataFrame(rows).sort_values("group").reset_index(drop=True)
print("\nTest Spearman by group:")
print(group_summary)
group_summary.to_csv(os.path.join(OUT_DIR, "tp53_test_spearman_by_group.tsv"), sep="\t", index=False)


# -----------------------------------------------------
# 5. Plots: bar graph of Spearman by group + scatter plots per group
# -----------------------------------------------------

plt.figure(figsize=(7, 4))
plt.bar(group_summary["group"], group_summary["spearman"])
plt.ylabel("Spearman correlation")
plt.xlabel("Mutation group")
plt.title("TP53 test (REF + VAR concat)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "tp53_spearman_by_group_bar.png"), dpi=200)

groups_order = group_summary["group"].tolist()
n = len(groups_order)
cols = 2
rows = int(np.ceil(n / cols))
plt.figure(figsize=(6*cols, 4*rows))
for i, g in enumerate(groups_order, start=1):
    sub = df_test_eval[df_test_eval["Group"] == g]
    plt.subplot(rows, cols, i)
    plt.scatter(sub["RFS_H1299"].values, sub["y_pred"].values, s=8, alpha=0.7)
    mn = min(sub["RFS_H1299"].min(), sub["y_pred"].min())
    mx = max(sub["RFS_H1299"].max(), sub["y_pred"].max())
    plt.plot([mn, mx], [mn, mx])
    plt.xlabel("Ground truth (DMS relative fitness score)")
    plt.ylabel("Prediction")
    r, _ = spearmanr(sub["RFS_H1299"].values, sub["y_pred"].values)
    plt.title(f"{g} (n={len(sub)}), ρ={r:.2f}")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "tp53_group_scatters.png"), dpi=200)

# Save history
pd.DataFrame(history).to_csv(os.path.join(OUT_DIR, "history_reg.csv"), index=False)
print("Completed!! Outputs in:", OUT_DIR)

