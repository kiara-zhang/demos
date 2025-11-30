"""
Workflow:

(Run in previous script -> Evo2_emb.py)
1.
- Load clinvar_subset.tsv. Exclude rows with GENEINFO containing 'TP53' as TP53 is used as held-out test set.
- Split the remaining dataset into train (80%) / val (20%).
- Load a separate TP53-only test set that has been extracted from a full ClinVar dataset.
2.
- Extract Evo2 embeddings for parsed REF and ALT sequences with a fixed window size (128nt).
- Average embeddings across tokens.
- Concatenate feature vectors for REF and ALT after mean-pooling.

(Run in the current script)
3.
- Train the classifier (architecture and training parameters from Evo2 paper).
4.
- Evaluate on val set and then on TP53 test set.
- Plot results.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, auc

CLINVAR_TSV = "./demo/clinvar_subset.tsv"
FASTA_PATH = "./fasta/grch38_fasta.fa"
WINDOW_SIZE = 128
LAYER_NAME = "blocks.28.mlp.l3"
EVO2_MODEL_NAME = "evo2_7b"
OUT_DIR = "./evo2_emb"
os.makedirs(OUT_DIR, exist_ok=True)

batch_size_embed = 4

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

ref_embs = np.load(os.path.join(OUT_DIR, "ref_embs.npy"))
var_train_embs = np.load(os.path.join(OUT_DIR, "var_train_embs.npy"))
var_val_embs = np.load(os.path.join(OUT_DIR, "var_val_embs.npy"))
var_tp53_embs = np.load(os.path.join(OUT_DIR, "var_tp53_embs.npy"))
ref_idx_train_global = np.load(os.path.join(OUT_DIR, "ref_idx_train_global.npy"), allow_pickle=True)
ref_idx_val_global = np.load(os.path.join(OUT_DIR, "ref_idx_val_global.npy"), allow_pickle=True)
ref_idx_tp53_global = np.load(os.path.join(OUT_DIR, "ref_idx_tp53_global.npy"), allow_pickle=True)
df_train = pd.read_csv(os.path.join(OUT_DIR, "df_train_rows.tsv"), sep="\t", dtype={"CHROM": str})
df_val   = pd.read_csv(os.path.join(OUT_DIR, "df_val_rows.tsv"), sep="\t", dtype={"CHROM": str})
df_tp53  = pd.read_csv(os.path.join(OUT_DIR, "df_tp53_rows.tsv"), sep="\t", dtype={"CHROM": str})

print("Data loaded!!")
print("ref_embs:", ref_embs.shape)
print("var_train_embs:", var_train_embs.shape)
print("var_val_embs:", var_val_embs.shape)
print("var_tp53_embs:", var_tp53_embs.shape)
print("df_train:", df_train.shape)

# Build feature arrays
def build_features_for_split(df_split, ref_idx_global, var_embs_split):
    X = []
    y = []
    var_pos = 0
    for ridx, row in enumerate(df_split.itertuples(index=False)):
        ridx_global = ref_idx_global[ridx]
        if ridx_global is None:
            continue
        v_emb = var_embs_split[var_pos]
        var_pos += 1
        r_emb = ref_embs[ridx_global]
        # Concat REF and VAR embeddings!
        feature = np.concatenate([r_emb, v_emb], axis=0)
        X.append(feature)
        y.append(row.LABEL_BIN)
    X = np.vstack(X)
    y = np.array(y, dtype=np.int64)
    return X, y
X_train, y_train = build_features_for_split(df_train, ref_idx_train_global, var_train_embs)
X_val, y_val = build_features_for_split(df_val, ref_idx_val_global, var_val_embs)
X_tp53, y_tp53 = build_features_for_split(df_tp53, ref_idx_tp53_global, var_tp53_embs)
print("Feature shapes - train/val/tp53:", X_train.shape, X_val.shape, X_tp53.shape)


# -----------------------------------------------------
# 3. Train the classifier according to the Evo2 paper on BCRA1 classification
D = X_train.shape[1]
print("Classifier input dim:", D)
class Classifier(nn.Module):
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
        x = self.fc1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.relu(x)
        x = self.bn3(x)

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
val_loader = DataLoader(NumpyDataset(X_val, y_val), batch_size=batch_size_train, shuffle=False)

model_clf = Classifier(D).to(device)
optimizer = torch.optim.Adam(model_clf.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss()

best_val_auc = -np.inf
patience_counter = 0
history = {"train_loss": [], "val_loss": [], "val_auc": []}

for epoch in range(1, MAX_EPOCHS+1):
    model_clf.train()
    total_loss = 0.0
    for xb, yb in train_loader:
        xb = xb.to(device); yb = yb.to(device)
        optimizer.zero_grad()
        logits = model_clf(xb)
        loss = criterion(logits, yb)
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model_clf.parameters(), GRAD_CLIP)
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    train_loss = total_loss / len(train_loader.dataset)

    # Validation
    model_clf.eval()
    val_loss = 0.0
    preds = []
    truths = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device); yb = yb.to(device)
            logits = model_clf(xb)
            loss = criterion(logits, yb)
            val_loss += loss.item() * xb.size(0)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            preds.extend(probs.tolist())
            truths.extend(yb.detach().cpu().numpy().tolist())
    val_loss = val_loss / len(val_loader.dataset)
    val_auc = roc_auc_score(np.array(truths), np.array(preds))

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_auc"].append(val_auc)

    print(f"Epoch {epoch:03d} TrainLoss={train_loss:.4f} ValAUROC={val_auc:.4f}")

    # Early stopping
    if val_auc > best_val_auc + 1e-6:
        best_val_auc = val_auc
        patience_counter = 0
        torch.save(model_clf.state_dict(), os.path.join(OUT_DIR, "best_clf.pt"))
    else:
        patience_counter += 1
    if patience_counter >= PATIENCE:
        # Reduce lr
        for g in optimizer.param_groups:
            old = g["lr"]
            new = max(old * LR_FACTOR, MIN_LR)
            g["lr"] = new
        print(f"Patience reached: reducing lr. New lr: {optimizer.param_groups[0]['lr']:.1e}")
        patience_counter = 0
        if optimizer.param_groups[0]["lr"] <= MIN_LR + 1e-12:
            break


# 4. Evaluate final model on val set and TP53 test set
model_clf.load_state_dict(torch.load(os.path.join(OUT_DIR, "best_clf.pt"), map_location=device))
model_clf.eval()

def eval_model(X, y, batch_size=128):
    ds = DataLoader(NumpyDataset(X, y), batch_size=batch_size, shuffle=False)
    preds = []
    truths = []
    with torch.no_grad():
        for xb, yb in ds:
            xb = xb.to(device)
            logits = model_clf(xb)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            preds.extend(probs.tolist())
            truths.extend(yb.numpy().tolist())
    preds = np.array(preds)
    truths = np.array(truths)
    roc = roc_auc_score(truths, preds) if len(np.unique(truths))>1 else float("nan")
    precision, recall, _ = precision_recall_curve(truths, preds)
    pr_auc = auc(recall, precision)
    fpr, tpr, _ = roc_curve(truths, preds) if len(np.unique(truths))>1 else (None,None,None)
    return {"roc_auc": roc, "pr_auc": pr_auc, "preds": preds, "truths": truths, "precision": precision, "recall": recall, "fpr": fpr, "tpr": tpr}

print("\nEvaluating on validation set...")
val_stats = eval_model(X_val, y_val)
print("Val AUROC:", val_stats["roc_auc"], "Val AUPRC:", val_stats["pr_auc"])

print("\nEvaluating on TP53 test set...")
tp53_stats = eval_model(X_tp53, y_tp53)
print("TP53 AUROC:", tp53_stats["roc_auc"], "TP53 AUPRC:", tp53_stats["pr_auc"])

# ROC plot for TP53 and val
plt.figure(figsize=(6,5))
if val_stats["fpr"] is not None:
    plt.plot(val_stats["fpr"], val_stats["tpr"], label=f"Val AUROC={val_stats['roc_auc']:.3f}")
if X_tp53.shape[0] > 0 and tp53_stats["fpr"] is not None:
    plt.plot(tp53_stats["fpr"], tp53_stats["tpr"], label=f"TP53 AUROC={tp53_stats['roc_auc']:.3f}")
plt.plot([0,1],[0,1],'k--', alpha=0.3)
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(); plt.title("ROC curves")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "roc_curves.png"), dpi=150)
plt.close()

# PR curves
plt.figure(figsize=(6,5))
plt.plot(val_stats["recall"], val_stats["precision"], label=f"Val AUPRC={val_stats['pr_auc']:.3f}")
if X_tp53.shape[0] > 0:
    plt.plot(tp53_stats["recall"], tp53_stats["precision"], label=f"TP53 AUPRC={tp53_stats['pr_auc']:.3f}")
pos_rate_val = y_val.mean() if len(y_val)>0 else 0
plt.hlines(pos_rate_val, xmin=0, xmax=1, colors='k', linestyles='--', alpha=0.3, label=f"no-skill={pos_rate_val:.3f}")
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend(); plt.title("Precision-recall")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "pr_curves.png"), dpi=150)
plt.close()

# Save predictions on TP53 test set
if X_tp53.shape[0] > 0:
    df_tp53_out = df_tp53.copy().reset_index(drop=True)
    df_tp53_out = df_tp53_out.loc[[i for i in range(len(df_tp53_out)) if ref_idx_tp53_global[i] is not None]].reset_index(drop=True)
    df_tp53_out["pred_prob"] = tp53_stats["preds"]
    df_tp53_out.to_csv(os.path.join(OUT_DIR, "tp53_predictions.tsv"), sep="\t", index=False)
    print("Saved TP53 predictions to:", os.path.join(OUT_DIR, "tp53_predictions.tsv"))

# Save training history
pd.DataFrame(history).to_csv(os.path.join(OUT_DIR, "training_history.csv"), index=False)
print("Completed!! Outputs in:", OUT_DIR)
