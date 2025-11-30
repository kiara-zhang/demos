"""
Workflow:

1.
- Load clinvar_subset.tsv. Exclude rows with GENEINFO containing 'TP53' as TP53 is used as held-out test set.
- Split the remaining dataset into train (80%) / val (20%).
- Load a separate TP53-only test set that has been extracted from a full ClinVar dataset.
2.
- Extract Evo2 embeddings for parsed REF and ALT sequences with a fixed window size (128nt).
- Average embeddings across tokens.
- Concatenate feature vectors for REF and ALT after mean-pooling.

(Followed in separate script -> Evo2_clf.py)
3.
- Train the classifier (architecture and training parameters from Evo2 paper).
4.
- Evaluate on val set and then on TP53 test set.
- Plot results.
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import contextlib
import logging, sys
from pyfaidx import Fasta
from evo2 import Evo2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("run.log"), logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

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


# -----------------------------------------------------
# 1. Load data
df = pd.read_csv(CLINVAR_TSV, sep="\t", dtype={"CHROM": str})
print("Total rows loaded:", len(df))

# Map LABEL column to binary labels (1 = pathogenic, 0 = benign)
def label_to_binary(l):
    s = str(l)
    if "Pathogenic" in s:
        return 1
    if "Benign" in s:
        return 0
    raise ValueError(f"Unrecognised label: {l}")
df["LABEL_BIN"] = df["LABEL"].apply(label_to_binary)

# Separate TP53 rows
tp53 = df["GENEINFO"].astype(str).str.contains("TP53", case=True, na=False)
df_rest = df[~tp53].reset_index(drop=True)
print(f"Remaining rows: {len(df_rest)}")

# Split df_rest into train/val
rand_seed = 42
df_rest = df_rest.sample(frac=1.0, random_state=rand_seed).reset_index(drop=True)
n_val = int(0.20 * len(df_rest))
df_val = df_rest.iloc[:n_val].reset_index(drop=True)
df_train = df_rest.iloc[n_val:].reset_index(drop=True)

# Load TP53 test set
df_tp53 = pd.read_csv("./demo/clinvar_tp53_test.tsv", sep="\t", dtype={"CHROM": str})
df_tp53["LABEL_BIN"] = df_tp53["LABEL"].apply(label_to_binary)

print("Train / Val / TP53-test sizes:", len(df_train), len(df_val), len(df_tp53))


# -----------------------------------------------------
# 2. Prepare sequences and extract Evo2 embeddings
fasta = Fasta(FASTA_PATH)

def _chrom_to_fasta(chrom):
    chrom = str(chrom)
    return chrom if chrom.startswith("chr") else "chr" + chrom
def parse_sequences(chrom, pos, ref, alt):
    fasta_chrom = _chrom_to_fasta(str(chrom))
    pos = int(pos)
    p = pos - 1
    chrom_len = len(fasta[fasta_chrom])
    half = WINDOW_SIZE // 2
    ref_seq_start = max(0, p - half)
    ref_seq_end = min(chrom_len, p + half)
    ref_seq = str(fasta[fasta_chrom][ref_seq_start:ref_seq_end]).upper()
    snv_pos_in_ref = p - ref_seq_start
    if snv_pos_in_ref < 0 or snv_pos_in_ref >= len(ref_seq):
        raise IndexError("SNV index out of bounds in window. Check WINDOW_SIZE and position.")
    var_seq = ref_seq[:snv_pos_in_ref] + alt.upper() + ref_seq[snv_pos_in_ref + 1:]
    # Sanity checks
    assert len(var_seq) == len(ref_seq)
    assert ref_seq[snv_pos_in_ref] == ref.upper(), f"Ref mismatch at {chrom}:{pos}"
    assert var_seq[snv_pos_in_ref] == alt.upper(), f"Alt mismatch at {chrom}:{pos}"
    return ref_seq, var_seq

def build_seq_lists(df_rows):
    ref_seqs = []
    ref_index = {}
    ref_idx_per_row = []
    var_seqs = []
    skip = 0
    for _, row in df_rows.iterrows():
        try:
            rseq, vseq = parse_sequences(row["CHROM"], row["POS"], row["REF"], row["ALT"])
        except Exception as e:
            skip += 1
            ref_idx_per_row.append(None)
            var_seqs.append(None)
            continue
        if rseq not in ref_index:
            ref_index[rseq] = len(ref_seqs)
            ref_seqs.append(rseq)
        ref_idx_per_row.append(ref_index[rseq])
        var_seqs.append(vseq)
    if skip:
        print(f"Skipped {skip} rows in build_seq_lists due to fetch errors.")
    return ref_seqs, np.array(ref_idx_per_row, dtype=object), var_seqs
print("Building sequence lists for train/val/test...")
ref_train, ref_idx_train, var_train = build_seq_lists(df_train)
ref_val, ref_idx_val, var_val = build_seq_lists(df_val)
ref_tp53, ref_idx_tp53, var_tp53 = build_seq_lists(df_tp53)
print("Unique ref windows - train/val/tp53:", len(ref_train), len(ref_val), len(ref_tp53))
all_unique_ref = list({s: None for s in (ref_train + ref_val + ref_tp53)}.keys())
print("Total unique reference windows (union):", len(all_unique_ref))
all_ref_index = {s: i for i, s in enumerate(all_unique_ref)}
def remap_ref_idx(ref_idx_per_row, ref_seqs_local):
    remapped = []
    for idx, v in enumerate(ref_idx_per_row):
        if v is None:
            remapped.append(None)
            continue
        # find ref sequence string:
        rseq = None
        # need to map local index to string: ref_seqs_local[v]
        rseq = ref_seqs_local[v]
        remapped.append(all_ref_index[rseq])
    return np.array(remapped, dtype=object)
ref_idx_train_global = remap_ref_idx(ref_idx_train, ref_train)
ref_idx_val_global = remap_ref_idx(ref_idx_val, ref_val)
ref_idx_tp53_global = remap_ref_idx(ref_idx_tp53, ref_tp53)
def filter_none(seq_list):
    return [s for s in seq_list if s is not None]
var_train_filt = filter_none(var_train)
var_val_filt = filter_none(var_val)
var_tp53_filt = filter_none(var_tp53)

# Load Evo2 and extract embeddings

# Monkeypatch Transformer Engine FP8 to a no-op before importing evo2 / vortex
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

# Extract averaged embedding vectors for a list of sequences
def extract_avg_embeddings(seqs, layer_name, batch_size=4):
    """
    seqs: list[str]
    returns: numpy array shape (len(seqs), D) where D is embedding dimension
    """
    all_vecs = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            batch = seqs[i:i+batch_size]
            tokenized = [torch.tensor(evo2_model.tokenizer.tokenize(s), dtype=torch.int64) for s in batch]
            lengths = [t.shape[0] for t in tokenized]
            maxlen = max(lengths)
            padded = [torch.cat([t, torch.full((maxlen - t.shape[0],), evo2_model.tokenizer.pad_token_id, dtype=torch.int64)]) if t.shape[0] < maxlen else t for t in tokenized]
            input_ids = torch.stack(padded).to(device)
            # forward pass!
            outputs, embeddings = evo2_model(input_ids, return_embeddings=True, layer_names=[layer_name])
            emb = embeddings[layer_name]  # (batch, seq_len, D)
            if emb.dim() == 3:
                avg = emb.mean(dim=1)   # (batch, D)
            else:
                raise RuntimeError(f"Unexpected embedding dim: {emb.shape}")
            avg = avg.detach().cpu().float().numpy()
            all_vecs.append(avg)
    if len(all_vecs) == 0:
        return np.zeros((0,0), dtype=float)
    return np.vstack(all_vecs)
# Extract embeddings for all unique reference windows
print("Extracting embeddings for all unique reference windows (count=%d)..." % len(all_unique_ref))
ref_embs = extract_avg_embeddings(all_unique_ref, LAYER_NAME, batch_size=batch_size_embed)
print("Reference embeddings shape:", ref_embs.shape)
# Extract embeddings for variant sequences for each split
print("Extracting embeddings for variant sequences (train)...")
var_train_embs = extract_avg_embeddings(var_train_filt, LAYER_NAME, batch_size=batch_size_embed)
print("train var embeddings:", var_train_embs.shape)
print("Extracting embeddings for variant sequences (val)...")
var_val_embs = extract_avg_embeddings(var_val_filt, LAYER_NAME, batch_size=batch_size_embed)
print("val var embeddings:", var_val_embs.shape)
print("Extracting embeddings for variant sequences (tp53 test)...")
var_tp53_embs = extract_avg_embeddings(var_tp53_filt, LAYER_NAME, batch_size=batch_size_embed)
print("tp53 var embeddings:", var_tp53_embs.shape)

np.save(os.path.join(OUT_DIR, "ref_embs.npy"), ref_embs)
np.save(os.path.join(OUT_DIR, "var_train_embs.npy"), var_train_embs)
np.save(os.path.join(OUT_DIR, "var_val_embs.npy"), var_val_embs)
np.save(os.path.join(OUT_DIR, "var_tp53_embs.npy"), var_tp53_embs)
np.save(os.path.join(OUT_DIR, "ref_idx_train_global.npy"), ref_idx_train_global, allow_pickle=True)
np.save(os.path.join(OUT_DIR, "ref_idx_val_global.npy"), ref_idx_val_global, allow_pickle=True)
np.save(os.path.join(OUT_DIR, "ref_idx_tp53_global.npy"), ref_idx_tp53_global, allow_pickle=True)
df_train.to_csv(os.path.join(OUT_DIR, "df_train_rows.tsv"), sep="\t", index=False)
df_val.to_csv(os.path.join(OUT_DIR, "df_val_rows.tsv"), sep="\t", index=False)
df_tp53.to_csv(os.path.join(OUT_DIR, "df_tp53_rows.tsv"), sep="\t", index=False)
print("Data saved!!")
log.info(f"Reference embeddings shape: {ref_embs.shape}")
log.info(f"First few embedding values (ref):\n{ref_embs[:2, :10]}")  # prints first 10 dims of 2 samples


