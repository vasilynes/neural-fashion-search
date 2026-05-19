import pandas as pd
from tqdm.notebook import tqdm
import numpy as np
import scipy.sparse as sp
import torch
import gc
from pathlib import Path

def dense_embed(df, col, dense_batch_size, model_service):
    queries = df[col].tolist()
    dense_embeds = []
    for batch_num, i in enumerate(tqdm(range(0, len(queries), dense_batch_size), desc=f"Processing {col}")):
        batch = queries[i:i+dense_batch_size]
        batch_embeds = model_service.embed_text(batch)
        dense_embeds.extend(batch_embeds)

        if batch_num % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    dense_embeds = np.stack(dense_embeds)
    return dense_embeds
        
def sparse_embed(df, col, model_service):
    queries = df[col].tolist()
    sparse_embeds = model_service.embed_text_sparse(queries)
    return sparse_embeds

def save_sparse(sparse_embeds, save_path):
    rows = []
    cols = []
    data = []
    for i, sv in enumerate(sparse_embeds):
        rows.extend([i] * len(sv.indices))
        cols.extend(sv.indices)
        data.extend(sv.values)
    sparse_matrix = sp.csr_matrix((data, (rows, cols)), dtype=np.float32)
    sp.save_npz(save_path, sparse_matrix)
    
def generate_embeds_dense(
    model_service,
    df, 
    cols,
    save_dir,
    dense_batch_size,
):
    save_dir = Path(save_dir)

    for col in cols:
        dense_embeds = dense_embed(df, col, dense_batch_size, model_service)
        np.save(save_dir / f"{col}_dense_embeds.npy", dense_embeds)

def generate_embeds_sparse(
    model_service,
    df, 
    cols,
    save_dir,
):
    save_dir = Path(save_dir)

    for col in cols:
        sparse_embeds = sparse_embed(df, col, model_service)
        path = save_dir / f"{col}_sparse_embeds.npz"
        save_sparse(sparse_embeds, path)