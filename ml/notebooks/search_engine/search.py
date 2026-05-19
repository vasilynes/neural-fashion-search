import numpy as np
import scipy.sparse as sp
from qdrant_client.models import SparseVector
from tqdm.notebook import tqdm
from pathlib import Path

def collect_search_res_alpha(
        search_service,
        df,
        cols,
        batch_size,
        embeds_dir,
        alpha=.5,
):

    embeds_dir = Path(embeds_dir)
    for col in cols:
        dense_embeds = np.load(embeds_dir / f"{col}_dense_embeds.npy")
        sparse_matrix = sp.load_npz(embeds_dir /  f"{col}_sparse_embeds.npz")
        sparse_embeds = []
        for i in range(sparse_matrix.shape[0]):
            row = sparse_matrix.getrow(i)
            sparse_embeds.append(SparseVector(
                indices=row.indices.tolist(),
                values=row.data.tolist()
            ))

        res = []
        for i in tqdm(range(0, len(dense_embeds), batch_size), desc=f"Searching {col}"):
            dense_batch = dense_embeds[i:i+batch_size]
            sparse_batch = sparse_embeds[i:i+batch_size]
            results = search_service.search_by_embeddings_alpha(dense_batch, sparse_batch, alpha=alpha)
            res.extend([
                [p.payload['article_id'] for p in r]
                for r in results
                ])
        df[f"{col}_res"] = res

    return df