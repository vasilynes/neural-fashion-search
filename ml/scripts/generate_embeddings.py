from pathlib import Path 
from src.data import FashionDataset, get_dataloader
from src.config import config
import os 
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
from peft import PeftModel
import torch
import numpy as np
import json
from fastembed import SparseTextEmbedding
import scipy.sparse as sp

# Do not shuffle the dataset for consistency of indices across embeddings 
SHUFFLE = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
processor = AutoProcessor.from_pretrained('patrickjohncyh/fashion-clip', use_fast=False)
model = AutoModelForZeroShotImageClassification.from_pretrained('patrickjohncyh/fashion-clip')
checkpoint_dir = Path('checkpoints/lora8_best')
model = PeftModel.from_pretrained(model, checkpoint_dir)
model.to(device)
model.eval()

# The whole dataset is used for both, dense and sparse embeddings
dataset = FashionDataset(manifest_path=config.MANIFEST_FILE)

# Loader parameters for running the checkpoint on GPU 
loader_params = {
    'batch_size': 256,
    'shuffle': SHUFFLE,
    'num_workers': os.cpu_count(),
    'pin_memory': True
}
dataloader = get_dataloader(dataset, loader_params)

image_embeds = []
text_embeds = []
# Store indices once for dense embeddings, sparse ones are assumed to be ordered in the same way
article_ids = []

with torch.no_grad():
    for batch in dataloader:
        inputs = processor(
            images=batch['image'],
            text=batch['caption'],
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=77
        ).to(device)
        outputs = model(**inputs)

        image_embeds.append(outputs.image_embeds.cpu())
        text_embeds.append(outputs.text_embeds.cpu())
        
        article_ids.extend(batch['article_id'])

image_embeds = torch.cat(image_embeds, dim=0).float().numpy()
text_embeds = torch.cat(text_embeds, dim=0).float().numpy()

embeddings_dir = config.BASE_DIR / 'embeddings'
embeddings_dir.mkdir(parents=True, exist_ok=True)

np.save(embeddings_dir / 'image_embeddings.npy', image_embeds)
np.save(embeddings_dir / 'text_embeddings.npy', text_embeds)

with open(embeddings_dir / 'article_ids.json', 'w') as f:
    json.dump(article_ids, f)

# SPLADE runs on CPU, so loader parameters must be adjusted
loader_params = {
    'batch_size': 32, 
    'shuffle': SHUFFLE,
    'num_workers': os.cpu_count(),
    'pin_memory': False
}
dataloader = get_dataloader(dataset, loader_params)

sparse_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")

# Store sparse embeddings as a sparse matrix with scipy.sparse
rows, cols, vals = [], [], []
n_processed = 0
max_col = 0
total = len(dataset)
n_batches = len(dataloader)

for batch_idx, batch in enumerate(dataloader):
    captions = batch['caption']
    sparse_embeds = list(sparse_model.embed(captions))

    for i, embed in enumerate(sparse_embeds):
        global_idx = n_processed + i
        rows.extend([global_idx] * len(embed.indices))
        cols.extend(embed.indices.tolist())
        vals.extend(embed.values.tolist())
        if len(embed.indices) > 0:
            max_col = max(max_col, max(embed.indices))

    n_processed += len(captions)
    # Tracking since SPLADE is quite slow on CPU
    print(f"Batch {batch_idx+1}/{n_batches} | {n_processed}/{total} samples ({100*n_processed/total:.1f}%)")

sparse_matrix = sp.csr_matrix(
    (vals, (rows, cols)),
    shape=(n_processed, max_col + 1)
)

sp.save_npz(embeddings_dir / 'sparse_embeddings.npz', sparse_matrix)