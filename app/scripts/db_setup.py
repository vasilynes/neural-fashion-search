from qdrant_client.models import VectorParams, Distance, SparseVectorParams, PointStruct, SparseVector
import numpy as np
import json
import scipy.sparse as sp
from qdrant_client import QdrantClient
import pandas as pd

client = QdrantClient(host='localhost', port=6333)

if not client.collection_exists('fashion'):
    client.create_collection(
        collection_name='fashion',
        vectors_config={
            'image': VectorParams(size=512, distance=Distance.COSINE),
            'text': VectorParams(size=512, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            'text_sparse': SparseVectorParams()
        }
    )

image_embeds = np.load('ml/embeddings/image_embeddings.npy')
text_embeds = np.load('ml/embeddings/text_embeddings.npy')
sparse_matrix = sp.load_npz('ml/embeddings/sparse_embeddings.npz')

with open('ml/embeddings/article_ids.json') as f:
    article_ids = json.load(f)

df = pd.read_parquet('ml/data/articles.parquet').set_index('article_id')

points = []
for i, article_id in enumerate(article_ids):
    row = df.loc[article_id] 
    sparse_row = sparse_matrix[i]

    points.append(PointStruct(
        id=i,
        vector={
            'image': image_embeds[i].tolist(),
            'text': text_embeds[i].tolist(),
            'text_sparse': SparseVector(
                indices=sparse_row.indices.tolist(),
                values=sparse_row.data.tolist()
            )
        },
        payload={
            'article_id': article_id,
            'caption': row['caption'],
            'colour_group_name': row['colour_group_name'],
            'product_type_name': row['product_type_name'],
            'image_path': str(row['image_path'])
        }
        )
    )

batch_size = 256
for start in range(0, len(points), batch_size):
    client.upsert(
        collection_name='fashion',
        points=points[start:start + batch_size]
    )
    print(f"Indexed {min(start + batch_size, len(points))}/{len(points)}")