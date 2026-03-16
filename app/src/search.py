import torch 
import torch.nn.functional as F
from qdrant_client.models import SparseVector, Fusion, FusionQuery

class SearchHandler:
    def __init__(self, client, processor, device, model, sparse_model):
        self.client = client
        self.processor = processor
        self.device = device
        self.model = model
        self.sparse_model = sparse_model

    def embed_text(self, text):
        inputs = self.processor(
            text=[text],
            truncation=True,
            padding=True, 
            return_tensors='pt',
            max_length=77
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return F.normalize(outputs.text_embeds, dim=-1).cpu().numpy()[0]

    def embed_image(self, image):
        inputs = self.processor(
            images=[image],
            return_tensors='pt',
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return F.normalize(outputs.image_embeds, dim=-1).cpu().numpy()[0]

    def embed_text_sparse(self, text):
        embeds = list(self.sparse_model.embed([text]))[0]
        return SparseVector(
            indices=embeds.indices.tolist(),
            values=embeds.values.tolist()
        )

    def search_by_text(self, query, limit=10):
        dense = self.embed_text(query)
        sparse = self.embed_text_sparse(query)

        return self.client.query_points(
            collection_name='fashion database',
            prefetch=[
                {'query': dense, 'using': 'text', 'limit': limit * 2},
                {'query': sparse, 'using': 'text_sparse', 'limit': limit * 2},
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=limit
        )

    def search_by_image(self, image, limit=10):
        dense = self.embed_image(image)

        return self.client.query_points(
            collection_name='fashion database',
            query=dense,
            using='image',
            limit=limit
        )