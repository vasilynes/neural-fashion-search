import torch 
import torch.nn.functional as F
from qdrant_client.models import SparseVector, Fusion, FusionQuery
from app.services.model import ModelService
from app.config import config

class SearchService:
    def __init__(self, client, model_service: ModelService):
        self.client = client
        self.model_service = model_service

    def search_by_text(self, query, limit=10):
        dense = self.model_service.embed_text(query)
        sparse = self.model_service.embed_text_sparse(query)

        return self.client.query_points(
            collection_name=config.DB_NAME,
            prefetch=[
                {'query': dense, 'using': 'text', 'limit': limit * 2},
                {'query': sparse, 'using': 'text_sparse', 'limit': limit * 2},
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=limit
        )

    def search_by_image(self, image, limit=10):
        dense = self.model_service.embed_image(image)

        return self.client.query_points(
            collection_name=config.DB_NAME,
            query=dense,
            using='image',
            limit=limit
        )