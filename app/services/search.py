from qdrant_client.models import Fusion, FusionQuery, QueryRequest, Prefetch
from app.services.model import ModelService
from app.config import config

class SearchService:
    def __init__(self, client, model_service: ModelService):
        self.client = client
        self.model_service = model_service

    def search_by_text(self, query: str, limit=10):
        return self.search_by_text_batch([query], limit=limit)[0]

    def search_by_image(self, image, limit=10):
        dense = self.model_service.embed_image(image)

        return self.client.query_points(
            collection_name=config.DB_NAME,
            query=dense,
            using='image',
            limit=limit
        )
    
    def search_by_text_batch(self, queries: list[str], limit=10):
        dense_batch = self.model_service.embed_text(queries)
        sparse_batch = self.model_service.embed_text_sparse(queries)
        return self._search_by_embeddings(dense_batch, sparse_batch, limit)

    def search_by_embeddings(self, dense: list, sparse: list, limit=10):
        return self._search_by_embeddings([dense], [sparse], limit)[0]

    def _search_by_embeddings(self, dense_batch: list, sparse_batch: list, limit=10):
        requests = [
            QueryRequest(
                prefetch=[
                    Prefetch(query=dense, using='text', limit=limit * 2),
                    Prefetch(query=sparse, using='text_sparse', limit=limit * 2),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=limit
            )
            for dense, sparse in zip(dense_batch, sparse_batch)
        ]

        return self.client.query_batch_points(
            collection_name=config.DB_NAME,
            requests=requests,
        )
    

