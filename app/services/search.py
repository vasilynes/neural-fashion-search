from qdrant_client.models import Fusion, FusionQuery, QueryRequest, Prefetch
from app.services.model import ModelService
from app.config import config

class SearchService:
    def __init__(self, client, model_service: ModelService):
        self.client = client
        self.model_service = model_service

    def search_by_text(self, query: str, limit=10):
        return self.search_by_text_batch([query], limit=limit)[0].points

    def search_by_text_batch(self, queries: list[str], limit=10):
        dense_batch = self.model_service.embed_text(queries)
        sparse_batch = self.model_service.embed_text_sparse(queries)
        return self.search_by_embeddings(dense_batch, sparse_batch, limit)
    
    def search_by_embeddings(self, dense_batch: list, sparse_batch: list, limit=10):
        requests = [
            QueryRequest(
                prefetch=[
                    Prefetch(query=dense, using='text', limit=limit * 2),
                    Prefetch(query=sparse, using='text_sparse', limit=limit * 2),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=limit,
                with_payload=True
            )
            for dense, sparse in zip(dense_batch, sparse_batch)
        ]

        return self.client.query_batch_points(
            collection_name=config.DB_NAME,
            requests=requests,
        )
    
    def search_by_image(self, image, limit=10):
        dense = self.model_service.embed_image(image)

        return self.client.query_points(
            collection_name=config.DB_NAME,
            query=dense,
            using='image',
            limit=limit
        ).points
    
    def search_by_embeddings_alpha(self, dense_batch: list, sparse_batch: list, alpha=0.5, limit=10):
        dense_results = self.client.query_batch_points(
            collection_name=config.DB_NAME,
            requests=[
                QueryRequest(query=dense, using='text', limit=limit * 2, with_payload=True)
                for dense in dense_batch
            ]
        )
        
        sparse_results = self.client.query_batch_points(
            collection_name=config.DB_NAME,
            requests=[
                QueryRequest(query=sparse, using='text_sparse', limit=limit * 2, with_payload=True)
                for sparse in sparse_batch
            ]
        )
        
        all_results = []
        for dense_res, sparse_res in zip(dense_results, sparse_results):
            d_scores = [p.score for p in dense_res.points]
            s_scores = [p.score for p in sparse_res.points]

            d_min, d_max = (min(d_scores), max(d_scores)) if d_scores else (0, 1)
            s_min, s_max = (min(s_scores), max(s_scores)) if s_scores else (0, 1)
            d_range = (d_max - d_min) or 1.0
            s_range = (s_max - s_min) or 1.0

            combined_scores = {}
            unique_points = {}
            for point in dense_res.points:
                norm_score = (point.score - d_min) / d_range
                combined_scores[point.id] = alpha * norm_score
                unique_points[point.id] = point
            
            for point in sparse_res.points:
                norm_score = (point.score - s_min) / s_range
                if point.id in combined_scores:
                    combined_scores[point.id] += (1 - alpha) * norm_score
                else:
                    combined_scores[point.id] = (1 - alpha) * norm_score
                unique_points[point.id] = point
            
            sorted_points = sorted(
                unique_points.values(),
                key=lambda p: combined_scores.get(p.id, 0),
                reverse=True
            )[:limit]
            
            all_results.append(sorted_points)
        
        return all_results

