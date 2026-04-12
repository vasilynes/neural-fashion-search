from fastapi import FastAPI, File, UploadFile, Request, HTTPException, Query
from fastapi.responses import FileResponse
from PIL import Image
import io
from contextlib import asynccontextmanager
import torch
from qdrant_client import QdrantClient
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
from fastembed import SparseTextEmbedding
from peft import PeftModel
import logging
from ml.src.data import preprocess_image
from app.services.search import SearchService
import pandas as pd
from pathlib import Path
from pydantic import BaseModel
from app.config import config
from app.services.model import ModelService
import os
from prometheus_fastapi_instrumentator import Instrumentator

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)

def create_model_service():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = AutoProcessor.from_pretrained(config.DENSE_MODEL_NAME, use_fast=False)
    model = AutoModelForZeroShotImageClassification.from_pretrained(config.DENSE_MODEL_NAME)
    model = PeftModel.from_pretrained(model, str(config.PEFT_CHECKPOINT))
    model.to(device)
    model.eval()

    logger.info(f"PEFT-model loaded from {config.PEFT_CHECKPOINT}")
    logger.info(f"On device: {device}")
    
    sparse_model = SparseTextEmbedding(
        model_name=config.SPARSE_MODEL_NAME, 
        threads=os.cpu_count()
    )

    model_service = ModelService(
        processor,
        device,
        model,
        sparse_model,
    )

    return model_service

def create_search_service(snapshot_path=None, model_service=None):
    client = QdrantClient(host=config.DB_HOST, port=config.DB_PORT, timeout=300)
    if snapshot_path:
        existing = [c.name for c in client.get_collections().collections]
        if config.DB_NAME not in existing:
            client.recover_snapshot(
                collection_name=config.DB_NAME,
                location=snapshot_path
            )
    
    if not model_service:
        model_service = create_model_service()

    search_service = SearchService(
        client,
        model_service,
    )

    return search_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    search_service = create_search_service()
    df = pd.read_parquet(config.MANIFEST_FILE).set_index('article_id')

    app.state.search_service = search_service
    app.state.df = df

    logger.info('App is started')

    yield

    logger.info('Shutting down...')

app = FastAPI(title='Multimodal Search Engine', lifespan=lifespan)

Instrumentator().instrument(app).expose(app, endpoint='/metrics')

@app.post('/search/image')
async def search_by_image(
    request: Request,  
    file: UploadFile = File(...),
    query: str = Query(None), 
    beta: float = Query(0.6),
    limit: int = Query(10)
):
    image = Image.open(io.BytesIO(await file.read())).convert('RGB')
    image = preprocess_image(image)
    results = request.app.state.search_service.search_by_image(image, query, beta, limit)
    return [
        {
            'article_id': r.payload['article_id'],
            'detail_desc': r.payload['detail_desc'],
            'score': r.score,
            'colour_group_name': r.payload['colour_group_name'],
            'product_type_name': r.payload['product_type_name'],
        }
        for r in results
    ]

class TextQuery(BaseModel):
    query: str
    limit: int = 10

@app.post('/search/text')
async def search_by_text(request: Request, body: TextQuery):
    results = request.app.state.search_service.search_by_text(body.query, body.limit)
    return [
        {
            'article_id': r.payload['article_id'],
            'detail_desc': r.payload['detail_desc'],
            'score': r.score,
            'colour_group_name': r.payload['colour_group_name'],
            'product_type_name': r.payload['product_type_name'],
        }
        for r in results
    ]

@app.get('/images/{article_id}')
async def get_image(article_id: str, request: Request):
    df = request.app.state.df

    if article_id not in df.index:
        raise HTTPException(status_code=404, detail=f"Article {article_id} not found")

    row = df.loc[article_id]
    image_path = Path(row['image_path'])
    image_path = Path(str(image_path).replace('\\', '/'))
    parts = image_path.parts
    abs_path = config.DATA_DIR / Path(*parts[1:])

    if not abs_path.exists():
        raise HTTPException(status_code=404, detail=f"Image file not found")
    
    return FileResponse(abs_path, media_type='image_jpeg')

@app.get("/health")
async def health():
    return {"status": "ok", "qdrant": "connected"}