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
from ml.src.config import config
import logging
from ml.src.data import preprocess_image
from app.services.search import SearchService
import pandas as pd
from pathlib import Path
from pydantic import BaseModel

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    client = QdrantClient(host='localhost', port=6333)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = AutoProcessor.from_pretrained('patrickjohncyh/fashion-clip', use_fast=False)
    model = AutoModelForZeroShotImageClassification.from_pretrained('patrickjohncyh/fashion-clip')
    model = PeftModel.from_pretrained(model, str(config.CHECKPOINT_DIR / 'lora8_best'))
    model.to(device)
    model.eval()

    sparse_model = SparseTextEmbedding(model_name='prithivida/Splade_PP_en_v1')

    search_service = SearchService(
        client,
        processor,
        device,
        model,
        sparse_model,
    )

    df = pd.read_parquet(config.MANIFEST_FILE).set_index('article_id')

    app.state.search_service = search_service
    app.state.df = df

    logger.info(f"Model loaded from {config.CHECKPOINT_DIR / 'lora8_best'}")
    logger.info(f"On device: {device}")

    yield

    logger.info('Shutting down...')

app = FastAPI(title='Multimodal Search Engine', lifespan=lifespan)

@app.post('/search/image')
async def search_by_image(
    request: Request,  
    file: UploadFile = File(...),
    limit: int = Query(10)
):
    image = Image.open(io.BytesIO(await file.read())).convert('RGB')
    image = preprocess_image(image)
    results = request.app.state.search_service.search_by_image(image)
    return [
        {
            'article_id': r.payload['article_id'],
            'caption': r.payload['caption'],
            'score': r.score,
            'colour_group_name': r.payload['colour_group_name'],
            'product_type_name': r.payload['product_type_name'],
        }
        for r in results.points
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
            'caption': r.payload['caption'],
            'score': r.score,
            'colour_group_name': r.payload['colour_group_name'],
            'product_type_name': r.payload['product_type_name'],
        }
        for r in results.points
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