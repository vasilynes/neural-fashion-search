from fastapi import FastAPI, File, UploadFile, Request
from PIL import Image
import io
from contextlib import asynccontextmanager
import torch
from qdrant_client import QdrantClient
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
from fastembed import SparseTextEmbedding
from peft import PeftModel
from ml.src import config
import logging
from ml.src.data import preprocess_image
from src.search import SearchHandler

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
    client = QdrantClient(path='qdrant_db')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = AutoProcessor.from_pretrained('patrickjohncyh/fashion-clip', use_fast=False)
    model = AutoModelForZeroShotImageClassification.from_pretrained('patrickjohncyh/fashion-clip')
    model = PeftModel.from_pretrained(model, config.CHECKPOINT_DIR / 'lora8_best')
    model.to(device)
    model.eval()

    sparse_model = SparseTextEmbedding(model_name='prithivida/Splade_PP_en_v1')

    search_handler = SearchHandler(
        client,
        processor,
        device,
        model,
        sparse_model,
    )

    app.state.search_handler = search_handler

    logger.info(f"Model loaded from {config.CHECKPOINT_DIR / 'lora8_best'}")
    logger.info(f"On device: {device}")

    yield

    logger.info('Shutting down...')

app = FastAPI(title='Hybrid Search Engine', lifespan=lifespan)

@app.post('/search/image')
async def search_by_image(request: Request, file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert('RGB')
    image = preprocess_image(image)
    results = request.app.state.search_handler.search_by_image(image)
    return results

@app.post('/search/text')
async def search_by_text(request: Request, query: str):
    results = request.app.state.search_handler.search_by_text(query)
    return results

