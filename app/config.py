from pathlib import Path
from dataclasses import dataclass
from ml.src.config import config as ml_config

@dataclass
class Config:
    DB_NAME: str = 'fashion_database'

    DB_HOST: str = 'localhost'  
    DB_PORT: int = 6333

    DENSE_MODEL_NAME: str = 'patrickjohncyh/fashion-clip'
    SPARSE_MODEL_NAME: str = 'prithivida/Splade_PP_en_v1'

    PEFT_CHECKPOINT: Path = ml_config.CHECKPOINT_DIR / 'lora8_best'

    MANIFEST_FILE: Path = ml_config.MANIFEST_FILE

    DATA_DIR: Path = ml_config.DATA_DIR

config = Config()
