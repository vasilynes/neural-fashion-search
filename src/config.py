from pathlib import Path
from dataclasses import dataclass, field
import os

@dataclass
class Config:
    DATA_DIR: Path = Path(os.getenv('DATA_DIR', './data'))

    @property
    def PROCESSED_DATA_DIR(self) -> Path:
        return self.DATA_DIR / 'processed'
    
    @property
    def IMAGES_DIR(self) -> Path:
        return self.DATA_DIR / 'images'
    
    @property
    def MANIFEST_FILE(self) -> Path:
        return self.PROCESSED_DATA_DIR / 'articles.parquet'
    
    @property
    def TRAIN_FILE(self) -> Path:
        return self.PROCESSED_DATA_DIR / 'articles_train.parquet'
    
    @property
    def TEST_FILE(self) -> Path:
        return self.PROCESSED_DATA_DIR / 'articles_test.parquet'
    
    @property
    def VAL_FILE(self) -> Path:
        return self.PROCESSED_DATA_DIR / 'articles_val.parquet'

    CLIP_IMAGE_STATS: dict = field(default_factory=lambda: {
        'mean': (0.48145466, 0.4578275, 0.40821073),
        'std': (0.26862954, 0.26130258, 0.27577711)
    })

    TRAIN_RESIZE: tuple = (256, 256)

    CLIP_IMAGE_SIZE: tuple = (224, 224)

    PAD_COLOR: tuple = (255, 255, 255)      # white

config = Config()