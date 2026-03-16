from pathlib import Path
from dataclasses import dataclass, field
import os

USERNAME = ''
DATASET = ''

@dataclass
class Config:
    IS_KAGGLE: bool  = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ

    @property
    def BASE_DIR(self) -> Path:
        if self.IS_KAGGLE:
            return Path('/kaggle/working')
        else:
            return Path(os.getenv('BASE_DIR', str(Path(__file__).parent.parent)))  
        
    @property
    def RESULTS_DIR(self) -> Path:
        return self.BASE_DIR / 'results'
    
    @property
    def METRICS_DIR(self) -> Path:
        return self.RESULTS_DIR / 'metrics'
        
    @property
    def INPUT_DIR(self) -> Path:
        if self.IS_KAGGLE:
            return Path(f"/kaggle/input/datasets/{USERNAME}")
        else: 
            return self.BASE_DIR

    @property
    def DATA_DIR(self) -> Path:
        if self.IS_KAGGLE:
            return self.INPUT_DIR / DATASET
        else:
            return Path(os.getenv('DATA_DIR', './data'))
        
    @property
    def CHECKPOINT_DIR(self):
        return self.BASE_DIR / 'checkpoints'
    
    @property
    def LOG_DIR(self):
        return self.BASE_DIR / 'logs'
    
    @property
    def IMAGES_DIR(self) -> Path:
        return self.DATA_DIR / 'images'
    
    @property
    def MANIFEST_FILE(self) -> Path:
        return self.DATA_DIR / 'articles.parquet'
    
    @property
    def TRAIN_FILE(self) -> Path:
        return self.DATA_DIR / 'articles_train.parquet'
    
    @property
    def TEST_FILE(self) -> Path:
        return self.DATA_DIR / 'articles_test.parquet'
    
    @property
    def VAL_FILE(self) -> Path:
        return self.DATA_DIR / 'articles_val.parquet'

    CLIP_IMAGE_STATS: dict = field(default_factory=lambda: {
        'mean': (0.48145466, 0.4578275, 0.40821073),        # CLIP images stats
        'std': (0.26862954, 0.26130258, 0.27577711)
    })

    TRAIN_RESIZE: tuple = (256, 256)

    CLIP_IMAGE_SIZE: tuple = (224, 224)

    PAD_COLOR: tuple = (255, 255, 255)      # white

config = Config()

