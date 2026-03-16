from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import ImageOps
import random
from torch.utils.data import Sampler
import torch
from src.config import config

class SquarePad:
    def __init__(self, fill=(255, 255, 255)):
        self.fill = fill
    
    def __call__(self, image):
        w, h = image.size 
        max_side = max(w, h)

        p_left = (max_side - w) // 2
        p_top = (max_side - h) // 2
        p_right = max_side - w - p_left
        p_bottom = max_side - h - p_top

        return ImageOps.expand(
            image, 
            (p_left, p_top, p_right, p_bottom), 
            fill=self.fill
        )
    
def preprocess_image(image):
    image.thumbnail((256, 256), Image.BICUBIC)
    image = SquarePad()(image)
    return image

def text_dropout(color, product, desc, caption):
    if not color or not product:
        return caption
    
    product_color = f"{color} {product}"
    color_desc = f"{color}. {desc}"

    text = random.choices(
        [caption, product_color, color_desc],
        weights=[0.5, 0.35, 0.15],
        k=1
    )[0]

    return text

class CollateFn:
    def __init__(self, augment=False):
        self.pad = SquarePad()
        self.augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        ]) if augment else None
        self.augment = augment

    def __call__(self, batch):
        images = [item['image'] for item in batch]
        images = [self.pad(img) for img in images] 
        if self.augment:
            images = [self.augmentation(img) for img in images]
            captions = [
                text_dropout(
                    item['colour_group_name'],
                    item['product_type_name'],
                    item['detail_desc'],
                    item['caption']
                ) for item in batch
            ]
        else:
            captions = [item['caption'] for item in batch]
        return {
            'image': images, 
            'caption': captions,
            'colour_group_name': [item['colour_group_name'] for item in batch],
            'product_type_name': [item['product_type_name'] for item in batch],
            'detail_desc': [item['detail_desc'] for item in batch],
            'image_path': [item['image_path'] for item in batch],
            'article_id': [item['article_id'] for item in batch],
        }

class FashionDataset:
    @classmethod 
    def for_split(cls, split, config):
        if split not in ('train', 'val', 'test'):
            raise ValueError(f"Invalid split: '{split}'")
        manifest_path = getattr(config, f"{split.upper()}_FILE")
        return cls(manifest_path)

    def __init__(self, manifest_path, filter_col=None, filter_val=None):
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_path}")
        
        self._df = pd.read_parquet(self.manifest_path)

        if filter_col and filter_val:
            temp_col = self._df[filter_col].replace('', 'unknown')
            mask = temp_col == filter_val
            self.indices = mask[mask].index.tolist()
        else:
            self.indices = list(range(len(self._df)))
    
    def filter(self, filter_col, filter_val):
        new_ds = FashionDataset.__new__(FashionDataset)
        new_ds.manifest_path = self.manifest_path
        new_ds._df = self._df 
        
        temp_col = self._df[filter_col].replace('', 'unknown')
        mask = temp_col == filter_val
        new_ds.indices = mask[mask].index.tolist()
        return new_ds
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        row = self._df.iloc[self.indices[idx]]  
        image_path = row['image_path']
        parts = Path(image_path.replace('\\', '/')).parts  # ('data', 'images', '069', '0694236008.jpg')
        image_path = config.DATA_DIR.joinpath(*parts[1:])   # skip 'data', join rest
        
        image = Image.open(image_path).convert('RGB')
        
        return {
            'image': image,
            'colour_group_name': row['colour_group_name'],
            'product_type_name': row['product_type_name'],
            'detail_desc': row['detail_desc'],
            'caption': row['caption'],
            'image_path': str(image_path),
            'article_id': row['article_id'],
        }
    
def get_dataloader(dataset, loader_params, augment=False, sampler=None):
    if sampler is not None:
        loader_params = {k: v for k, v in loader_params.items() 
                         if k not in ('batch_size', 'shuffle')}
        return DataLoader(
            dataset, 
            batch_sampler=sampler,
            collate_fn=CollateFn(augment),
            **loader_params
        )
    
    return DataLoader(
        dataset, 
        collate_fn=CollateFn(augment),
        **loader_params
    )

class HardNegativesBatchSampler(Sampler):
    def __init__(self, dataset_size, batch_size, hard_negatives=None, hard_negatives_per_anchor=1):
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.hard_negatives = hard_negatives or {}
        self.hard_negatives_per_anchor = hard_negatives_per_anchor

        self.anchors_per_batch = batch_size // (1 + hard_negatives_per_anchor)
        if self.anchors_per_batch == 0:
          raise ValueError(f"batch_size ({batch_size}) is too small for hard_negatives_per_anchor ({hard_negatives_per_anchor}). Need batch_size >= {1 + hard_negatives_per_anchor}")

    def update_hard_negatives(self, hard_negatives):
        self.hard_negatives = hard_negatives

    def __iter__(self):
        indices = torch.randperm(self.dataset_size).tolist()

        for start in range(0, len(indices), self.anchors_per_batch):
            anchors = indices[start:start+self.anchors_per_batch]
            batch = list(anchors)

            for anchor_idx in anchors:
                if anchor_idx in self.hard_negatives and self.hard_negatives[anchor_idx]:
                    neg = random.sample(self.hard_negatives[anchor_idx], 
                                        min(self.hard_negatives_per_anchor,
                                            len(self.hard_negatives[anchor_idx])))
                    batch.extend(neg)
                else:
                    fallback_pool = list(set(indices) - set(batch))
                    batch.extend(random.sample(fallback_pool, 
                                            min(self.hard_negatives_per_anchor, 
                                                len(fallback_pool))))
            yield batch
    
    def __len__(self):
        return self.dataset_size // self.anchors_per_batch

def stratified_split(
        manifest_path: str,
        output_dir: str = None,
        stratify_col: str = 'product_type_name',
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        random_state: int = 42,
        min_samples_per_class: int = 3
    ):
    """
    Tries to split a dataset preserving 
    as much stratification as possible.
    Then saves the split to three parquet files.

    Args:
    min_samples_per_class: controls how crude stratification will be,
                           by default attemps finest stratification possible
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    if not abs(train_size + val_size + test_size - 1.0) < 1e-6:
        raise ValueError(f"Split sizes must add up to 1, got {train_size + val_size + test_size}")

    df = pd.read_parquet(manifest_path)
    print(f"Loaded {len(df)} samples from {manifest_path}")

    if stratify_col not in df.columns:
        raise ValueError(f"Column {stratify_col} not found")

    df['_stratify_col'] = df[stratify_col].copy()

    counts = df[stratify_col].value_counts()
    rare_classes = counts[counts < min_samples_per_class].index.tolist()
    if rare_classes:
        print(f"\nWarning: Some classes have less than {min_samples_per_class} samples:")
        print(rare_classes)
        print('Grouping them as "RARE_CLASS" for stratification')
        df.loc[df[stratify_col].isin(rare_classes), '_stratify_col'] = 'RARE_CLASS'

        new_counts = df['_stratify_col'].value_counts()
        problematic = new_counts[new_counts < min_samples_per_class].index.tolist()
        if problematic:
            print(f"Warning: Still have classes with < {min_samples_per_class} samples:")
            print(problematic)
            print("Falling back to random split (no stratification)")
            use_stratification = False
        else:
            use_stratification = True
    else:
        use_stratification = True

    if use_stratification:
        stratify_first = df['_stratify_col']
    else:
        stratify_first = None

    train_df, temp_df = train_test_split(
        df,
        train_size=train_size,
        stratify=stratify_first,
        random_state=random_state
    )

    if use_stratification:
        temp_counts = temp_df['_stratify_col'].value_counts()
        small_in_temp = temp_counts[temp_counts < 2].index.tolist()
        if small_in_temp:
            print(f"\nWarning: After first split, some classes have less than 2 samples in temp:")
            print(small_in_temp)
            print("Grouping them as 'SECOND_RARE_CLASS' for second split")
            temp_df['_second_stratify'] = temp_df['_stratify_col'].copy()
            temp_df.loc[temp_df['_stratify_col'].isin(small_in_temp), '_second_stratify'] = 'SECOND_RARE_CLASS'
            stratify_second = temp_df['_second_stratify']
        else:
            stratify_second = temp_df['_stratify_col']
    else:
        stratify_second = None

    val_ratio = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_ratio,
        stratify=stratify_second,
        random_state=random_state
    )

    print('\nSplit sizes:')
    print(f"  Train: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val: {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")

    output_path = Path(output_dir) if output_dir else manifest_path.parent
    output_path.mkdir(parents=True, exist_ok=True)

    base = manifest_path.stem
    train_path = output_path / f"{base}_train.parquet"
    val_path = output_path / f"{base}_val.parquet"
    test_path = output_path / f"{base}_test.parquet"

    cols = df.columns.difference(['_stratify_col', '_second_stratify']).tolist()
    train_df[cols].to_parquet(train_path, index=False)
    val_df[cols].to_parquet(val_path, index=False)
    test_df[cols].to_parquet(test_path, index=False)

    print('\nSplits saved to')
    print(f"Train: {train_path}")
    print(f"Val: {val_path}")
    print(f"Test: {test_path}")

    return train_df, val_df, test_df

