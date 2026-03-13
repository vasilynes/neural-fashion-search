from src.data import stratified_split
from src.train import train_routine 
from src.data import get_dataloader
from src.config import config
from src.cli import load_params
from src.data import HardNegativesBatchSampler, FashionDataset
from src.test import test_routine
import pandas as pd

def train_handler(args):
    params = load_params(args.params)

    train_dataset = FashionDataset.for_split('train', config)
    sampler = HardNegativesBatchSampler(
        dataset_size=len(train_dataset),
        batch_size=params['train']['dataloader']['batch_size'],
        hard_negatives_per_anchor=params['train']['training']['hard_negatives_per_anchor']
    )
    train_loader = get_dataloader(train_dataset, params['train']['dataloader'], 
                                augment=True, sampler=sampler)
    
    embed_loader_params = {**params['train']['dataloader'], 'shuffle': False}
    embed_loader = get_dataloader(train_dataset, embed_loader_params, augment=False)

    val_dataset = FashionDataset.for_split('val', config)
    val_loader = get_dataloader(val_dataset, params['val']['dataloader'], augment=False)

    train_routine(train_loader, embed_loader, val_loader, params['train'], sampler)

def test_handler(args):
    params = load_params(args.params)
    test_loaders = {}

    test_dataset = FashionDataset.for_split('test', config)
    test_loaders['full'] = get_dataloader(test_dataset, params['test']['dataloader'])

    test_df = pd.read_parquet(config.TEST_FILE)
    for col in params['test']['filtering_features']:
        unique_vals = test_df[col].dropna().replace('', 'unknown').unique()
        for val in unique_vals:
            filtered_dataset = test_dataset.filter(col, val)
            safe_name = f"{col}_{val.lower().replace(' ', '_')}"
            test_loaders[safe_name] = get_dataloader(filtered_dataset, params['test']['dataloader'])

    test_routine(test_loaders, params['test'])

def split_handler(args):
    """Handles stratified split of the dataset."""
    if not args.manifest_path:
        raise Exception("--manifest_path required for split command")
    print("Running data split...")
    stratified_split(
        manifest_path=args.manifest_path,
        output_dir=args.output_dir,
        stratify_col=args.stratify_col,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.random_state,
        min_samples_per_class=args.min_samples_per_class
    )
    print("Split completed")

command_handlers = {
    'split': split_handler,
    'train': train_handler,
    'test': test_handler
}