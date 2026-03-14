import argparse
import yaml
import os

def get_parser():
    parser = argparse.ArgumentParser(description='FahionCLIP fine-tuning manager')
    subparsers = parser.add_subparsers(dest='command', help='Commands', required=True)
    
    split_parser = subparsers.add_parser('split', help='Split dataset into train/val/test')
    split_parser.add_argument('manifest_path', type=str, help='Path to manifest parquet file')
    split_parser.add_argument('--output_dir', type=str, default=None, help='Output directory for splits')
    split_parser.add_argument('--stratify_col', type=str, default='product_type_name', help='Column to stratify on')
    split_parser.add_argument('--train_size', type=float, default=0.7, help='Training set proportion')
    split_parser.add_argument('--val_size', type=float, default=0.15, help='Validation set proportion')
    split_parser.add_argument('--test_size', type=float, default=0.15, help='Test set proportion')
    split_parser.add_argument('--random_state', type=int, default=42, help='Random seed')
    split_parser.add_argument('--min_samples_per_class', type=int, default=3, help='The least number of samples a class in stratify_col must contain')
    
    train_parser = subparsers.add_parser('train', help='Train FahionCLIP model')
    train_parser.add_argument('--params', type=str, default='params.yaml', help='Training params file')

    test_parser = subparsers.add_parser('test', help='Test FahionCLIP model')
    test_parser.add_argument('--params', type=str, default='params.yaml', help='Test params file')

    return parser

def load_params(params_path):
    with open(params_path) as f:
        params = yaml.safe_load(f)
    for split in params:
        if params[split]['dataloader']['num_workers'] == 'auto':
            params[split]['dataloader']['num_workers'] = min(os.cpu_count(), 8)
    return params