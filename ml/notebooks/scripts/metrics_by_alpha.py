# Full script to calculate search metrics on Colab

# After:
# drive.mount('/content/drive')
# sys.path.append('/content/drive/MyDrive/model')

# Install Colab specific prerequisites 
import subprocess
subprocess.run(['pip', 'install', 'fastembed', '-q'], check=True)
subprocess.run(['pip', 'install', 'qdrant-client', '-q'], check=True)

from search_engine import qdrant_setup
# Start Qdrant process in Colab and load DB from snapshot
proc = qdrant_setup.setup(
    snapshot_source='/content/drive/MyDrive/model/fashion_database-4540778058153887-2026-03-25-19-14-45.snapshot',
    snapshot_dest='/content/snapshots/'
)

import pandas as pd
from app.main import create_model_service
from search_engine import embed
from app.main import create_search_service
from search_engine import search
from search_engine import metrics
import json

# Read the cleaned dataframe with described queries
df = pd.read_parquet('/content/drive/MyDrive/model/data/queries/desc_query_clean.parquet')

# The colums containing queries
cols = ['typos', 'synonyms', 'attr_swaps', 'real_user']
# Where the code will save embeddings
save_dir = '/content/drive/MyDrive/model/data/queries/'

SNAPSHOT_PATH = "file:///content/snapshots/fashion_database-4540778058153887-2026-03-25-19-14-45.snapshot"
search_service = create_search_service(SNAPSHOT_PATH, model_service='_')

alphas = [.0, .25, .5, .75, 1]
alpha_metrics = {}
for a in alphas:
    # For different alphas, calculate what article_ids the queries retrieve 
    df = search.collect_search_res_alpha(
        search_service,
        df,
        cols,
        batch_size=16,
        embeds_dir=save_dir,
        alpha=a,
    )

    alpha_metrics[a] = {}
    for col in ['typos', 'synonyms', 'real_user']:
        hits, rrs = metrics.hits_and_rrs(
            df, col
        )
        alpha_metrics[a][f"{col}_hit_rate"] = sum(hits) / len(hits)
        alpha_metrics[a][f"{col}_mrr"] = sum(rrs) / len(rrs)
    # Attr swaps require different logic
    hits = metrics.failure_rate(df, col='attr_swaps')
    alpha_metrics[a]['attr_swaps_hit_rate'] = sum(hits) / len(hits)

with open('/content/drive/MyDrive/model/alpha_metrics.json', 'w') as f:
    json.dump(alpha_metrics, f)

proc.terminate()