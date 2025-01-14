import json
import pandas as pd
from tqdm import tqdm

from utils import readit_utils
from utils import corpora_loader

original_corpus_df, simplified_corpora_dfs = corpora_loader.load_corpora_dfs()

print('Starting original readit jobs...')
original_readit_jobs = []
for row in tqdm(original_corpus_df.to_dict(orient='records')):
  original_readit_jobs.append({
    'document': row['document'],
    'paragraph_index': row['paragraph_index'],
    'readit_jid': readit_utils.readit_start_job(row['original_text'])
  })

print('Saving original readit jobs...')
with open('simplified_corpora_with_metrics/original_readit_jobs.json', 'w') as f:
  json.dump(original_readit_jobs, f)

for CORPUS_NAME, corpus_df in simplified_corpora_dfs.items():
    print(f'Starting {CORPUS_NAME} readit jobs...')
    simplified_readit_jobs = []
    for row in tqdm(corpus_df.to_dict(orient='records')):
      simplified_readit_jobs.append({
        'document': row['document'],
        'paragraph_index': row['paragraph_index'],
        'readit_jid': readit_utils.readit_start_job(row['simplified_text'])
      })
    
    print(f'Saving {CORPUS_NAME} readit jobs...')
    with open(f'simplified_corpora_with_metrics/{CORPUS_NAME}_readit_jobs.json', 'w') as f:
      json.dump(simplified_readit_jobs, f)